#include "importance_map.hpp"
#include "util/log.hpp"

namespace mufflon::renderer::decimaters::importance {

ImportanceMap::ImportanceMap(std::vector<scene::geometry::PolygonMeshType*> meshes) :
	m_meshes(std::move(meshes)),
	m_importanceSums(m_meshes.size(), 0.0)
{
	m_normalizedImportance.reserve(m_meshes.size());
	m_collapseHistory.reserve(m_meshes.size());
	m_vertexOffsets.reserve(m_meshes.size());
	m_meshAreas.reserve(m_meshes.size());

	for(auto* mesh : m_meshes) {
		m_normalizedImportance.emplace_back();
		m_collapseHistory.emplace_back();
		m_vertexOffsets.push_back(m_totalVertexCount);
		m_meshAreas.emplace_back();

		if(mesh != nullptr) {
			if(!mesh->get_property_handle(m_normalizedImportance.back(), std::string(NORMALIZED_IMPORTANCE_PROP_NAME)))
				mesh->add_property(m_normalizedImportance.back(), std::string(NORMALIZED_IMPORTANCE_PROP_NAME));
			if (!mesh->get_property_handle(m_collapseHistory.back(), std::string(COLLAPSE_HISTORY_PROP_NAME)))
				mesh->add_property(m_collapseHistory.back(), std::string(COLLAPSE_HISTORY_PROP_NAME));
			m_totalVertexCount += static_cast<u32>(mesh->n_vertices());

			// Compute mesh area
			double area = 0.0;
			for (auto face : mesh->faces()) {
				if(face.is_valid()) {
					auto circIter = mesh->cfv_ccwbegin(face);
					if(circIter.is_valid()) {
						mAssertMsg(std::distance(circIter, mesh->cfv_ccwend(face)) == 3u,
								   "Area formula only valid for triangles");
						OpenMesh::Vec3f a = mesh->point(*circIter);
						OpenMesh::Vec3f b = mesh->point(*(++circIter));
						OpenMesh::Vec3f c = mesh->point(*(++circIter));
						area += ((a - b) % (c - b)).length();
					}
				}
			}
			m_meshAreas.back() = 0.5 * area;
		}
	}

	m_importance = std::make_unique<std::atomic<float>[]>(m_totalVertexCount);

	this->reset();
}

ImportanceMap::~ImportanceMap() {
	this->clear();
}

void ImportanceMap::clear() {
	for (std::size_t i = 0u; i < m_meshes.size(); ++i) {
		m_normalizedImportance[i].invalidate();
		m_collapseHistory[i].invalidate();
	}
	m_meshes.clear();
}

void ImportanceMap::reset() {
	for(auto& sum : m_importanceSums)
		sum = 0.0;
	
	for(std::size_t i = 0u; i < m_meshes.size(); ++i) {
		if(m_meshes[i] != nullptr) {
			auto& mesh = *m_meshes[i];

			for(std::size_t v = 0u; v < mesh.n_vertices(); ++v)
				m_importance[i].store(0.f);
			if(m_normalizedImportance[i].is_valid())
				for(u32 v = 0u; v < static_cast<u32>(mesh.n_vertices()); ++v)
					mesh.property(m_normalizedImportance[i], mesh.vertex_handle(v)) = 0.f;
			// TODO: reset collapse history?
		}
	}
}

void ImportanceMap::update_normalized() {
	u32 vertexIndex = 0u;

	for(std::size_t i = 0u; i < m_meshes.size(); ++i) {
		m_importanceSums[i] = 0.0;
		if(m_meshes[i] == nullptr || !m_normalizedImportance[i].is_valid())
			continue;

		auto& mesh = *m_meshes[i];
		for(u32 v = 0u; v < mesh.n_vertices(); ++v) {
			if(mesh.has_vertex_status() && mesh.status(mesh.vertex_handle(v)).deleted())
				continue;

			const float importance = m_importance[vertexIndex].load();
			mAssert(!isnan(importance));

			m_importanceSums[i] += importance;
			float area = 0.f;

			const auto vh = mesh.vertex_handle(v);
			auto circIter = mesh.cvv_ccwbegin(vh);
			if(circIter.is_valid()) {
				const OpenMesh::Vec3f center = mesh.point(vh);
				OpenMesh::Vec3f a = mesh.point(*circIter);
				OpenMesh::Vec3f b;
				// We assume that we only have triangles!
				for(; circIter != mesh.cvv_ccwend(vh); ++circIter) {
					b = a;
					a = mesh.point(*circIter);
					area += ((a - center) % (b - center)).length();
				}

				mAssertMsg(area != 0.f, "Degenerated vertex");
				mesh.property(m_normalizedImportance[i], vh) = 2.f * importance / area;
			} else {
				//logWarning("Invalid circular iterator for mesh ", i, ", vertex ", v);
				mesh.property(m_normalizedImportance[i], vh) = 0.f;
			}
			++vertexIndex;
		}
	}
}

void ImportanceMap::add(u32 meshIndex, u32 localIndex, float val) {
	mAssert(meshIndex < static_cast<u32>(m_meshes.size()));
	mAssert(localIndex < static_cast<u32>(m_meshes[meshIndex]->n_vertices()));
	const u32 vertexIndex = m_vertexOffsets[meshIndex] + localIndex;
	mAssert(vertexIndex < m_totalVertexCount);
	float expected = m_importance[vertexIndex].load();
	float desired;
	do {
		desired = expected + val;
	} while(!m_importance[vertexIndex].compare_exchange_weak(expected, desired));
}

void ImportanceMap::collapse(u32 meshIndex, u32 localFrom, u32 localTo) {
	mAssert(meshIndex < static_cast<u32>(m_meshes.size()));
	mAssert(localFrom < static_cast<u32>(m_meshes[meshIndex]->n_vertices()));
	mAssert(localTo < static_cast<u32>(m_meshes[meshIndex]->n_vertices()));
	const u32 vertexFrom = m_vertexOffsets[meshIndex] + localFrom;
	const u32 vertexTo = m_vertexOffsets[meshIndex] + localTo;

	float expected = m_importance[vertexTo].load();
	float desired;
	do {
		desired = expected + m_importance[vertexFrom].load();
	} while(!m_importance[vertexTo].compare_exchange_weak(expected, desired));
}

} // namespace mufflon::renderer::decimaters::importance