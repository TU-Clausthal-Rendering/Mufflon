#include "importance_map.hpp"
#include "util/log.hpp"

namespace mufflon::renderer {

ImportanceMap::ImportanceMap(std::vector<scene::geometry::PolygonMeshType*> meshes) :
	m_meshes(std::move(meshes)),
	m_importanceSums(m_meshes.size(), 0.0)
{
	m_normalizedImportance.reserve(m_meshes.size());
	m_collapseHistory.reserve(m_meshes.size());
	m_vertexOffsets.reserve(m_meshes.size());

	for(auto* mesh : m_meshes) {
		m_normalizedImportance.emplace_back();
		m_collapseHistory.emplace_back();
		m_vertexOffsets.push_back(m_totalVertexCount);

		if(mesh != nullptr) {
			mesh->add_property(m_normalizedImportance.back(), std::string(NORMALIZED_IMPORTANCE_PROP_NAME));
			mesh->add_property(m_collapseHistory.back(), std::string(COLLAPSE_HISTORY_PROP_NAME));
			m_totalVertexCount += static_cast<u32>(mesh->n_vertices());
		}
	}

	m_importance = std::make_unique<std::atomic<float>[]>(m_totalVertexCount);

	this->reset();
}

ImportanceMap::~ImportanceMap() {
	for(std::size_t i = 0u; i < m_meshes.size(); ++i) {
		if(m_meshes[i] != nullptr) {
			auto& mesh = *m_meshes[i];

			if(m_normalizedImportance[i].is_valid())
				mesh.remove_property(m_normalizedImportance[i]);
			if(m_collapseHistory[i].is_valid())
				mesh.remove_property(m_collapseHistory[i]);
		}
	}
}

ImportanceMap::ImportanceMap(ImportanceMap&& map) :
	m_meshes(std::move(map.m_meshes)),
	m_normalizedImportance(std::move(map.m_normalizedImportance)),
	m_collapseHistory(std::move(map.m_collapseHistory)),
	m_vertexOffsets(std::move(map.m_vertexOffsets)),
	m_totalVertexCount(map.m_totalVertexCount),
	m_importance(std::move(map.m_importance)),
	m_importanceSums(std::move(map.m_importanceSums))
{}

ImportanceMap& ImportanceMap::operator=(ImportanceMap&& map) {
	std::swap(m_meshes, map.m_meshes);
	std::swap(m_normalizedImportance, map.m_normalizedImportance);
	std::swap(m_collapseHistory, map.m_collapseHistory);
	std::swap(m_vertexOffsets, map.m_vertexOffsets);
	std::swap(m_totalVertexCount, map.m_totalVertexCount);
	std::swap(m_importance, map.m_importance);
	std::swap(m_importanceSums, map.m_importanceSums);
	return *this;
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
				for(; circIter != mesh.cvv_ccwend(vh); ++circIter) {
					b = a;
					a = mesh.point(*circIter);
					area += ((a - center) % (b - center)).length();
				}

				mAssertMsg(area != 0.f, "Degenerated vertex");
				mesh.property(m_normalizedImportance[i], vh) = importance / area;
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

} // namespace mufflon::renderer