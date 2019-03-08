#include "sil_imp_map.hpp"
#include "util/log.hpp"
#include "util/punning.hpp"

namespace mufflon::renderer::silhouette {

namespace {

template < class T >
void atomic_add(std::atomic<T>& atom, const T& val) {
	float expected = atom.load();
	float desired;
	do {
		desired = expected + val;
	} while(!atom.compare_exchange_weak(expected, desired));
}

// Computes the area surrounding a vertex (only non-deleted vertices)
float compute_vertex_area(const scene::geometry::PolygonMeshType& mesh, const OpenMesh::VertexHandle& vertex) {
	float area = 0.f;

	auto circIter = mesh.cvv_ccwbegin(vertex);
	while(!circIter.is_valid() && circIter != mesh.cvv_ccwend(vertex))
		++circIter;

	if(circIter.is_valid() && circIter != mesh.cvv_ccwend(vertex)) {
		const OpenMesh::Vec3f center = mesh.point(vertex);
		OpenMesh::Vec3f a = mesh.point(*circIter);
		OpenMesh::Vec3f b;
		// We assume that we only have triangles!
		for(; circIter != mesh.cvv_ccwend(vertex); ++circIter) {
			if(circIter.is_valid()) {
				b = a;
				a = mesh.point(*circIter);
				area += ((a - center) % (b - center)).length();
			}
		}

		mAssertMsg(area != 0.f, "Degenerated vertex");
		area *= 0.5f;
	}

	return area;
}

} // namespace 

ImportanceMap::ImportanceMap(std::vector<scene::geometry::PolygonMeshType*> meshes) :
	m_meshes(std::move(meshes)),
	m_importanceSums(m_meshes.size(), 0.0) {
	m_importanceDensity.reserve(m_meshes.size());
	m_collapseHistory.reserve(m_meshes.size());
	m_vertexOffsets.reserve(m_meshes.size());
	m_meshAreas.reserve(m_meshes.size());

	for(auto* mesh : m_meshes) {
		m_importanceDensity.emplace_back();
		m_collapseHistory.emplace_back();
		m_vertexOffsets.push_back(m_totalVertexCount);
		m_meshAreas.emplace_back();

		if(mesh != nullptr) {
			if(!mesh->get_property_handle(m_importanceDensity.back(), std::string(NORMALIZED_IMPORTANCE_PROP_NAME)))
				mesh->add_property(m_importanceDensity.back(), std::string(NORMALIZED_IMPORTANCE_PROP_NAME));
			if(!mesh->get_property_handle(m_collapseHistory.back(), std::string(COLLAPSE_HISTORY_PROP_NAME)))
				mesh->add_property(m_collapseHistory.back(), std::string(COLLAPSE_HISTORY_PROP_NAME));
			m_totalVertexCount += static_cast<u32>(mesh->n_vertices());

			// Compute mesh area
			double area = 0.0;
			for(auto face : mesh->faces()) {
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
	for(std::size_t i = 0u; i < m_meshes.size(); ++i) {
		m_importanceDensity[i].invalidate();
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
			if(m_importanceDensity[i].is_valid())
				for(u32 v = 0u; v < static_cast<u32>(mesh.n_vertices()); ++v)
					mesh.property(m_importanceDensity[i], mesh.vertex_handle(v)) = 0.f;
			// TODO: reset collapse history?
		}
	}
}

void ImportanceMap::update_normalized() {
	u32 vertexIndex = 0u;

	for(std::size_t i = 0u; i < m_meshes.size(); ++i) {
		m_importanceSums[i] = 0.0;
		if(m_meshes[i] == nullptr || !m_importanceDensity[i].is_valid())
			continue;

		auto& mesh = *m_meshes[i];
		for(u32 v = 0u; v < mesh.n_vertices(); ++v) {
			if(mesh.has_vertex_status() && mesh.status(mesh.vertex_handle(v)).deleted())
				continue;

			const float importance = m_importance[vertexIndex].load();
			mAssert(!isnan(importance));

			m_importanceSums[i] += importance;

			const auto vh = mesh.vertex_handle(v);
			mesh.property(m_importanceDensity[i], vh) = importance / compute_vertex_area(mesh, vh);
			++vertexIndex;
		}
	}
}

double ImportanceMap::get_importance_density_sum() const noexcept {
	double sum = 0.0;
	for(std::size_t i = 0u; i < m_meshes.size(); ++i) {
		if(m_meshes[i] != nullptr)
			sum += (m_meshAreas[i] != 0.0) ? m_importanceSums[i] / m_meshAreas[i] : 0.0;
	}
	return sum;
}

u32 ImportanceMap::get_not_deleted_vertex_count() const noexcept {
	u32 count = 0u;
	for(std::size_t i = 0u; i < m_meshes.size(); ++i) {
		if(m_meshes[i] != nullptr) {
			for(u32 v = 0u; v < m_meshes[i]->n_vertices(); ++v) {
				if(!m_meshes[i]->has_vertex_status() || !m_meshes[i]->status(m_meshes[i]->vertex_handle(v)).deleted())
					++count;
			}
		}
	}
	return count;
}

const ImportanceMap::CollapseEvent& ImportanceMap::get_collapse_event(const u32 meshIndex, const OpenMesh::VertexHandle& vertex) const {
	mAssert(meshIndex < static_cast<u32>(m_meshes.size()));
	mAssert(m_meshes[meshIndex] != nullptr);
	mAssert(vertex.is_valid());
	mAssert(static_cast<std::size_t>(vertex.idx()) < m_meshes[meshIndex]->n_vertices());
	return m_meshes[meshIndex]->property(m_collapseHistory[meshIndex], vertex);
}

void ImportanceMap::record_vertex_contribution(const u32 meshIndex, const u32 localIndex, const float importance) {
	const u32 vertexIndex = m_vertexOffsets[meshIndex] + localIndex;
	atomic_add(m_importance[vertexIndex], importance);
}

void ImportanceMap::record_face_contribution(const u32 meshIndex, const u32* vertexIndices, const u32 vertexCount,
										const u32 vertexOffset, const u32 primId, const ei::Vec3& hitpoint,
										const float importance) {
	// Importance gets added to every vertex of the triangle - weighted by the squared distance
	// First we need to compute the sum - to speed up the code we assume only triangles
	mAssert(meshIndex < static_cast<u32>(m_meshes.size()));
	mAssert(m_meshes[meshIndex] != nullptr);
	const auto& mesh = *m_meshes[meshIndex];

	float distSqrSum = 0.f;
	for(u32 v = 0u; v < vertexCount; ++v) {
		const u32 localIndex = vertexIndices[vertexOffset + vertexCount * primId + v];
		distSqrSum += ei::lensq(hitpoint - util::pun<ei::Vec3>(mesh.point(mesh.vertex_handle(localIndex))));
	}
	const float distSqrSumInv = 1.f / distSqrSum;

	// Now do the actual attribution
	for(u32 v = 0u; v < vertexCount; ++v) {
		const u32 localIndex = vertexIndices[vertexOffset + vertexCount * primId + v];
		const u32 vertexIndex = localIndex + m_vertexOffsets[meshIndex];

		const float distSqr = ei::lensq(hitpoint - util::pun<ei::Vec3>(mesh.point(mesh.vertex_handle(localIndex))));
		const float weightedImportance = importance * distSqr * distSqrSumInv;
		atomic_add(m_importance[vertexIndex], weightedImportance);
	}
}

void ImportanceMap::collapse(const u32 meshIndex, const OpenMesh::Decimater::CollapseInfoT<scene::geometry::PolygonMeshType>& ci) {
	mAssert(meshIndex < static_cast<u32>(m_meshes.size()));
	mAssert(static_cast<std::size_t>(ci.v0.idx()) < static_cast<u32>(m_meshes[meshIndex]->n_vertices()));
	mAssert(static_cast<std::size_t>(ci.v1.idx()) < static_cast<u32>(m_meshes[meshIndex]->n_vertices()));
	const u32 vertexFrom = m_vertexOffsets[meshIndex] + static_cast<u32>(ci.v0.idx());
	const u32 vertexTo = m_vertexOffsets[meshIndex] + static_cast<u32>(ci.v1.idx());

	auto& mesh = *m_meshes[meshIndex];
	const ei::Vec3 fromPos = util::pun<ei::Vec3>(mesh.point(ci.v0));

	// Instead of simple addition, we spread the importance weighted by distance^2 to neighboring vertices
	// To easily adopt the 

	// For that we need to sum the distances first
	float distSqrSum = 0.f;
	for(auto circIter = mesh.cvv_ccwbegin(ci.v0); circIter != mesh.cvv_ccwend(ci.v0); ++circIter) {
		if(circIter.is_valid())
			distSqrSum += ei::lensq(fromPos - util::pun<ei::Vec3>(mesh.point(*circIter)));
	}
	const float distSqrSumInv = 1.f / distSqrSum;

	// Now we can actually distribute the importance
	const float importance = m_importance[vertexFrom].load();
	for(auto circIter = mesh.cvv_ccwbegin(ci.v0); circIter != mesh.cvv_ccwend(ci.v0); ++circIter) {
		if(circIter.is_valid()) {
			const float distSqr = ei::lensq(fromPos - util::pun<ei::Vec3>(mesh.point(*circIter)));
			const float weightedImportance = distSqr * distSqrSumInv * importance;
			const u32 vertexIndex = m_vertexOffsets[meshIndex] + circIter->idx();
			mAssert(vertexIndex < m_meshes[meshIndex]->n_vertices());
			atomic_add(m_importance[vertexIndex], weightedImportance);

			// Recompute the density
			const float area = compute_vertex_area(mesh, *circIter);
			mesh.property(m_importanceDensity[meshIndex], *circIter) = m_importance[vertexIndex].load() / area;
		}
	}

	// Save the collapse in the vertex's history
	// Since we cannot reconstruct the halfedge from a vertex handle, we instead store the halfedge handle
	mesh.property(m_collapseHistory[meshIndex], ci.v0) = CollapseEvent{ ci.v1, ci.vl, ci.vr };
}

void ImportanceMap::uncollapse(const u32 meshIndex, const OpenMesh::VertexHandle& vertex) {
	mAssert(meshIndex < static_cast<u32>(m_meshes.size()));
	mAssert(m_meshes[meshIndex] != nullptr);
	mAssert(vertex.is_valid());
	mAssert(static_cast<std::size_t>(vertex.idx()) < m_meshes[meshIndex]->n_vertices());

	// Restore the importance of the (now no longer deleted) vertex
	// TODO
}

} // namespace mufflon::renderer::silhouette