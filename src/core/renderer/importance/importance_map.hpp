#pragma once

#include "util/assert.hpp"
#include "util/int_types.hpp"
#include "util/string_view.hpp"
#include "core/scene/geometry/polygon_mesh.hpp"
#include <atomic>
#include <memory>
#include <vector>

namespace mufflon::renderer {

// Keeps importance information on a per-mesh basis
class ImportanceMap {
public:
	static constexpr StringView NORMALIZED_IMPORTANCE_PROP_NAME = "normImp";
	static constexpr StringView COLLAPSE_HISTORY_PROP_NAME = "collHist";

	ImportanceMap() = default;
	ImportanceMap(std::vector<scene::geometry::PolygonMeshType*> meshes);
	~ImportanceMap();

	ImportanceMap(const ImportanceMap&) = delete;
	ImportanceMap(ImportanceMap&&);
	ImportanceMap& operator=(const ImportanceMap&) = delete;
	ImportanceMap& operator=(ImportanceMap&&);

	void reset();
	void update_normalized();

	float& normalized(u32 meshIndex, u32 localIndex) {
		mAssert(meshIndex < static_cast<u32>(m_meshes.size()));
		mAssert(localIndex < static_cast<u32>(m_meshes[meshIndex]->n_vertices()));
		mAssert(m_normalizedImportance[meshIndex].is_valid());
		auto& mesh = *m_meshes[meshIndex];
		return mesh.property(m_normalizedImportance[meshIndex], mesh.vertex_handle(localIndex));
	}
	const float& normalized(u32 meshIndex, u32 localIndex) const {
		mAssert(meshIndex < static_cast<u32>(m_meshes.size()));
		mAssert(localIndex < static_cast<u32>(m_meshes[meshIndex]->n_vertices()));
		mAssert(m_normalizedImportance[meshIndex].is_valid());
		const auto& mesh = *m_meshes[meshIndex];
		return mesh.property(m_normalizedImportance[meshIndex], mesh.vertex_handle(localIndex));
	}

	void add(u32 mesh, u32 localIndex, float val);

	u32 size() const noexcept { return m_totalVertexCount; }
	u32 get_vertex_count(u32 mesh) const {
		mAssert(mesh < static_cast<u32>(m_meshes.size()));
		if(m_meshes[mesh] == nullptr)
			return 0u;
		else if(mesh + 1u == m_meshes.size())
			return m_totalVertexCount - m_vertexOffsets[mesh];
		else
			return m_vertexOffsets[mesh + 1u] - m_vertexOffsets[mesh];
	}
	double get_importance_sum(u32 mesh) const {
		mAssert(mesh < static_cast<u32>(m_meshes.size()));
		return m_importanceSums[mesh];
	}

	OpenMesh::VPropHandleT<float> get_importance_property(u32 meshIndex) const {
		mAssert(meshIndex < m_meshes.size());
		return m_normalizedImportance[meshIndex];
	}

private:
	std::vector<scene::geometry::PolygonMeshType*> m_meshes;
	std::vector<OpenMesh::VPropHandleT<float>> m_normalizedImportance;
	std::vector<OpenMesh::VPropHandleT<u32>> m_collapseHistory;
	std::vector<u32> m_vertexOffsets;
	u32 m_totalVertexCount = 0u;
	std::unique_ptr<std::atomic<float>[]> m_importance;
	std::vector<double> m_importanceSums;
};

} // namespace mufflon::renderer 