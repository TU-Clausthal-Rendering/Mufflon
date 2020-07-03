#pragma once

#include "util/int_types.hpp"
#include "core/scene/geometry/polygon_mesh.hpp"
#include <ei/3dtypes.hpp>
#include <ei/vector.hpp>

namespace mufflon::scene::clustering {

class UniformVertexClusterer {
public:
	UniformVertexClusterer(ei::UVec3 gridRes) :
		m_gridRes{ gridRes },
		m_collapsedTo{} {}

	void enable_collapse_history(OpenMesh::VPropHandleT<OpenMesh::VertexHandle> handle) {
		m_collapsedTo = handle;
	}

	std::size_t cluster(geometry::PolygonMeshType& mesh, const ei::Box& aabb,
						const bool garbageCollect = false);

private:
	ei::UVec3 m_gridRes;
	OpenMesh::VPropHandleT<OpenMesh::VertexHandle> m_collapsedTo;
};

} // namespace mufflon::scene::clustering