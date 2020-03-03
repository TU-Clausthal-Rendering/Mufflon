#pragma once

#include "util/int_types.hpp"
#include "core/scene/geometry/polygon_mesh.hpp"
#include <ei/3dtypes.hpp>
#include <ei/vector.hpp>

namespace mufflon::scene::clustering {

class UniformVertexClusterer {
public:
	UniformVertexClusterer(ei::UVec3 gridRes) :
		m_gridRes{ gridRes } {}

	std::size_t cluster(geometry::PolygonMeshType& mesh, const ei::Box& aabb);

private:
	ei::UVec3 m_gridRes;
};

} // namespace mufflon::scene::clustering