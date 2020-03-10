#pragma once

#include "util/int_types.hpp"
#include "core/renderer/decimaters/util/octree.hpp"
#include "core/scene/geometry/polygon_mesh.hpp"
#include <ei/3dtypes.hpp>
#include <ei/vector.hpp>

namespace mufflon::scene::clustering {

template < class O >
class OctreeVertexClusterer {
public:
	using OctreeType = O;

	OctreeVertexClusterer(const OctreeType& octree, const std::size_t maxDepth,
						  const std::size_t maxCount) :
		m_octree{ octree },
		m_maxDepth{ maxDepth },
		m_maxCount{ maxCount }
	{}

	std::size_t cluster(geometry::PolygonMeshType& mesh, const ei::Box& aabb);

private:
	const OctreeType& m_octree;
	const std::size_t m_maxDepth;
	const std::size_t m_maxCount;
};

} // namespace mufflon::scene::clustering