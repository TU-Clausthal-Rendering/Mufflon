#pragma once

#include "util/int_types.hpp"
#include "core/renderer/decimaters/octree/octree.hpp"
#include "core/scene/geometry/polygon_mesh.hpp"
#include <ei/3dtypes.hpp>
#include <ei/vector.hpp>
#include <optional>

namespace mufflon::scene::clustering {

template < class O >
class OctreeVertexClusterer {
public:
	using OctreeType = O;

	OctreeVertexClusterer(const OctreeType& octree, const std::size_t maxCount,
						  std::optional<float> maxDensity = std::nullopt) :
		m_octree{ octree },
		m_maxCount{ maxCount },
		m_maxDensity{ maxDensity },
		m_collapsedTo{}
	{}

	void enable_collapse_history(OpenMesh::VPropHandleT<OpenMesh::VertexHandle> handle) {
		m_collapsedTo = handle;
	}

	// Performs the clustering. Note that, if garbageCollect == false, you 
	// MUST request status for vertices, edges, and faces prior
	std::size_t cluster(geometry::PolygonMeshType& mesh, const ei::Box& aabb,
						const bool garbageCollect = false, std::vector<bool>* octreeNodeMask = nullptr,
						std::vector<typename O::NodeIndex>* currLevel = nullptr,
						std::vector<typename O::NodeIndex>* nextLevel = nullptr);

private:
	const OctreeType& m_octree;
	const std::size_t m_maxCount;
	const std::optional<float> m_maxDensity;
	OpenMesh::VPropHandleT<OpenMesh::VertexHandle> m_collapsedTo;
};

} // namespace mufflon::scene::clustering