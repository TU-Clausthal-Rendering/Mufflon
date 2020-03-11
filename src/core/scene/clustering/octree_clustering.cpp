#include "octree_clustering.hpp"
#include "util.hpp"
#include "util/log.hpp"
#include "core/renderer/decimaters/util/octree.inl"
#include <OpenMesh/Core/Geometry/QuadricT.hh>

namespace mufflon::scene::clustering {

namespace {

template < class Iter >
u32 compute_cluster_center(const Iter begin, const Iter end) {
	u32 clusterCount = 0u;
	for(auto iter = begin; iter != end; ++iter) {
		auto& cluster = *iter;
		if(cluster.count > 0) {
			// Attempt to compute the optimal contraction point by inverting the quadric matrix
			const auto q = cluster.q;
			const ei::Mat4x4 w{
				q.a(), q.b(), q.c(), q.d(),
				q.b(), q.e(), q.f(), q.g(),
				q.c(), q.f(), q.h(), q.i(),
				0,	   0,	  0,	 1
			};
			const auto inverse = invert_opt(w);
			if(inverse.has_value())
				cluster.posAccum = ei::Vec3{ inverse.value() * ei::Vec4{ 0.f, 0.f, 0.f, 1.f } };
			else
				cluster.posAccum /= static_cast<float>(cluster.count);
			clusterCount += 1;
		}
	}
	return clusterCount;
}

template < class O >
std::pair<std::vector<bool>, u32> find_clusters(const O& octree, geometry::PolygonMeshType& mesh, const ei::Box& aabb,
												const std::size_t maxCount) {
	using NodeIndex = typename O::NodeIndex;

	// Perform a breadth-first search to bring clusters down in equal levels
	std::vector<bool> octreeNodeMask(octree.capacity(), false);
	std::vector<NodeIndex> currLevel;
	std::vector<NodeIndex> nextLevel;
	currLevel.reserve(static_cast<std::size_t>(ei::log2(std::max(maxCount, 1llu))));
	nextLevel.reserve(static_cast<std::size_t>(ei::log2(std::max(maxCount, 1llu))));
	std::size_t finalNodeCount = 0u;

	currLevel.push_back(octree.root_index());
	u32 maxIndex = currLevel.back().index;
	while(finalNodeCount < maxCount && !currLevel.empty()) {
		nextLevel.clear();

		// First, count how many new clusters we'd get from this level
		std::size_t count = 0u;
		for(const auto& cluster : currLevel) {
			if(!octree.children(cluster).has_value())
				count += 1u;
		}

		if(finalNodeCount + count > maxCount) {
			// We have to prioritize the nodes we add
			std::vector<std::pair<NodeIndex, float>> samples;
			samples.reserve(count);
			for(const auto& cluster : currLevel) {
				if(!octree.children(cluster).has_value())
					samples.emplace_back(cluster, octree.get_samples(cluster));
			}
			std::sort(samples.begin(), samples.end(), [](const auto& a, const auto& b) { return a.second < b.second; });
			const auto remainingCount = maxCount - finalNodeCount;
			for(std::size_t i = 0u; i < remainingCount; ++i) {
				octreeNodeMask[samples[i].first.index] = true;
				maxIndex = std::max(maxIndex, samples[i].first.index);
			}
			finalNodeCount = maxCount;
		} else {
			// No issue, we can add the entire level
			for(const auto& cluster : currLevel) {
				const auto children = octree.children(cluster);
				if(children.has_value()) {
					for(const auto& c : children.value())
						nextLevel.push_back(c);
				} else {
					octreeNodeMask[cluster.index] = true;
					maxIndex = std::max(maxIndex, cluster.index);
				}
			}
			finalNodeCount += count;
		}
		std::swap(currLevel, nextLevel);
	}

	return std::make_pair(octreeNodeMask, maxIndex);
}

template < class O >
std::pair<std::vector<bool>, u32> find_clusters_with_max_density(const O& octree, geometry::PolygonMeshType& mesh, const ei::Box& aabb,
																 const std::size_t maxCount, const float maxDensity) {
	using NodeIndex = typename O::NodeIndex;

	// Perform a breadth-first search to bring clusters down in equal levels
	std::vector<bool> octreeNodeMask(octree.capacity(), false);
	std::vector<NodeIndex> currLevel;
	std::vector<NodeIndex> nextLevel;
	currLevel.reserve(static_cast<std::size_t>(ei::log2(std::max(maxCount, 1llu))));
	nextLevel.reserve(static_cast<std::size_t>(ei::log2(std::max(maxCount, 1llu))));
	std::size_t finalNodeCount = 0u;

	currLevel.push_back(octree.root_index());
	u32 maxIndex = currLevel.back().index;
	while(finalNodeCount < maxCount && !currLevel.empty()) {
		nextLevel.clear();

		// First, count how many new clusters we'd get from this level
		std::size_t count = 0u;
		for(const auto& cluster : currLevel) {
			if(!octree.children(cluster).has_value())
				count += 1u;
		}

		if(finalNodeCount + count > maxCount) {
			// We have to prioritize the nodes we add
			std::vector<std::pair<NodeIndex, float>> samples;
			samples.reserve(count);
			for(const auto& cluster : currLevel) {
				if(const auto children = octree.children(cluster); children.has_value()) {
					// Since it may be that we omit leafs due to density we still track the children
					for(const auto& c : children.value())
						nextLevel.push_back(c);
				} else {
					const auto volume = octree.get_inverse_cell_volume(cluster);
					const auto sample = octree.get_samples(cluster);
					const auto density = sample / volume;
					if(density < maxDensity)
						samples.emplace_back(cluster, octree.get_samples(cluster));
				}
			}
			std::sort(samples.begin(), samples.end(), [](const auto& a, const auto& b) { return a.second < b.second; });
			const auto remainingCount = std::min(maxCount - finalNodeCount, samples.size());
			for(std::size_t i = 0u; i < remainingCount; ++i) {
				octreeNodeMask[samples[i].first.index] = true;
				maxIndex = std::max(maxIndex, samples[i].first.index);
			}
			finalNodeCount += samples.size();
		} else {
			// No issue, we can add the entire level
			for(const auto& cluster : currLevel) {
				if(const auto children = octree.children(cluster); children.has_value()) {
					for(const auto& c : children.value())
						nextLevel.push_back(c);
				} else {
					// Check if the density is low enough
					const auto volume = octree.get_inverse_cell_volume(cluster);
					const auto sample = octree.get_samples(cluster);
					const auto density = sample / volume;
					if(density < maxDensity) {
						octreeNodeMask[cluster.index] = true;
						maxIndex = std::max(maxIndex, cluster.index);
						finalNodeCount += 1u;
					}
				}
			}
		}
		std::swap(currLevel, nextLevel);
	}
	return std::make_pair(octreeNodeMask, maxIndex);
}

} // namespace

struct VertexCluster {
	ei::Vec3 posAccum{ 0.f };
	u32 count{ 0u };
	OpenMesh::Geometry::Quadricf q{};

	void add_vertex(const ei::Vec3& pos, const OpenMesh::Geometry::Quadricf& quadric) noexcept {
		count += 1u;
		posAccum += pos;
		q += quadric;
	}
};

template < class O >
std::size_t OctreeVertexClusterer<O>::cluster(geometry::PolygonMeshType& mesh, const ei::Box& aabb,
											  const bool garbageCollect) {
	const auto [octreeNodeMask, maxIndex] = m_maxDensity.has_value()
		? find_clusters_with_max_density(m_octree, mesh, aabb, m_maxCount, m_maxDensity.value())
		: find_clusters(m_octree, mesh, aabb, m_maxCount);

	// We have to track a few things per cluster
	// TODO: better bound!
	std::vector<VertexCluster> clusters(maxIndex + 1u);

	const auto aabbMin = aabb.min;
	const auto aabbDiag = aabb.max - aabb.min;
	// Convenience function to compute the cluster index from a position
	auto get_cluster_index = [this, &octreeNodeMask](const ei::Vec3& pos) -> std::optional<typename O::NodeIndex> {
		return m_octree.get_node_index(pos, octreeNodeMask);
	};
	auto get_cluster_index_no_stop = [this](const ei::Vec3& pos) -> typename O::NodeIndex {
		return m_octree.get_node_index(pos);
	};

	OpenMesh::VPropHandleT<OpenMesh::Geometry::Quadricf> quadricProps{};
	mesh.add_property(quadricProps);
	if(!quadricProps.is_valid())
		throw std::runtime_error("failed to add error quadric property");
	// compute the error quadrics for each vertex
	compute_error_quadrics(mesh, quadricProps);

	// For each vertex, determine the cluster it belongs to and update its statistics
	for(const auto vertex : mesh.vertices()) {
		const auto pos = util::pun<ei::Vec3>(mesh.point(vertex));
		const auto clusterIndex = get_cluster_index(pos);
		if(clusterIndex.has_value())
			clusters[clusterIndex->index].add_vertex(pos, mesh.property(quadricProps, vertex));
	}
	mesh.remove_property(quadricProps);

	// Calculate the representative cluster position for every cluster
	u32 clusterCount = compute_cluster_center(clusters.begin(), clusters.end());

	std::vector<geometry::PolygonMeshType::VertexHandle> representative(clusters.size());
	// then we set the position of every vertex to that of its
	// cluster representative
	for(const auto vertex : mesh.vertices()) {
		const auto pos = util::pun<ei::Vec3>(mesh.point(vertex));
		const auto clusterIndex = get_cluster_index(pos);
		if(clusterIndex.has_value()) {
			// new cluster position has been previously computed
			const auto newPos = clusters[clusterIndex->index].posAccum;
			mesh.point(vertex) = util::pun<geometry::PolygonMeshType::Point>(newPos);
			representative[clusterIndex->index] = vertex;
		}
	}

	std::vector<geometry::PolygonMeshType::HalfedgeHandle> removableHalfedges;
	removableHalfedges.reserve(mesh.n_vertices() - clusterCount);	// TODO!
	for(const auto vertex : mesh.vertices()) {
		const auto pos = util::pun<ei::Vec3>(mesh.point(vertex));
		const auto clusterIndex = get_cluster_index(pos);
		if(clusterIndex.has_value()) {
			if(vertex == representative[clusterIndex->index])
				continue;
#
			// Tell all incoming half-edges from outside the cluster that they now point to the cluster vertex instead
			for(auto iter = mesh.vih_ccwbegin(vertex); iter.is_valid(); ++iter) {
				const auto from = mesh.from_vertex_handle(*iter);
				if((from == representative[clusterIndex->index])
					|| (get_cluster_index_no_stop(util::pun<ei::Vec3>(mesh.point(from))).index == clusterIndex->index))
					removableHalfedges.push_back(*iter);
			}
		}
	}
	if(garbageCollect) {
		mesh.request_vertex_status();
		mesh.request_edge_status();
		mesh.request_face_status();
	}

	// We do up to 10 takes to remove as many as possible
	for(std::size_t tries = 0u; tries < 10u && !removableHalfedges.empty(); ++tries) {
		std::size_t free = 0u;
		for(std::size_t i = 0u; i < removableHalfedges.size(); ++i) {
			const auto heh = removableHalfedges[i];
			if(mesh.is_collapse_ok(heh))
				mesh.collapse(heh);
			else
				removableHalfedges[free++] = heh;
		}
		removableHalfedges.resize(free);
	}
	if(garbageCollect) {
		mesh.garbage_collection();
		mesh.release_vertex_status();
		mesh.release_edge_status();
		mesh.release_face_status();
	}


	return clusterCount;
}

template class OctreeVertexClusterer<renderer::decimaters::FloatOctree>;

} // namespace mufflon::scene::clustering