#include "uniform_clustering.hpp"
#include "util.hpp"
#include "util/log.hpp"
#include "util/punning.hpp"
#include <OpenMesh/Core/Geometry/QuadricT.hh>

namespace mufflon::scene::clustering {

namespace {

template < class Iter >
u32 compute_cluster_center(const Iter begin, const Iter end) {
	u32 clusterCount = 0u;
	for(auto iter = begin; iter != end; ++iter) {
		auto& cluster = *iter;
		if(cluster.count > 0) {
			// TODO: fails for e.g. shadow holder (broken mesh?)
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
			// TODO: are quads at fault?
			clusterCount += 1;
		}
	}
	return clusterCount;
}

} // namespace

// Cluster for vertex clustering
struct UniformVertexCluster {
	ei::Vec3 posAccum{ 0.f };
	u32 count{ 0u };
	OpenMesh::Geometry::Quadricd q{};

	void add_vertex(const ei::Vec3& pos, const OpenMesh::Geometry::Quadricd& quadric) noexcept {
		count += 1u;
		posAccum += pos;
		q += quadric;
	}
};

std::size_t UniformVertexClusterer::cluster(geometry::PolygonMeshType& mesh, const ei::Box& aabb) {
	using OpenMesh::Geometry::Quadricd;

	// we have to track a few things per cluster
	const auto gridRes = m_gridRes;
	std::vector<UniformVertexCluster> clusters(ei::prod(m_gridRes));

	const auto aabbMin = aabb.min;
	const auto aabbDiag = aabb.max - aabb.min;
	// convenience function to compute the cluster index from a position
	auto get_cluster_index = [aabbMin, aabbDiag, gridRes](const ei::Vec3& pos) -> u32 {
		// get the normalized position [0, 1]^3
		const auto normPos = (pos - aabbMin) / aabbDiag;
		// get the discretized grid position
		const auto gridPos = ei::min(ei::UVec3{ normPos * ei::Vec3{ gridRes } }, gridRes - 1u);
		// convert the 3d grid position into a 1d index (x -> y -> z)
		const auto gridIndex = gridPos.x + gridPos.y * gridRes.x + gridPos.z * gridRes.x * gridRes.y;
		return static_cast<u32>(gridIndex);
	};

	OpenMesh::VPropHandleT<Quadricd> quadricProps{};
	mesh.add_property(quadricProps);
	if(!quadricProps.is_valid())
		throw std::runtime_error("failed to add error quadric property");
	// compute the error quadrics for each vertex
	compute_error_quadrics(mesh, quadricProps);

	// for each vertex, determine the cluster it belongs to and update its statistics
	for(const auto vertex : mesh.vertices()) {
		const auto pos = util::pun<ei::Vec3>(mesh.point(vertex));
		const auto clusterIndex = get_cluster_index(pos);
		mAssert(clusterIndex < clusters.size());
		clusters[clusterIndex].add_vertex(pos, mesh.property(quadricProps, vertex));
	}
	mesh.remove_property(quadricProps);

	// calculate the representative cluster position for every cluster
	u32 clusterCount = compute_cluster_center(clusters.begin(), clusters.end());


	std::vector<geometry::PolygonMeshType::VertexHandle> representative(clusters.size());
	// then we set the position of every vertex to that of its
	// cluster representative
	for(const auto vertex : mesh.vertices()) {
		const auto pos = util::pun<ei::Vec3>(mesh.point(vertex));
		const auto clusterIndex = get_cluster_index(pos);
		// new cluster position has been previously computed
		const auto newPos = clusters[clusterIndex].posAccum;
		mesh.point(vertex) = util::pun<geometry::PolygonMeshType::Point>(newPos);
		representative[clusterIndex] = vertex;
	}

	std::vector<geometry::PolygonMeshType::HalfedgeHandle> removableHalfedges;
	removableHalfedges.reserve(mesh.n_vertices() - std::min(static_cast<std::size_t>(clusterCount), mesh.n_vertices()));	// TODO!
	for(const auto vertex : mesh.vertices()) {
		const auto pos = util::pun<ei::Vec3>(mesh.point(vertex));
		const auto clusterIndex = get_cluster_index(pos);
		if(vertex == representative[clusterIndex])
			continue;
#
		// Tell all incoming half-edges from outside the cluster that they now point to the cluster vertex instead
		for(auto iter = mesh.vih_ccwbegin(vertex); iter.is_valid(); ++iter) {
			const auto from = mesh.from_vertex_handle(*iter);
			if((from == representative[clusterIndex]) || (get_cluster_index(util::pun<ei::Vec3>(mesh.point(from))) == clusterIndex))
				removableHalfedges.push_back(*iter);
		}
	}
	mesh.request_vertex_status();
	mesh.request_edge_status();
	mesh.request_face_status();

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
	mesh.garbage_collection();
	mesh.release_vertex_status();
	mesh.release_edge_status();
	mesh.release_face_status();


		return mesh.n_vertices();
}

} // namespace mufflon::scene::clustering