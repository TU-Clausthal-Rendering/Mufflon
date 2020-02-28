#pragma once

#include "polygon_mesh.hpp"
#include "util/punning.hpp"
#include <ei/vector.hpp>

namespace mufflon::scene::geometry {

// Computes the area of the given face (tri/quad only)
inline float compute_area(const PolygonMeshType& mesh, const OpenMesh::FaceHandle face) {
	auto vIter = mesh.cfv_ccwbegin(face);
	const auto a = *vIter; ++vIter;
	const auto b = *vIter; ++vIter;
	const auto c = *vIter; ++vIter;
	const auto pA = util::pun<ei::Vec3>(mesh.point(a));
	const auto pB = util::pun<ei::Vec3>(mesh.point(b));
	const auto pC = util::pun<ei::Vec3>(mesh.point(c));
	float area = ei::len(ei::cross(pB - pA, pC - pA));
	if(vIter.is_valid()) {
		const auto d = *vIter;
		const auto pD = util::pun<ei::Vec3>(mesh.point(d));
		area += ei::len(ei::cross(pC - pA, pD - pA));
	}
	return 0.5f * area;
}

// Computes the area sum of all faces bordering the given vertex (tri/quad only)
inline float compute_area(const PolygonMeshType& mesh, const OpenMesh::VertexHandle vertex) {
	float area = 0.f;
	for(auto fIter = mesh.cvf_ccwbegin(vertex); fIter.is_valid(); ++fIter)
		area += compute_area(mesh, *fIter);
	return area;
}

} // namespace mufflon::scene::geometry