#pragma once

#include "util/punning.hpp"
#include "core/scene/geometry/polygon_mesh.hpp"
#include <ei/vector.hpp>
#include <OpenMesh/Core/Geometry/QuadricT.hh>
#include <limits>
#include <optional>

namespace ei {

// Invert function that returns optional to indicate non-invertability instead of identity matrix
inline std::optional<ei::Mat4x4> invert_opt(const ei::Mat4x4& mat0) noexcept {
	ei::Mat4x4 LU;
	ei::UVec4 p;
	if(ei::decomposeLUp(mat0, LU, p))
		return ei::solveLUp(LU, p, ei::identity4x4());
	return std::nullopt;
}

} // namespace ei

namespace mufflon::scene::clustering {

template < class T >
inline void compute_error_quadrics(geometry::PolygonMeshType& mesh, const OpenMesh::VPropHandleT<OpenMesh::Geometry::QuadricT<T>> prop) {
	static_assert(std::is_floating_point_v<T>, "Quadrics are only defined for floating-point types");

	using OpenMesh::Geometry::QuadricT;
	for(const auto vertex : mesh.vertices())
		mesh.property(prop, vertex).clear();
	for(const auto face : mesh.faces()) {
		// Assume planar n-gon
		auto vIter = mesh.cfv_ccwbegin(face);
		const auto vh0 = *vIter;
		const auto vh1 = *(++vIter);
		const auto vh2 = *(++vIter);

		const auto p0 = util::pun<ei::Vec3>(mesh.point(vh0));
		const auto p1 = util::pun<ei::Vec3>(mesh.point(vh1));
		const auto p2 = util::pun<ei::Vec3>(mesh.point(vh2));
		auto normal = ei::cross(p1 - p0, p2 - p0);
		auto area = ei::len(normal);
		if(area > std::numeric_limits<decltype(area)>::min()) {
			normal /= area;
			area *= 0.5f;
		}

		const auto d = -ei::dot(p0, normal);
		QuadricT<T> q{ normal.x, normal.y, normal.z, d };
		q *= area;
		mesh.property(prop, vh0) += q;
		mesh.property(prop, vh1) += q;
		mesh.property(prop, vh2) += q;
	}
}

} // namespace mufflon::scene::clustering