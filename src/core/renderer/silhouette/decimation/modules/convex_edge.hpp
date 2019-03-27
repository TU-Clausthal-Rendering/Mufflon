#pragma once

#include "util/int_types.hpp"
#include "core/scene/geometry/polygon_mesh.hpp"
#include <OpenMesh/Tools/Decimater/ModBaseT.hh>

namespace mufflon::renderer::silhouette::decimation::modules {

// Disallows non-convex edges from collapsing
template < class MeshT = scene::geometry::PolygonMeshType >
class ConvexDecimationModule : public OpenMesh::Decimater::ModBaseT<MeshT> {
public:
	DECIMATING_MODULE(ConvexDecimationModule, MeshT, ConvexDecimationModule);

	ConvexDecimationModule(MeshT& mesh) :
		Base(mesh, true) {}
	virtual ~ConvexDecimationModule() = default;
	ConvexDecimationModule(const ConvexDecimationModule&) = delete;
	ConvexDecimationModule(ConvexDecimationModule&&) = delete;
	ConvexDecimationModule& operator=(const ConvexDecimationModule&) = delete;
	ConvexDecimationModule& operator=(ConvexDecimationModule&&) = delete;

	float collapse_priority(const CollapseInfo& ci) final {
		const auto p0 = util::pun<ei::Vec3>(ci.p0);
		const auto p1 = util::pun<ei::Vec3>(ci.p1);
		const auto p0p1 = p1 - p0;
		const auto pl = util::pun<ei::Vec3>(Base::mesh().point(ci.vl));
		const auto pr = util::pun<ei::Vec3>(Base::mesh().point(ci.vr));
		const auto flNormal = ei::cross(p0p1, pl - p0);
		const auto frNormal = ei::cross(pr - p0, p0p1);
		const auto p0p1Normal = 0.5f * (flNormal + frNormal); // Not normalized because not needed
		{
			// First for v0: vx -> v0
			for(auto circIter = Base::mesh().cvv_ccwbegin(ci.v0); circIter.is_valid(); ++circIter) {
				if(*circIter == ci.v1)
					continue;
				const auto pxp0 = p0 - util::pun<ei::Vec3>(Base::mesh().point(*circIter));
				const auto dot = ei::dot(p0p1Normal, pxp0);
				if(dot < 0.f)
					return Base::ILLEGAL_COLLAPSE;
			}
			// Then for v1: vx -> v1
			for(auto circIter = Base::mesh().cvv_ccwbegin(ci.v1); circIter.is_valid(); ++circIter) {
				if(*circIter == ci.v0)
					continue;
				const auto pxp1 = p1 - util::pun<ei::Vec3>(Base::mesh().point(*circIter));
				const auto dot = ei::dot(p0p1Normal, pxp1);
				if(dot < 0.f)
					return Base::ILLEGAL_COLLAPSE;
			}
		}

		return Base::LEGAL_COLLAPSE;
	}
};

} // namespace mufflon::renderer::silhouette::decimation::modules