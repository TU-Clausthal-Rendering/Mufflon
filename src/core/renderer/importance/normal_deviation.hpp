#pragma once

#include "util/degrad.hpp"
#include "core/scene/geometry/polygon_mesh.hpp"
#include <OpenMesh/Tools/Decimater/ModBaseT.hh>

namespace mufflon::renderer::importance {

// Makes sure there are no normal flips
template < class MeshT = scene::geometry::PolygonMeshType >
class NormalDeviationModule : public OpenMesh::Decimater::ModBaseT<MeshT> {
public:
	DECIMATING_MODULE(NormalDeviationModule, MeshT, NormalDeviationModule);

	NormalDeviationModule(MeshT& mesh);
	virtual ~NormalDeviationModule() = default;
	NormalDeviationModule(const NormalDeviationModule&) = delete;
	NormalDeviationModule(NormalDeviationModule&&) = delete;
	NormalDeviationModule& operator=(const NormalDeviationModule&) = delete;
	NormalDeviationModule& operator=(NormalDeviationModule&&) = delete;

	float collapse_priority(const CollapseInfo& ci) final;
	void set_max_deviation(const Degrees deviation);

private:
	float m_minCos;
};

} // namespace mufflon::renderer::importance