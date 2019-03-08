#pragma once

#include "importance_map.hpp"
#include "util/assert.hpp"
#include "util/int_types.hpp"
#include "core/scene/geometry/polygon_mesh.hpp"
#include <OpenMesh/Core/Utils/Property.hh>
#include <OpenMesh/Tools/Decimater/ModBaseT.hh>
#include <OpenMesh/Tools/Decimater/DecimaterT.hh>

namespace mufflon::renderer::importance {

// Decimater solely based on importance.
template < typename MeshT = scene::geometry::PolygonMeshType >
class ModImportance : public OpenMesh::Decimater::ModBaseT<MeshT> {
public:
	DECIMATING_MODULE(ModImportance, MeshT, ModImportance);

	ModImportance(MeshT &mesh);
	virtual ~ModImportance() = default;
	ModImportance(const ModImportance&) = delete;
	ModImportance(ModImportance&&) = delete;
	ModImportance& operator=(const ModImportance&) = delete;
	ModImportance& operator=(ModImportance&&) = delete;

	void initialize() final;
	float collapse_priority(const CollapseInfo& ci) final;
	void use_collapse_history(bool val);
	void set_importance_map(ImportanceMap& importanceMap, const u32 meshIndex);
	// Post-process halfedge collapse (accumulate importance)
	void postprocess_collapse(const CollapseInfo& ci) final;

private:
	//const std::vector<unsigned int>* m_meshDataCollapseHistory;
	//const MeshT* m_importanceMapCreationMesh;
	//OpenMesh::VPropHandleT<unsigned int> m_indexMapHandle;
	bool m_useCollapseHistory;
	ImportanceMap* m_importanceMap;
	u32 m_meshIndex;
};

template < typename MeshT = scene::geometry::PolygonMeshType >
class MaxNormalDeviation : public OpenMesh::Decimater::ModBaseT<MeshT> {
public:
	DECIMATING_MODULE(MaxNormalDeviation, MeshT, MaxNormalDeviation);

	MaxNormalDeviation(MeshT &mesh);
	virtual ~MaxNormalDeviation() = default;
	MaxNormalDeviation(const MaxNormalDeviation&) = delete;
	MaxNormalDeviation(MaxNormalDeviation&&) = delete;
	MaxNormalDeviation& operator=(const MaxNormalDeviation&) = delete;
	MaxNormalDeviation& operator=(MaxNormalDeviation&&) = delete;

	float collapse_priority(const CollapseInfo& ci) final;
	void set_max_deviation(const double deviation);
	double get_max_devation() const noexcept;
	void set_error_tolerance_factor(const double factor);

private:
	std::vector<typename MeshT::Normal> m_normalStorage;

	double m_maxDeviation;
	double m_minCos;
};

} // namespace mufflon::renderer::importance