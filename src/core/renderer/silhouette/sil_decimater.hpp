#pragma once

#include "sil_imp_map.hpp"
#include "util/assert.hpp"
#include "util/int_types.hpp"
#include "core/scene/geometry/polygon_mesh.hpp"
#include <OpenMesh/Core/Utils/Property.hh>
#include <OpenMesh/Tools/Decimater/ModBaseT.hh>
#include <OpenMesh/Tools/Decimater/DecimaterT.hh>

namespace mufflon::renderer::silhouette {

template < typename MeshT = scene::geometry::PolygonMeshType >
class ImportanceModule : public OpenMesh::Decimater::ModBaseT<MeshT> {
public:
	DECIMATING_MODULE(ImportanceModule, MeshT, ImportanceModule);

	ImportanceModule(MeshT &mesh);
	virtual ~ImportanceModule() = default;
	ImportanceModule(const ImportanceModule&) = delete;
	ImportanceModule(ImportanceModule&&) = delete;
	ImportanceModule& operator=(const ImportanceModule&) = delete;
	ImportanceModule& operator=(ImportanceModule&&) = delete;

	void initialize() final;
	float collapse_priority(const CollapseInfo& ci) final;
	void use_collapse_history(bool val);
	void set_importance_map(ImportanceMap& importanceMap, const u32 meshIndex, const float threshold);
	void postprocess_collapse(const CollapseInfo& ci) final;

private:
	bool m_useCollapseHistory;
	ImportanceMap* m_importanceMap;
	u32 m_meshIndex;

	float m_threshold;
};

} // namespace mufflon::renderer::silhouette