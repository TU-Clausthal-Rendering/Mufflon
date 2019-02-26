#pragma once

#include "util/assert.hpp"
#include "util/int_types.hpp"
#include "core/renderer/importance/importance_map.hpp"
#include "core/scene/geometry/polygon_mesh.hpp"
#include <OpenMesh/Core/Utils/Property.hh>
#include <OpenMesh/Tools/Decimater/ModBaseT.hh>
#include <OpenMesh/Tools/Decimater/DecimaterT.hh>

namespace mufflon::renderer {

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
	void set_importance_map(const ImportanceMap& importanceMap, const u32 meshIndex);
	void postprocess_collapse(const CollapseInfo& ci) final;

private:
	OpenMesh::VPropHandleT<float> m_importancePropHandle;
	bool m_useCollapseHistory;
	const ImportanceMap* m_importanceMap;
	u32 m_meshIndex;
};

} // namespace mufflon::renderer