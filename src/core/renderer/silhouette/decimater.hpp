#pragma once

#include "util/assert.hpp"
#include "util/int_types.hpp"
#include "core/scene/geometry/polygon_mesh.hpp"
#include <OpenMesh/Core/Utils/Property.hh>
#include <OpenMesh/Tools/Decimater/ModBaseT.hh>
#include <OpenMesh/Tools/Decimater/DecimaterT.hh>

namespace mufflon::renderer::silhouette {

void test() {
	using namespace OpenMesh;
}

template < class MeshType = scene::geometry::PolygonMeshType >
class ImportanceModule : public OpenMesh::Decimater::ModBaseT<MeshType> {
public:
	DECIMATING_MODULE(ImportanceModule, MeshType, ImportanceModule);

	ImportanceModule(Mesh &mesh) : Base(mesh, false) {}

	virtual ~ImportanceModule() = default;

	virtual void initialize() override {
	}

	virtual float collapse_priority(const CollapseInfo& ci) override {
		float importance = Base::mesh().property(m_importancePropHandle, ci.v0);
		for(auto ringVertexHandle = Base::mesh().vv_iter(ci.v0); ringVertexHandle.is_valid(); ++ringVertexHandle) {
			float factor = 1.0f;
			if(*ringVertexHandle == ci.v1)
				factor += 0.001f;

			importance += Base::mesh().property(m_importancePropHandle, *ringVertexHandle) * factor;
		}
		return importance;
	}

	virtual void postprocess_collapse(const CollapseInfo& _ci) override {
		// TODO
	}

	void set_importance_property(OpenMesh::VPropHandleT<float> importancePropHandle) {
		m_importancePropHandle = importancePropHandle;
	}

private:
	OpenMesh::VPropHandleT<float> m_importancePropHandle;
};


template < class MeshType = scene::geometry::PolygonMeshType >
class ImportanceBinaryModule : public OpenMesh::Decimater::ModBaseT<MeshType> {
public:
	DECIMATING_MODULE(ImportanceBinaryModule, MeshType, ImportanceBinaryModule);

	ImportanceBinaryModule(Mesh &mesh) : Base(mesh, true) {}

	virtual ~ImportanceBinaryModule() = default;

	virtual void initialize() override {
	}

	virtual float collapse_priority(const CollapseInfo& ci) override {
		const float importance = Base::mesh().property(m_importancePropHandle, ci.v0);
		if(importance != 0.f)
			return OpenMesh::Decimater::ModBaseT<Mesh>::ILLEGAL_COLLAPSE;
		return OpenMesh::Decimater::ModBaseT<Mesh>::LEGAL_COLLAPSE;
	}

	void set_importance_property(OpenMesh::VPropHandleT<float> importancePropHandle) {
		m_importancePropHandle = importancePropHandle;
	}

	virtual void postprocess_collapse(const CollapseInfo& _ci) override {
		// TODO
	}

private:
	OpenMesh::VPropHandleT<float> m_importancePropHandle;
};

} // namespace mufflon::renderer::silhouette