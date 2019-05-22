#pragma once

#include "tessellater.hpp"
#include "core/scene/attribute.hpp"

namespace mufflon::scene {

class Scenario;

namespace tessellation {

// Special tessellater which ignores phong-tessellation, displaces the created vertices
// according to a potential displacement map and corrects the normal properly
class DisplacementMapper : public Tessellater {
public:
	DisplacementMapper(TessLevelOracle& oracle) :
		Tessellater(oracle) {}

	void set_scenario(const Scenario& scenario) noexcept {
		m_scenario = &scenario;
	}

	void set_material_idx_hdl(OpenMesh::FPropHandleT<MaterialIndex> matHdl) {
		m_matHdl = matHdl;
	}

protected:
	void set_edge_vertex(const float x, const OpenMesh::EdgeHandle edge,
						 const OpenMesh::VertexHandle vertex) override;
	void set_quad_inner_vertex(const float x, const float y,
							   const OpenMesh::VertexHandle vertex,
							   const OpenMesh::FaceHandle face,
							   const OpenMesh::VertexHandle(&vertices)[4u]) override;
	void set_triangle_inner_vertex(const float x, const float y,
								   const OpenMesh::VertexHandle vertex,
								   const OpenMesh::FaceHandle face,
								   const OpenMesh::VertexHandle(&vertices)[4u]) override;

	void post_tessellate() override;

private:
	const Scenario* m_scenario;
	OpenMesh::FPropHandleT<MaterialIndex> m_matHdl;
};

} // namespace tessellation

} // namespace mufflon::scene