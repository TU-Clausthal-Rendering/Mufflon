#pragma once

#include "tessellater.hpp"
#include <ei/vector.hpp>
#include <tuple>

namespace mufflon::scene {

class Scenario;

namespace materials {
class IMaterial;
} // namespace materials

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
							   const std::vector<std::pair<OpenMesh::VertexHandle, AddedVertices>>& vertices) override;
	void set_triangle_inner_vertex(const float x, const float y,
								   const OpenMesh::VertexHandle vertex,
								   const OpenMesh::FaceHandle face,
								   const std::vector<std::pair<OpenMesh::VertexHandle, AddedVertices>>& vertices) override;

	void post_tessellate() override;

private:
	std::pair<ei::Vec3, ei::Vec3> get_edge_vertex_tangents(const OpenMesh::EdgeHandle edge,
															 const ei::Vec3& p0,
															 const ei::Vec3& normal);
	std::tuple<ei::Vec3, ei::Vec3, ei::Vec3> get_face_vertex_tangents(const OpenMesh::FaceHandle face,
																	  const ei::Vec2 surfaceParams);

	std::pair<float, ei::Vec3> compute_displacement(const materials::IMaterial& mat, const ei::Vec3& tX,
													   const ei::Vec3& tY, const ei::Vec3& normal,
													   const ei::Vec2& uv);
		
	const Scenario* m_scenario;
	OpenMesh::FPropHandleT<MaterialIndex> m_matHdl;
};

} // namespace tessellation

} // namespace mufflon::scene