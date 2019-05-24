#include "displacement_mapper.hpp"
#include "profiler/cpu_profiler.hpp"
#include "util/parallel.hpp"
#include "util/punning.hpp"
#include "core/scene/scenario.hpp"
#include "core/scene/materials/material.hpp"
#include "core/scene/textures/interface.hpp"
#include <ei/vector.hpp>

namespace mufflon::scene::tessellation {

namespace {

ei::Vec3 get_displaced_position() {
	return {};
}

// Returns the 

} // namespace

void DisplacementMapper::set_edge_vertex(const float x, const OpenMesh::EdgeHandle edge,
										 const OpenMesh::VertexHandle vertex) {
	using namespace materials;

	mAssert(x >= 0.f && x <= 1.f);
	ei::Vec3 normal{ 0.f };
	float displacement = 0.f;
	u32 faceCount = 0u;

	const OpenMesh::FaceHandle f0 = m_mesh->face_handle(m_mesh->halfedge_handle(edge, 0));
	const OpenMesh::FaceHandle f1 = m_mesh->face_handle(m_mesh->halfedge_handle(edge, 1));
	const MaterialIndex matIdx0 = m_mesh->property(m_matHdl, f0);
	const MaterialIndex matIdx1 = m_mesh->property(m_matHdl, f1);
	const IMaterial* mat0 = m_scenario->get_assigned_material(matIdx0);

	if(matIdx0 != matIdx1) {
		const IMaterial* mat1 = m_scenario->get_assigned_material(matIdx1);
		// Two different materials, need to compute extra things
		if(mat0->get_displacement_map() != nullptr && mat1->get_displacement_map() != nullptr) {
			// Regular tessellation
			Tessellater::set_edge_vertex(x, edge, vertex);
		} else {

		}
	} else {
		if(mat0->get_displacement_map() != nullptr) {
			const OpenMesh::VertexHandle from = m_mesh->from_vertex_handle(m_mesh->halfedge_handle(edge, 0u));
			const OpenMesh::VertexHandle to = m_mesh->to_vertex_handle(m_mesh->halfedge_handle(edge, 0u));
			const ei::Vec3& p0 = util::pun<ei::Vec3>(m_mesh->point(from));
			const ei::Vec3& p1 = util::pun<ei::Vec3>(m_mesh->point(to));
			const ei::Vec3& n0 = util::pun<ei::Vec3>(m_mesh->normal(from));
			const ei::Vec3& n1 = util::pun<ei::Vec3>(m_mesh->normal(to));
			// Use phong tessellation to have the tessellation do something useful in absence
			// of displacement mapping
			const ei::Vec3 pos = ei::lerp(p0, p1, x);
			const ei::Vec3 normal = ei::lerp(n0, n1, x);
			const ei::Vec2 uv = ei::lerp(util::pun<ei::Vec2>(m_mesh->texcoord2D(from)),
										 util::pun<ei::Vec2>(m_mesh->texcoord2D(to)), x);

			// Compute new point according to displacement map
			const auto dispMap = mat0->get_displacement_map();
			const auto dispMapHdl = dispMap->acquire_const<Device::CPU>();
			displacement += mat0->get_displacement_bias() + textures::sample(dispMapHdl, uv).x * mat0->get_displacement_scale();
			// Compute new normal
			const ei::Vec3 geomTangentX = p1 - p0;
			const ei::Vec3 tangentY = ei::cross(geomTangentX, normal);
			const ei::Vec3 tangentX = ei::cross(normal, tangentY);

			/*const auto dispMapSize = textures::get_texture_size(dispMapHdl);

			ei::IVec2 bottomLeft{ static_cast<i32>(ei::floor(uv.x * dispMapSize.x)), static_cast<i32>(ei::floor(uv.y * dispMapSize.y)) };
			ei::IVec2 topLeft{ static_cast<i32>(ei::floor(uv.x * dispMapSize.x)), static_cast<i32>(ei::ceil(uv.y * dispMapSize.y)) };
			ei::IVec2 bottomRight{ static_cast<i32>(ei::ceil(uv.x * dispMapSize.x)), static_cast<i32>(ei::floor(uv.y * dispMapSize.y)) };
			ei::IVec2 topRight{ static_cast<i32>(ei::ceil(uv.x * dispMapSize.x)), static_cast<i32>(ei::ceil(uv.y * dispMapSize.y)) };
			normal += ei::normalize(ei::cross()*/

			if(m_usePhongTessellation) {
				const ei::Vec3 plane0 = pos - ei::dot(pos - p0, n0) * n0;
				const ei::Vec3 plane1 = pos - ei::dot(pos - p1, n1) * n1;
				const ei::Vec3 phongPos = ei::lerp(plane0, plane1, x);
				m_mesh->set_point(vertex, util::pun<OpenMesh::Vec3f>(phongPos));
			} else {
				m_mesh->set_point(vertex, util::pun<OpenMesh::Vec3f>(pos));
			}
			m_mesh->set_normal(vertex, util::pun<OpenMesh::Vec3f>(normal));
			m_mesh->set_texcoord2D(vertex, util::pun<OpenMesh::Vec2f>(uv));
		} else {
			// Regular tessellation
			Tessellater::set_edge_vertex(x, edge, vertex);
		}
	}
}

void DisplacementMapper::set_quad_inner_vertex(const float x, const float y,
											   const OpenMesh::VertexHandle vertex,
											   const OpenMesh::FaceHandle face,
											   const OpenMesh::VertexHandle(&vertices)[4u]) {
	// TODO
	Tessellater::set_quad_inner_vertex(x, y, vertex, face, vertices);
}
void DisplacementMapper::set_triangle_inner_vertex(const float x, const float y,
												   const OpenMesh::VertexHandle vertex,
												   const OpenMesh::FaceHandle face,
												   const OpenMesh::VertexHandle(&vertices)[4u]) {
	// TODO
	Tessellater::set_triangle_inner_vertex(x, y, vertex, face, vertices);
}


void DisplacementMapper::post_tessellate() {

	auto profileTimer = Profiler::instance().start<CpuProfileState>("DisplacementMapper::post_tessellate");
	// Actual displacement: go over all vertices, check if one of its faces is displacement mapped
	// according to the provided material assignment and if yes, adjust the vertex position
	// along the normal (TODO: geometric or shading); if multiple adjacent faces disagree
	// about the displacement (e.g. two different displacement maps), then the average
	// is taken
#pragma PARALLEL_FOR
	for(i64 i = 0; i < static_cast<i64>(m_mesh->n_vertices()); ++i) {
		const auto vertex = m_mesh->vertex_handle(static_cast<u32>(i));
		ei::Vec3 normal{ 0.f };
		float displacement = 0.f;
		u32 faceCount = 0u;
		const ei::Vec2 uv = util::pun<ei::Vec2>(m_mesh->texcoord2D(vertex));
		for(auto faceIter = m_mesh->cvf_ccwbegin(vertex); faceIter.is_valid(); ++faceIter) {
			const MaterialIndex matIdx = m_mesh->property(m_matHdl, *faceIter);
			const materials::IMaterial* mat = m_scenario->get_assigned_material(matIdx);
			if(const auto dispMap = mat->get_displacement_map(); dispMap != nullptr) {
				const auto dispMapHdl = dispMap->acquire_const<Device::CPU>();
				displacement += mat->get_displacement_bias() + textures::sample(dispMapHdl, uv).x * mat->get_displacement_scale();
			}
			++faceCount;
		}

		if(displacement != 0.f) {
			const ei::Vec3 normal = util::pun<ei::Vec3>(m_mesh->normal(vertex));
			displacement /= static_cast<float>(faceCount);
			m_mesh->set_point(vertex, util::pun<OpenMesh::Vec3f>(util::pun<ei::Vec3>(m_mesh->point(vertex)) + normal * displacement));
			// TODO: compute new normal!
		}
	}

	{
		auto normalProfileTimer = Profiler::instance().start<CpuProfileState>("DisplacementMapper::post_tessellate normals");
		// TODO: we recompute the geometric normals here, but we could probably compute them directly...
#pragma PARALLEL_FOR
		for(i64 i = 0; i < static_cast<i64>(m_mesh->n_vertices()); ++i) {
			const auto vertex = m_mesh->vertex_handle(static_cast<u32>(i));
			typename geometry::PolygonMeshType::Normal normal;
#pragma warning(push)
#pragma warning(disable : 4244)
			m_mesh->calc_vertex_normal_correct(vertex, normal);
#pragma warning(pop)
			m_mesh->set_normal(vertex, util::pun<typename geometry::PolygonMeshType::Normal>(ei::normalize(util::pun<ei::Vec3>(normal))));
		}
	}
}

} // namespace mufflon::scene::tessellation