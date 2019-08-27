#include "displacement_mapper.hpp"
#include "profiler/cpu_profiler.hpp"
#include "util/parallel.hpp"
#include "util/punning.hpp"
#include "core/scene/scenario.hpp"
#include "core/scene/materials/material.hpp"
#include "core/scene/textures/interface.hpp"
#include <ei/vector.hpp>

namespace mufflon::scene::tessellation {

constexpr inline bool USE_CENTRAL_DIFFERENCE = false;


void DisplacementMapper::set_edge_vertex(const float x, const OpenMesh::EdgeHandle edge,
										 const OpenMesh::VertexHandle vertex) {
	using namespace materials;

	// Perform regular tessellation first
	Tessellater::set_edge_vertex(x, edge, vertex);
	if constexpr(USE_CENTRAL_DIFFERENCE) {
		mAssert(x >= 0.f && x <= 1.f);
		float displacement = 0.f;
		u32 faceCount = 0u;

		if(x == 0.f) {
			// Corner -> n materials possible
			const ei::Vec3 pos = util::pun<ei::Vec3>(m_mesh->point(vertex));
			const ei::Vec3 normal = util::pun<ei::Vec3>(m_mesh->normal(vertex));
			const ei::Vec2 uv = util::pun<ei::Vec2>(m_mesh->texcoord2D(vertex));
			auto[tangentX, tangentY] = get_edge_vertex_tangents(edge, pos, normal);

			float offset = 0.f;
			ei::Vec3 newNormal{ 0.f };
			u32 count = 0u;

			for(auto iter = m_mesh->cvf_ccwbegin(vertex); iter.is_valid(); ++iter) {
				const MaterialIndex matIdx = m_mesh->property(m_matHdl, *iter);
				const IMaterial* mat = m_scenario->get_assigned_material(matIdx);

				if(mat->get_displacement_map() != nullptr) {
					auto[currOffset, currNewNormal] = compute_displacement(*mat, tangentX, tangentY, normal, uv);
					offset += currOffset;
					newNormal += currNewNormal;
					++count;
				}
			}
			offset /= static_cast<float>(count);
			newNormal /= static_cast<float>(count);
			m_mesh->set_point(vertex, util::pun<OpenMesh::Vec3f>(pos + offset * normal));
			m_mesh->set_normal(vertex, util::pun<typename geometry::PolygonMeshType::Normal>(newNormal));
		} else {
			// Edge -> max. two materials possible
			const OpenMesh::FaceHandle f0 = m_mesh->face_handle(m_mesh->halfedge_handle(edge, 0));
			const OpenMesh::FaceHandle f1 = m_mesh->face_handle(m_mesh->halfedge_handle(edge, 1));
			const MaterialIndex matIdx0 = m_mesh->property(m_matHdl, f0);
			const MaterialIndex matIdx1 = m_mesh->property(m_matHdl, f1);
			const IMaterial* mat0 = m_scenario->get_assigned_material(matIdx0);

			if(matIdx0 != matIdx1) {
				const IMaterial* mat1 = m_scenario->get_assigned_material(matIdx1);
				// Two different materials, need to compute extra things
				if(mat0->get_displacement_map() != nullptr || mat1->get_displacement_map() != nullptr) {
					const ei::Vec3 pos = util::pun<ei::Vec3>(m_mesh->point(vertex));
					const ei::Vec3 normal = util::pun<ei::Vec3>(m_mesh->normal(vertex));
					const ei::Vec2 uv = util::pun<ei::Vec2>(m_mesh->texcoord2D(vertex));
					auto[tangentX, tangentY] = get_edge_vertex_tangents(edge, pos, normal);

					float offset = 0.f;
					ei::Vec3 newNormal{ 0.f };
					u32 count = 0u;

					if(mat0->get_displacement_map() != nullptr) {
						auto[offset0, newNormal0] = compute_displacement(*mat0, tangentX, tangentY, normal, uv);
						offset += offset0;
						newNormal += newNormal0;
						++count;
					}
					if(mat1->get_displacement_map() != nullptr) {
						auto[offset1, newNormal1] = compute_displacement(*mat1, tangentX, tangentY, normal, uv);
						offset += offset1;
						newNormal += newNormal1;
						++count;
					}
					offset /= static_cast<float>(count);
					newNormal /= static_cast<float>(count);
					m_mesh->set_point(vertex, util::pun<OpenMesh::Vec3f>(pos + offset * normal));
					m_mesh->set_normal(vertex, util::pun<typename geometry::PolygonMeshType::Normal>(newNormal));
				}
			} else {
				if(mat0->get_displacement_map() != nullptr) {
					const ei::Vec3 pos = util::pun<ei::Vec3>(m_mesh->point(vertex));
					const ei::Vec3 normal = util::pun<ei::Vec3>(m_mesh->normal(vertex));
					const ei::Vec2 uv = util::pun<ei::Vec2>(m_mesh->texcoord2D(vertex));
					auto[tangentX, tangentY] = get_edge_vertex_tangents(edge, pos, normal);
					auto[offset, newNormal] = compute_displacement(*mat0, tangentX, tangentY, normal, uv);
					m_mesh->set_point(vertex, util::pun<OpenMesh::Vec3f>(pos + offset * normal));
					m_mesh->set_normal(vertex, util::pun<typename geometry::PolygonMeshType::Normal>(newNormal));
				}
			}
		}
	}
}

void DisplacementMapper::set_quad_inner_vertex(const float x, const float y,
											   const OpenMesh::VertexHandle vertex,
											   const OpenMesh::FaceHandle face,
											   const std::vector<std::pair<OpenMesh::VertexHandle, AddedVertices>>& vertices) {
	using namespace materials;
	// Perform regular tessellation first
	Tessellater::set_quad_inner_vertex(x, y, vertex, face, vertices);

	if constexpr(USE_CENTRAL_DIFFERENCE) {
		const MaterialIndex matIdx = m_mesh->property(m_matHdl, face);
		const IMaterial* mat = m_scenario->get_assigned_material(matIdx);

		if(mat->get_displacement_map() != nullptr) {
			const ei::Vec3 pos = util::pun<ei::Vec3>(m_mesh->point(vertex));
			const ei::Vec3 shadingNormal = util::pun<ei::Vec3>(m_mesh->normal(vertex));
			const ei::Vec2 uv = util::pun<ei::Vec2>(m_mesh->texcoord2D(vertex));

			const ei::Vec3 v[4] = {
				util::pun<ei::Vec3>(m_mesh->point(vertices[0].first)),
				util::pun<ei::Vec3>(m_mesh->point(vertices[1].first)),
				util::pun<ei::Vec3>(m_mesh->point(vertices[2].first)),
				util::pun<ei::Vec3>(m_mesh->point(vertices[3].first))
			};
			const ei::Vec2 uvV[4] = {
				util::pun<ei::Vec2>(m_mesh->texcoord2D(vertices[0].first)),
				util::pun<ei::Vec2>(m_mesh->texcoord2D(vertices[1].first)),
				util::pun<ei::Vec2>(m_mesh->texcoord2D(vertices[2].first)),
				util::pun<ei::Vec2>(m_mesh->texcoord2D(vertices[3].first))
			};
			const ei::Vec3 dxds = (1.f - uv.u) * (v[3u] - v[0u]) + uv.u * (v[2u] - v[1u]);
			const ei::Vec3 dxdt = (1.f - uv.v) * (v[1u] - v[0u]) + uv.v * (v[2u] - v[3u]);
			const ei::Matrix<float, 3, 2> dxdst{
				dxds.x, dxdt.x,
				dxds.y, dxdt.y,
				dxds.z, dxdt.z
			};
			const ei::Vec2 duds = (1.f - uv.u) * (uvV[3u] - uvV[0u]) + uv.u * (uvV[2u] - uvV[1u]);
			const ei::Vec2 dudt = (1.f - uv.v) * (uvV[1u] - uvV[0u]) + uv.v * (uvV[2u] - uvV[3u]);
			const ei::Matrix<float, 2, 2> dudst{
				duds.x, dudt.x,
				duds.y, dudt.y,
			};
			float detDudst = dudst[0] * dudst[3] - dudst[1] * dudst[2];
			ei::Vec3 tangentX;
			if(detDudst >= 1e-5f || detDudst <= -1e5f)
				tangentX = dxdst * ei::Vec2{ dudst[3] / detDudst, -dudst[2] / detDudst };
			else tangentX = dxds;

			// Determine shading normal and tangents
			// Gram-Schmidt
			const ei::Vec3 shadingTangentX = ei::normalize(
				tangentX - shadingNormal * ei::dot(tangentX, shadingNormal));
			const ei::Vec3 shadingTangentY = cross(shadingNormal, shadingTangentX);

			// Perform displacement
			auto[offset, newNormal] = compute_displacement(*mat, shadingTangentX, shadingTangentY, shadingNormal, uv);
			m_mesh->set_point(vertex, util::pun<OpenMesh::Vec3f>(pos + offset * shadingNormal));
			m_mesh->set_normal(vertex, util::pun<typename geometry::PolygonMeshType::Normal>(newNormal));
		}
	}
}
void DisplacementMapper::set_triangle_inner_vertex(const float x, const float y,
												   const OpenMesh::VertexHandle vertex,
												   const OpenMesh::FaceHandle face,
												   const std::vector<std::pair<OpenMesh::VertexHandle, AddedVertices>>& vertices) {
	using namespace materials;
	// Perform regular tessellation first
	Tessellater::set_triangle_inner_vertex(x, y, vertex, face, vertices);

	if constexpr(USE_CENTRAL_DIFFERENCE) {
		const MaterialIndex matIdx = m_mesh->property(m_matHdl, face);
		const IMaterial* mat = m_scenario->get_assigned_material(matIdx);

		if(mat->get_displacement_map() != nullptr) {
			const ei::Vec3 pos = util::pun<ei::Vec3>(m_mesh->point(vertex));
			const ei::Vec3 shadingNormal = util::pun<ei::Vec3>(m_mesh->normal(vertex));
			const ei::Vec2 uv = util::pun<ei::Vec2>(m_mesh->texcoord2D(vertex));

			const ei::Vec3 v[3] = {
				util::pun<ei::Vec3>(m_mesh->point(vertices[0].first)),
				util::pun<ei::Vec3>(m_mesh->point(vertices[1].first)),
				util::pun<ei::Vec3>(m_mesh->point(vertices[2].first))
			};
			const ei::Vec2 uvV[3] = {
				util::pun<ei::Vec2>(m_mesh->texcoord2D(vertices[0].first)),
				util::pun<ei::Vec2>(m_mesh->texcoord2D(vertices[1].first)),
				util::pun<ei::Vec2>(m_mesh->texcoord2D(vertices[2].first))
			};
			// Compute the tangent space by solving LES
			const ei::Vec3 dx0 = v[1u] - v[0u];
			const ei::Vec3 dx1 = v[2u] - v[0u];
			const ei::Vec2 du0 = uvV[1u] - uvV[0u];
			const ei::Vec2 du1 = uvV[2u] - uvV[0u];
			float det = (du0.x * du1.y - du0.y * du1.x);
			// TODO: fetch the instance instead (issue #44)
			ei::Vec3 tangentX;
			if(det >= 1e-5f || det <= -1e5f)
				tangentX = (dx0 * du1.y - dx1 * du0.y) / det;
			else tangentX = dx0;

			// Determine shading normal and tangents
			// Gram-Schmidt
			const ei::Vec3 shadingTangentX = ei::normalize(
				tangentX - shadingNormal * ei::dot(tangentX, shadingNormal));
			const ei::Vec3 shadingTangentY = cross(shadingNormal, shadingTangentX);

			// Perform displacement
			auto[offset, newNormal] = compute_displacement(*mat, shadingTangentX, shadingTangentY, shadingNormal, uv);
			m_mesh->set_point(vertex, util::pun<OpenMesh::Vec3f>(pos + offset * shadingNormal));
			m_mesh->set_normal(vertex, util::pun<typename geometry::PolygonMeshType::Normal>(newNormal));
		}
	}
}


void DisplacementMapper::post_tessellate() {
	auto profileTimer = Profiler::instance().start<CpuProfileState>("DisplacementMapper::post_tessellate");
	if constexpr(!USE_CENTRAL_DIFFERENCE) {
		// Actual displacement: go over all vertices, check if one of its faces is displacement mapped
		// according to the provided material assignment and if yes, adjust the vertex position
		// along the normal (TODO: geometric or shading); if multiple adjacent faces disagree
		// about the displacement (e.g. two different displacement maps), then the average
		// is taken
#pragma PARALLEL_FOR
		for(i64 i = 0; i < static_cast<i64>(m_mesh->n_vertices()); ++i) {
			const auto vertex = m_mesh->vertex_handle(static_cast<u32>(i));
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
				ei::Vec3 normal{ 0.f };

				typename geometry::PolygonMeshType::Normal inHeVec;
				auto heIter = m_mesh->cvih_iter(vertex);
				if(!heIter.is_valid())
					continue;
				m_mesh->calc_edge_vector(*heIter, inHeVec);
				ei::Vec3 normInHeVec = ei::normalize(util::pun<ei::Vec3>(inHeVec));
				for(; heIter.is_valid(); ++heIter) {
					if(m_mesh->is_boundary(*heIter))
						continue;
					OpenMesh::HalfedgeHandle outHeh(m_mesh->next_halfedge_handle(*heIter));
					typename geometry::PolygonMeshType::Normal outHeVec;
					m_mesh->calc_edge_vector(outHeh, outHeVec);
					const auto normOutHeVec = ei::normalize(util::pun<ei::Vec3>(outHeVec));
					const ei::Vec3 currNormal = ei::cross(util::pun<ei::Vec3>(inHeVec),
														  util::pun<ei::Vec3>(outHeVec));
					const float angle = std::acos(-ei::dot(normOutHeVec, normInHeVec));
					normal += currNormal * angle; // Contains sector area already
					inHeVec = -outHeVec;
					normInHeVec = -normOutHeVec;
				}
				m_mesh->set_normal(vertex, util::pun<typename geometry::PolygonMeshType::Normal>(ei::normalize(normal)));
			}
		}
	}
}

std::pair<ei::Vec3, ei::Vec3> DisplacementMapper::get_edge_vertex_tangents(const OpenMesh::EdgeHandle edge,
																		   const ei::Vec3& p0,
																		   const ei::Vec3& normal) {
	const OpenMesh::VertexHandle to = m_mesh->to_vertex_handle(m_mesh->halfedge_handle(edge, 0u));
	const ei::Vec3 p1 = util::pun<ei::Vec3>(m_mesh->point(to));
	const ei::Vec3 initTangentX = p1 - p0;
	const ei::Vec3 tangentY = ei::normalize(ei::cross(normal, initTangentX));
	return std::make_pair(ei::cross(tangentY, normal), tangentY);
}

std::tuple<ei::Vec3, ei::Vec3, ei::Vec3> DisplacementMapper::get_face_vertex_tangents(const OpenMesh::FaceHandle face,
																					  const ei::Vec2 surfaceParams) {
	// TODO
	return {};
}

std::pair<float, ei::Vec3> DisplacementMapper::compute_displacement(const materials::IMaterial& mat, const ei::Vec3& tX,
																	   const ei::Vec3& tY, const ei::Vec3& normal,
																	   const ei::Vec2& uv) {
	// Compute new point according to displacement map
	const auto dispMap = mat.get_displacement_map();
	const auto dispMapHdl = dispMap->acquire_const<Device::CPU>();
	// Compute new normal

	/*const auto dispMapSize = textures::get_texture_size(dispMapHdl);

	ei::IVec2 bottomLeft{ static_cast<i32>(ei::floor(uv.x * dispMapSize.x)), static_cast<i32>(ei::floor(uv.y * dispMapSize.y)) };
	ei::IVec2 topLeft{ static_cast<i32>(ei::floor(uv.x * dispMapSize.x)), static_cast<i32>(ei::ceil(uv.y * dispMapSize.y)) };
	ei::IVec2 bottomRight{ static_cast<i32>(ei::ceil(uv.x * dispMapSize.x)), static_cast<i32>(ei::floor(uv.y * dispMapSize.y)) };
	ei::IVec2 topRight{ static_cast<i32>(ei::ceil(uv.x * dispMapSize.x)), static_cast<i32>(ei::ceil(uv.y * dispMapSize.y)) };
	normal += ei::normalize(ei::cross()*/

	return std::make_pair(mat.get_displacement_bias() + textures::sample(dispMapHdl, uv).x * mat.get_displacement_scale(),
						  normal);
}

} // namespace mufflon::scene::tessellation