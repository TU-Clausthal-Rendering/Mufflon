#include "cam_dist.hpp"
#include "core/cameras/camera.hpp"
#include "core/cameras/pinhole.hpp"
#include "core/cameras/focus.hpp"
#include "core/scene/scenario.hpp"
#include "core/scene/materials/material.hpp"
#include "core/scene/materials/material_sampling.hpp"
#include "core/scene/textures/interface.hpp"

namespace mufflon::scene::tessellation {

namespace {

int compute_mipmap_level(ConstTextureHandle texture, const ei::Vec2& uvMin, const ei::Vec2& uvMax) {
	// Compute the texel the UVs fall onto, then find the number of equal high bits -> our mipmap level
	const ei::IVec2 texMin{
		static_cast<int>(ei::clamp(uvMin.u, 0.f, 1.f) * texture->get_width()),
		static_cast<int>(ei::clamp(uvMin.v, 0.f, 1.f) * texture->get_height())
	};
	const ei::IVec2 texMax{
		static_cast<int>(ei::clamp(uvMax.u, 0.f, 1.f) * texture->get_width()),
		static_cast<int>(ei::clamp(uvMax.v, 0.f, 1.f) * texture->get_height())
	};

	// Find the highest bit min and max disagree on
	const int simU = texMin.u ^ texMax.u;
	const int simV = texMin.v ^ texMax.v;
	// We shift once to get the actual mipmap level: equal texels means xor gives 0, thus one shift and the | 1
	// Examples:
	// 15 xor 13 -> 0b10 -> ilog2(0b101) -> 2
	// 127 xor 128 -> 0b11111111 -> ilog2(0b111111111) -> 8
	// 5 xor 5 -> 0b0 -> ilog2(0b1) -> 0
	const int levelU = ei::ilog2((simU << 1) | 1);
	const int levelV = ei::ilog2((simV << 1) | 1);
	return std::max(levelU, levelV);
}

ei::Vec2 get_min_max_displacement(const materials::IMaterial& mat, const ei::Vec2& minUv, const ei::Vec2& maxUv) {
	float minDisp = std::numeric_limits<float>::max();
	float maxDisp = -std::numeric_limits<float>::max();
	if(TextureHandle dispMap = mat.get_displacement_map(); dispMap != nullptr) {
		mAssert(mat.get_displacement_max_mips() != nullptr);
		textures::ConstTextureDevHandle_t<Device::CPU> dispMapHdl = dispMap->template acquire_const<Device::CPU>();
		textures::ConstTextureDevHandle_t<Device::CPU> dispMaxMIPHdl = mat.get_displacement_max_mips()->template acquire_const<Device::CPU>();
		const int mipmapLevel = compute_mipmap_level(dispMap, minUv, maxUv);
		minDisp = mat.get_displacement_bias() + mat.get_displacement_scale()
			* textures::sample(dispMapHdl, minUv, 0, static_cast<float>(mipmapLevel)).x;
		maxDisp = mipmapLevel == 0 ? minDisp : (mat.get_displacement_bias() + 
												textures::sample(dispMaxMIPHdl, minUv, 0, static_cast<float>(mipmapLevel) - 1.f).x
												* mat.get_displacement_scale());
	}
	return ei::Vec2{ minDisp, maxDisp };
}

float get_displacement_factor(const float maxDisplacement, const float maxShininess,
							  const float maxEdgeLen) {
	const float MAX_DISP_ERROR_FACTOR = 0.0005f / maxShininess;
	// TODO: proper factor, for now we define a fixed threshold for error
	if(maxDisplacement / maxEdgeLen > MAX_DISP_ERROR_FACTOR)
		return 1.f;
	// If we're below the threshold, we scale the factor down
	return maxDisplacement / maxEdgeLen * 1.f/MAX_DISP_ERROR_FACTOR;
}

materials::ParameterPack get_mat_params(const materials::IMaterial& mat, const ei::Vec2& uvs) {
	static thread_local std::vector<char> matDescBuffer;
	matDescBuffer.clear();
	matDescBuffer.resize(mat.get_descriptor_size(Device::CPU));
	mat.get_descriptor(Device::CPU, matDescBuffer.data());
	const auto& descriptor = *reinterpret_cast<const materials::MaterialDescriptorBase*>(matDescBuffer.data());
	materials::ParameterPack params;
	(void)materials::fetch(descriptor, uvs, &params);
	return params;
}

} // namespace


CameraDistanceOracle::CameraDistanceOracle(const float perPixelTessLevel, ConstCameraHandle cam,
										   const u32 animationPathIndex, const ei::IVec2& resolution,
										   const std::vector<ei::Mat3x4>& instTrans) :
	m_perPixelTessLevel(perPixelTessLevel),
	m_camPos(cam->get_position(animationPathIndex)),
	m_instanceTransformations(instTrans) {
	// Precompute the area of a pixel
	using namespace cameras;
	switch(cam->get_model()) {
		case CameraModel::PINHOLE: {
			const auto& pinholeCam = *reinterpret_cast<const Pinhole*>(cam);
			m_projPixelHeight = std::tan(pinholeCam.get_vertical_fov() / 2.f) / static_cast<float>(resolution.y);
		}	break;
		case CameraModel::FOCUS: {
			const auto& focusCam = *reinterpret_cast<const Focus*>(cam);
			m_projPixelHeight = focusCam.get_sensor_height();
		}	break;
		case CameraModel::ORTHOGRAPHIC:
		default:
			mAssertMsg(false, "Unknown or invalid camera model");
	}
}


u32 CameraDistanceOracle::get_edge_tessellation_level(const OpenMesh::EdgeHandle edge) const {
	mAssert(edge.is_valid());
	const auto halfedge = m_mesh->halfedge_handle(edge, 0);
	mAssert(halfedge.is_valid());
	const auto fromVertex = m_mesh->from_vertex_handle(halfedge);
	const auto toVertex = m_mesh->to_vertex_handle(halfedge);
	mAssert(fromVertex.is_valid());
	mAssert(toVertex.is_valid());
	const ei::Vec3 p0 = util::pun<ei::Vec3>(m_mesh->point(toVertex));
	const ei::Vec3 p1 = util::pun<ei::Vec3>(m_mesh->point(fromVertex));
	const ei::Vec3 n0 = util::pun<ei::Vec3>(m_mesh->normal(toVertex));
	const ei::Vec3 n1 = util::pun<ei::Vec3>(m_mesh->normal(fromVertex));
	const ei::Vec2 uv0 = util::pun<ei::Vec2>(m_mesh->texcoord2D(toVertex));
	const ei::Vec2 uv1 = util::pun<ei::Vec2>(m_mesh->texcoord2D(fromVertex));

	float maxFactor = 0.f;
	float maxEdgeLen = 0.f;
	for(const auto& instTrans : m_instanceTransformations) {
		const auto instP0 = ei::transform(p0, instTrans);
		const auto instP1 = ei::transform(p1, instTrans);
		const float dist = std::min(ei::len(instP0 - m_camPos), ei::len(instP1 - m_camPos));
		const float edgeLen = ei::len(instP0 - instP1);
		
		const float factor = edgeLen / dist;

		maxEdgeLen = std::max(maxEdgeLen, edgeLen);
		maxFactor = std::max(maxFactor, factor);
	}

	// Account for phong shading/tessellation
	float phongDisplacement = 0.f;
	if(m_usePhongTessellation) {
		// This is only a heuristic, not the exact solution!
		// Heuristic: normals at 90° == max. displacement, scale with edge length
		const float normalFactor = 1.f - ei::abs(ei::dot(n0, n1));
		phongDisplacement = normalFactor * maxEdgeLen;
	}

	// Edge might be boundary, might not be boundary, thus we have up to two faces to check
	const OpenMesh::FaceHandle f0 = m_mesh->face_handle(m_mesh->halfedge_handle(edge, 0));
	const OpenMesh::FaceHandle f1 = m_mesh->face_handle(m_mesh->halfedge_handle(edge, 1));

	float surfaceDisplacement = 0.f;
	// Account for displacement if applicable
	if(m_matHdl.is_valid()) {
		float minDisp;
		float maxDisp;

		// Compute min/max UV coordinates to later determine mipmap level
		const ei::Vec2 minUv{ std::min(uv0.u, uv1.u), std::min(uv0.v, uv1.v) };
		const ei::Vec2 maxUv{ std::max(uv0.u, uv1.u), std::max(uv0.v, uv1.v) };
		if(f0.is_valid()) {
			const materials::IMaterial& mat = *m_scenario->get_assigned_material(m_mesh->property(m_matHdl, f0));
			const ei::Vec2 minMaxDisp = get_min_max_displacement(mat, minUv, maxUv);
			minDisp = minMaxDisp.x;
			maxDisp = minMaxDisp.y;
		}
		if(f1.is_valid()) {
			const materials::IMaterial& mat = *m_scenario->get_assigned_material(m_mesh->property(m_matHdl, f1));
			const ei::Vec2 minMaxDisp = get_min_max_displacement(mat, minUv, maxUv);
			minDisp = std::min(minDisp, minMaxDisp.x);
			maxDisp = std::max(maxDisp, minMaxDisp.y);
		}

		// We'd only care if there's a discrepancy, not if everything get's displaced by the same amount
		if(minDisp != std::numeric_limits<float>::max() && maxDisp != -std::numeric_limits<float>::max())
			surfaceDisplacement = maxDisp - minDisp;
	}

	// Account for material shininess
	// TODO: use minimum roughness over edge instead of point samples!
	float maxShininess{ 0.f };
	if(f0.is_valid()) {
		const materials::IMaterial& mat = *m_scenario->get_assigned_material(m_mesh->property(m_matHdl, f0));
		if(mat.get_properties().is_emissive())
			maxShininess = 1.f;
		else
			maxShininess = ei::max(materials::pdf_max(get_mat_params(mat, uv0)),
								   materials::pdf_max(get_mat_params(mat, uv1)));
	}
	if(f1.is_valid()) {
		const materials::IMaterial& mat = *m_scenario->get_assigned_material(m_mesh->property(m_matHdl, f1));
		maxShininess = ei::max(maxShininess, ei::max(materials::pdf_max(get_mat_params(mat, uv0)),
													 materials::pdf_max(get_mat_params(mat, uv1))));
	}

	const float spannedPixels = std::max(1.f, maxFactor / m_projPixelHeight);
	const float displacementFactor = get_displacement_factor(std::max(surfaceDisplacement, phongDisplacement),
															 maxShininess, maxEdgeLen);
	return static_cast<u32>(m_perPixelTessLevel * spannedPixels * displacementFactor);
}

u32 CameraDistanceOracle::get_triangle_inner_tessellation_level(const OpenMesh::FaceHandle face) const {
	auto iter = m_mesh->cfv_ccwbegin(face);
	const OpenMesh::VertexHandle v0 = *iter;
	const OpenMesh::VertexHandle v1 = *(++iter);
	const OpenMesh::VertexHandle v2 = *(++iter);
	const ei::Vec3 p0 = util::pun<ei::Vec3>(m_mesh->point(v0));
	const ei::Vec3 p1 = util::pun<ei::Vec3>(m_mesh->point(v1));
	const ei::Vec3 p2 = util::pun<ei::Vec3>(m_mesh->point(v2));
	const ei::Vec2 uv0 = util::pun<ei::Vec2>(m_mesh->texcoord2D(v0));
	const ei::Vec2 uv1 = util::pun<ei::Vec2>(m_mesh->texcoord2D(v1));
	const ei::Vec2 uv2 = util::pun<ei::Vec2>(m_mesh->texcoord2D(v2));

	float maxFactor = 0.f;
	float maxEdgeLen = 0.f;
	for(const auto& instTrans : m_instanceTransformations) {
		const ei::Vec3 transP0 = ei::transform(p0, instTrans);
		const ei::Vec3 transP1 = ei::transform(p1, instTrans);
		const ei::Vec3 transP2 = ei::transform(p2, instTrans);

		maxEdgeLen = std::max(ei::len(transP1 - transP0),
							  std::max(ei::len(transP2 - transP0), ei::len(transP2 - transP1)));

		const ei::Vec3 centre = (transP0 + transP1 + transP2) / 3.f;
		const float distSq = ei::lensq(centre - m_camPos);
		// Compute area as another factor
		const float area = 0.5f * ei::len(ei::cross(transP1 - transP0, transP2 - transP0));
		const float factor = area / distSq;

		maxFactor = std::max(maxFactor, factor);
	}

	// Account for phong shading/tessellation
	float phongDisplacement = 0.f;
	if(m_usePhongTessellation) {
		const ei::Vec3 n0 = util::pun<ei::Vec3>(m_mesh->normal(v0));
		const ei::Vec3 n1 = util::pun<ei::Vec3>(m_mesh->normal(v1));
		const ei::Vec3 n2 = util::pun<ei::Vec3>(m_mesh->normal(v2));
		// This is only a heuristic, not the exact solution!
		// Heuristic: normals at 90° == max. displacement, scale with edge length
		const float normalFactor = 1.f - std::min(ei::abs(ei::dot(n0, n1)),
												  std::min(ei::abs(ei::dot(n0, n2)),
														   ei::abs(ei::dot(n1, n2))));
		phongDisplacement = normalFactor * maxEdgeLen;
	}

	float surfaceDisplacement = 0.f;
	// Account for displacement if applicable
	if(m_matHdl.is_valid()) {
		// Compute min/max UV coordinates to later determine mipmap level
		const ei::Vec2 minUv{
			std::min(uv0.u, std::min(uv1.u, uv2.u)),
			std::min(uv0.v, std::min(uv1.v, uv2.v))
		};
		const ei::Vec2 maxUv{
			std::max(uv0.u, std::max(uv1.u, uv2.u)),
			std::max(uv0.v, std::max(uv1.v, uv2.v))
		};

		const materials::IMaterial& mat = *m_scenario->get_assigned_material(m_mesh->property(m_matHdl, face));

		const ei::Vec2 minMaxDisp = get_min_max_displacement(mat, minUv, maxUv);
		// We'd only care if there's a discrepancy, not if everything get's displaced by the same amount
		if(minMaxDisp.x != std::numeric_limits<float>::max() && minMaxDisp.y != -std::numeric_limits<float>::max())
			surfaceDisplacement = minMaxDisp.y - minMaxDisp.x;
	}

	// Account for material shininess
	// TODO: use minimum roughness over edge instead of point samples!
	const materials::IMaterial& mat = *m_scenario->get_assigned_material(m_mesh->property(m_matHdl, face));
	const float maxShininess = mat.get_properties().is_emissive()
								? 1.f
								: ei::max(ei::max(materials::pdf_max(get_mat_params(mat, uv0)),
												  materials::pdf_max(get_mat_params(mat, uv1))),
										  materials::pdf_max(get_mat_params(mat, uv2)));

	const float spannedPixels = std::sqrt(maxFactor / (m_projPixelHeight * m_projPixelHeight));
	const float displacementFactor = get_displacement_factor(std::max(surfaceDisplacement, phongDisplacement),
															 maxShininess, maxEdgeLen);
	return static_cast<u32>(m_perPixelTessLevel * spannedPixels * displacementFactor);
}

std::pair<u32, u32> CameraDistanceOracle::get_quad_inner_tessellation_level(const OpenMesh::FaceHandle face) const {
	auto iter = m_mesh->cfv_ccwbegin(face);
	const OpenMesh::VertexHandle v0 = *iter;
	const OpenMesh::VertexHandle v1 = *(++iter);
	const OpenMesh::VertexHandle v2 = *(++iter);
	const OpenMesh::VertexHandle v3 = *(++iter);
	const ei::Vec3 p0 = util::pun<ei::Vec3>(m_mesh->point(v0));
	const ei::Vec3 p1 = util::pun<ei::Vec3>(m_mesh->point(v1));
	const ei::Vec3 p2 = util::pun<ei::Vec3>(m_mesh->point(v2));
	const ei::Vec3 p3 = util::pun<ei::Vec3>(m_mesh->point(v3));
	const ei::Vec2 uv0 = util::pun<ei::Vec2>(m_mesh->texcoord2D(v0));
	const ei::Vec2 uv1 = util::pun<ei::Vec2>(m_mesh->texcoord2D(v1));
	const ei::Vec2 uv2 = util::pun<ei::Vec2>(m_mesh->texcoord2D(v2));
	const ei::Vec2 uv3 = util::pun<ei::Vec2>(m_mesh->texcoord2D(v3));

	float maxFactorX = 0.f;
	float maxFactorY = 0.f;
	float maxEdgeLen = 0.f;
	for(const auto& instTrans : m_instanceTransformations) {
		iter = m_mesh->cfv_ccwbegin(face);

		const ei::Vec3 transP0 = ei::transform(util::pun<ei::Vec3>(p0), instTrans);
		const ei::Vec3 transP1 = ei::transform(util::pun<ei::Vec3>(p1), instTrans);
		const ei::Vec3 transP2 = ei::transform(util::pun<ei::Vec3>(p2), instTrans);
		const ei::Vec3 transP3 = ei::transform(util::pun<ei::Vec3>(p3), instTrans);

		const ei::Vec3 centre = (transP0 + transP1 + transP2 + transP3) / 4.f;
		const float dist = ei::len(centre - m_camPos);
		maxEdgeLen = std::max(ei::len(transP1 - transP0), std::max(ei::len(transP2 - transP0),
																   std::max(ei::len(transP3 - transP2),
																			ei::len(transP3 - transP1))));
		maxFactorX = std::max(maxFactorX, ei::len(transP2 - transP0) / dist);
		maxFactorY = std::max(maxFactorY, ei::len(transP1 - transP0) / dist);
	}

	// Account for phong shading/tessellation
	float phongDisplacement = 0.f;
	if(m_usePhongTessellation) {
		const ei::Vec3 n0 = util::pun<ei::Vec3>(m_mesh->normal(v0));
		const ei::Vec3 n1 = util::pun<ei::Vec3>(m_mesh->normal(v1));
		const ei::Vec3 n2 = util::pun<ei::Vec3>(m_mesh->normal(v2));
		const ei::Vec3 n3 = util::pun<ei::Vec3>(m_mesh->normal(v3));
		// This is only a heuristic, not the exact solution!
		// Heuristic: normals at 90° == max. displacement, scale with edge length
		const float normalFactor = 1.f - std::min(ei::abs(ei::dot(n0, n1)),
												  std::min(ei::abs(ei::dot(n0, n2)),
														   std::min(ei::abs(ei::dot(n3, n2)),
																	ei::abs(ei::dot(n3, n1)))));
		phongDisplacement = normalFactor * maxEdgeLen;
	}

	float surfaceDisplacement = 0.f;
	// Account for displacement if applicable
	if(m_matHdl.is_valid()) {
		// Compute min/max UV coordinates to later determine mipmap level
		const ei::Vec2 minUv{
			std::min(uv0.u, std::min(uv1.u, std::min(uv2.u, uv3.u))),
			std::min(uv0.v, std::min(uv1.v, std::min(uv2.v, uv3.v)))
		};
		const ei::Vec2 maxUv{
			std::max(uv0.u, std::max(uv1.u, std::max(uv2.u, uv3.u))),
			std::max(uv0.v, std::max(uv1.v, std::max(uv2.v, uv3.v)))
		};

		const materials::IMaterial& mat = *m_scenario->get_assigned_material(m_mesh->property(m_matHdl, face));

		const ei::Vec2 minMaxDisp = get_min_max_displacement(mat, minUv, maxUv);
		// We'd only care if there's a discrepancy, not if everything get's displaced by the same amount
		if(minMaxDisp.x != std::numeric_limits<float>::max() && minMaxDisp.y != -std::numeric_limits<float>::max())
			surfaceDisplacement = minMaxDisp.y - minMaxDisp.x;
	}

	// Account for material shininess
	// TODO: use minimum roughness over edge instead of point samples!
	const materials::IMaterial& mat = *m_scenario->get_assigned_material(m_mesh->property(m_matHdl, face));
	const float maxShininess = mat.get_properties().is_emissive()
								? 1.f
								: ei::max(ei::max(ei::max(materials::pdf_max(get_mat_params(mat, uv0)),
														  materials::pdf_max(get_mat_params(mat, uv1))),
												  materials::pdf_max(get_mat_params(mat, uv2))),
										  materials::pdf_max(get_mat_params(mat, uv3)));

	const float displacementFactor = get_displacement_factor(std::max(surfaceDisplacement, phongDisplacement),
															 maxShininess, maxEdgeLen);
	return { static_cast<u32>(m_perPixelTessLevel * displacementFactor * maxFactorX / m_projPixelHeight),
		static_cast<u32>(m_perPixelTessLevel * displacementFactor * maxFactorY / m_projPixelHeight) };
}

} // namespace mufflon::scene::tessellation
