#include "cam_dist.hpp"
#include "core/cameras/camera.hpp"
#include "core/cameras/pinhole.hpp"
#include "core/cameras/focus.hpp"

namespace mufflon::scene::tessellation {

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
			m_projPixelHeight = std::tan(focusCam.get_vertical_fov() / 2.f) / static_cast<float>(resolution.y);
		}	break;
		case CameraModel::ORTHOGRAPHIC:
		default:
			mAssertMsg(false, "Unknown or invalid camera model");
	}
}


u32 CameraDistanceOracle::get_edge_tessellation_level(const OpenMesh::EdgeHandle edge) const {
	const auto halfedge = m_mesh->halfedge_handle(edge, 0);
	const auto p0 = util::pun<ei::Vec3>(m_mesh->point(m_mesh->to_vertex_handle(halfedge)));
	const auto p1 = util::pun<ei::Vec3>(m_mesh->point(m_mesh->from_vertex_handle(halfedge)));

	float maxFactor = 0.f;
	for(const auto& instTrans : m_instanceTransformations) {
		const auto instP0 = ei::transform(p0, instTrans);
		const auto instP1 = ei::transform(p1, instTrans);
		const float dist = std::min(ei::len(instP0 - m_camPos), ei::len(instP1 - m_camPos));
		const float edgeLen = ei::len(instP0 - instP1);
		// TODO
		const float factor = edgeLen / dist;

		maxFactor = std::max(maxFactor, factor);
	}

	return static_cast<u32>(m_perPixelTessLevel * std::max(1.f, maxFactor / m_projPixelHeight));
}

u32 CameraDistanceOracle::get_inner_tessellation_level(const OpenMesh::FaceHandle face) const {
	static thread_local std::vector<ei::Vec3> transformedPoints;

	float maxFactor = 0.f;
	for(const auto& instTrans : m_instanceTransformations) {
		ei::Vec3 centre{ 0.f };
		transformedPoints.clear();
		for(auto iter = m_mesh->cfv_ccwbegin(face); iter.is_valid(); ++iter) {
			transformedPoints.push_back(ei::transform(util::pun<ei::Vec3>(m_mesh->point(*iter)), instTrans));
			centre += transformedPoints.back();
		}
		const float distSq = ei::lensq((centre / static_cast<float>(transformedPoints.size())) - m_camPos);

		// Compute area as another factor
		float area;
		if(transformedPoints.size() == 3u) {
			const auto& p0 = transformedPoints[0u];
			const auto& p1 = transformedPoints[1u];
			const auto& p2 = transformedPoints[2u];
			// Triangle
			area = 0.5f * ei::len(ei::cross(p1 - p0, p2 - p0));
		} else {
			// Quad; we assume that the quad is planar, if not we overestimate the area
			const auto& p0 = transformedPoints[0u];
			const auto& p1 = transformedPoints[1u];
			const auto& p2 = transformedPoints[2u];
			const auto& p3 = transformedPoints[3u];
			area = 0.5f * (ei::len(ei::cross(p1 - p0, p2 - p0)) + ei::len(ei::cross(p2 - p0, p3 - p0)));
		}

		// TODO
		const float factor = area / distSq;

		maxFactor = std::max(maxFactor, factor);
	}

	return static_cast<u32>(m_perPixelTessLevel * std::max(1.f, std::sqrt(maxFactor / (m_projPixelHeight * m_projPixelHeight))));
}

} // namespace mufflon::scene::tessellation