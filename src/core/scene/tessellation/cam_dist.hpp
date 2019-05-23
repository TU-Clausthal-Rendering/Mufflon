	#pragma once

#include "tessellater.hpp"
#include "util/punning.hpp"
#include "core/scene/handles.hpp"
#include <vector>

namespace mufflon::scene::tessellation {

// Tessellates purely based on distance to camera
class CameraDistanceOracle : public TessLevelOracle {
public:
	CameraDistanceOracle(const float perPixelTessLevel, ConstCameraHandle cam,
						 const u32 animationPathIndex, const ei::IVec2& resolution,
						 const std::vector<ei::Mat3x4>& instTrans);
	CameraDistanceOracle (const CameraDistanceOracle &) = delete;
	CameraDistanceOracle (CameraDistanceOracle &&) = delete;
	CameraDistanceOracle & operator=(const CameraDistanceOracle &) = delete;
	CameraDistanceOracle & operator=(CameraDistanceOracle &&) = delete;
	virtual ~CameraDistanceOracle () = default;

protected:
	u32 get_edge_tessellation_level(const OpenMesh::EdgeHandle edge) const override;
	u32 get_inner_tessellation_level(const OpenMesh::FaceHandle face) const override;

private:
	float m_perPixelTessLevel;
	float m_projPixelHeight;
	ei::Vec3 m_camPos;
	std::vector<ei::Mat3x4> m_instanceTransformations;
};

} // namespace mufflon::scene::tessellation 