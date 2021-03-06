#pragma once

#include "camera.hpp"
#include "core/export/core_api.h"
#include "util/types.hpp"
#include "core/math/sampling.hpp"
#include <ei/3dtypes.hpp>
#include <cuda_runtime.h>

namespace mufflon { namespace cameras {

/*
 * The pinhole camera is an infinite sharp perspective camera.
 */
class Pinhole : public Camera {
public:
	Pinhole(const ei::Vec3* position, const ei::Vec3* dir, const ei::Vec3* up,
			const u32 pathCount, Radians vFov, float near = 1e-10f,
			float far = 1e10f) :
		Camera(CameraModel::PINHOLE, position, dir, up, pathCount, near, far),
		m_vFov(vFov),
		m_tanVFov(std::tan(m_vFov / 2.f))
	{}

	Radians get_vertical_fov() const noexcept { return m_vFov; }
	void set_vertical_fov(Radians fov) noexcept {
		m_vFov = fov; m_tanVFov = std::tan(fov / 2);
		m_isDirty = true;
	}

	// Get the parameter bundle
	void get_parameter_pack(CameraParams* outBuffer, const Pixel& resolution, const u32 pathIndex) const final;

	// Get the required size of a parameter bundle.
	std::size_t get_parameter_pack_size() const final;
private:
	Radians m_vFov;		// Vertical field of view in radiant.
	float m_tanVFov;	// Tangents of the vfov halfed
};

// A GPU friendly packing of the camera parameters.
// TODO: smaller size and better alignment by packing one of the directions?
struct PinholeParams : public CameraParams {
	scene::Point position;
	float tanVFov;
	scene::Direction viewDir;
	float near;
	scene::Direction up;
	float far;
	ei::Vec<u16,2> resolution;	// Output buffer resoultion
};

inline CUDA_FUNCTION math::PositionSample
pinholecam_sample_position(const PinholeParams& params) {
	return { params.position, AreaPdf::infinite() };
}

inline CUDA_FUNCTION Importon
pinholecam_sample_ray(const PinholeParams& params, const Pixel& pixel, const math::RndSet2& rndSet) {
	// Get a (randomized) position in [-1,1]�
	ei::Vec2 subPixel = pixel + ei::Vec2(rndSet.u0, rndSet.u1);
	ei::Vec2 canonicalPos = subPixel / params.resolution * 2.0f - 1.0f;
	// Transform it into a point on the near plane (camera space)
	canonicalPos *= params.tanVFov;
	float aspectRatio = params.resolution.x / float(params.resolution.y);
	ei::Vec3 nPos {canonicalPos.x * aspectRatio, canonicalPos.y, 1.0f};
	// Go to world space
	scene::Direction xAxis = cross(params.viewDir, params.up);
	scene::Direction dirWorld = normalize(xAxis * nPos.x + params.up * nPos.y + params.viewDir * nPos.z);
	// Get the PDF of the pixel sampling procedure
	float pixelArea = ei::sq(2 * params.tanVFov) * aspectRatio; // includes reuse
	//float pixelArea = ei::sq(2 * params.tanVFov / float(params.resolution.y));
	float cosOut = ei::dot(params.viewDir, dirWorld);
	float pdf = 1.0f / (pixelArea * cosOut * cosOut * cosOut);
	return Importon{
		math::DirectionSample{ dirWorld, AngularPdf{ pdf } },
		pdf		// W is the same as the PDF by construction
	};
	// TODO: use the far-plane?
	// TODO: use near-plane. Support for both planes is best intrtoduced as parameters to the intersection test
}

// Compute pixel position and PDF
// position: a direction in world space.
inline CUDA_FUNCTION ProjectionResult
pinholecam_project(const PinholeParams& params, const scene::Direction& excident) {
	float cosOut = dot(params.viewDir, excident);
	if(cosOut < 0.0f) return ProjectionResult{};

	// Compute screen coordinate for this position
	ei::Vec3 xAxis = cross(params.viewDir, params.up);
	ei::Vec2 uv{ dot(xAxis, excident), dot(params.up, excident) };
	uv /= cosOut * params.tanVFov;
	float aspectRatio = params.resolution.x / float(params.resolution.y);
	uv.x /= aspectRatio;

	// On screen?
	if(!(uv.x > -1 && uv.x <= 1 && uv.y > -1 && uv.y <= 1))
		return ProjectionResult{};

	Pixel pixelCoord{ floor((uv * 0.5f + 0.5f) * params.resolution) };
	// Need to check the boundaries. In rare cases values like uv.x==-0.999999940
	// cause pixel coordinates equal to the resolution.
	if(pixelCoord.x >= params.resolution.x) { pixelCoord.x = u32(params.resolution.x) - 1; }
	if(pixelCoord.y >= params.resolution.y) { pixelCoord.y = u32(params.resolution.y) - 1; }

	float pixelArea = ei::sq(2.0f * params.tanVFov) * aspectRatio; // includes reuse
	//float pixelArea = ei::sq(2 * params.tanVFov / float(params.resolution.y));
	float pdf = 1.0f / (pixelArea * cosOut * cosOut * cosOut);

	return ProjectionResult{
		pixelCoord,
		AngularPdf{ pdf },
		pdf
	};
}

// Compute the PDF value only
// direction: a direction in world space.
/*inline CUDA_FUNCTION float
evaluate_pdf(const PinholeParams& params, const ei::Vec2& resolution, const scene::Direction& direction) {
	// TODO: only if inside frustum
	float aspectRatio = resolution.x / resolution.y;
	float cosAtCam = dot(params.viewDir, direction);
	float pixelArea = ei::sq(2.0f * params.tanVFov) * aspectRatio;
	return 1.0f / (pixelArea * cosAtCam * cosAtCam * cosAtCam);
}*/

static_assert(sizeof(PinholeParams) <= MAX_CAMERA_PARAM_SIZE,
	"MAX_CAMERA_PARAM_SIZE outdated please change the number in the header file.");

}} // namespace mufflon::cameras