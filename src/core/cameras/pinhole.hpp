#pragma once

#include "camera.hpp"
#include <ei/3dtypes.hpp>
#include <cuda_runtime.h>

namespace mufflon { namespace cameras {

/*
 * The pinhole camera is an infinite sharp perspective camera.
 */
class Pinhole : public Camera {
public:
	void set_vertical_fov(Radians fov) noexcept { m_vFov = fov; m_tanVFov = tan(fov / 2); }

	// Get the parameter bundle
	void get_parameter_pack(CameraParams* outBuffer, Device dev) const final;

	// Get the required size of a parameter bundle.
	std::size_t get_parameter_pack_size() const final;
private:
	Radians m_vFov;		// Vertical field of view in radiant.
	float m_tanVFov;	// Tangents of the vfov halfed
};

// A GPU friendly packing of the camera parameters.
struct PinholeParams : public CameraParams {
	scene::Point position;
	scene::Direction xAxis;
	float near;
	scene::Direction up;
	float far;
	scene::Direction viewDir;
	float tanVFov;
};

__host__ __device__ inline RaySample
pinholecam_sample_ray(const PinholeParams& params, const Pixel& coord, const ei::Vec2& resolution, const RndSet& rndSet) {
	// Get a (randomized) position in [-1,1]²
	ei::Vec2 subPixel = coord + ei::Vec2(rndSet.u0, rndSet.u1);
	ei::Vec2 canonicalPos = subPixel / resolution * 2.0f - 1.0f;
	// Transform it into a point on the near plane (camera space)
	canonicalPos *= params.tanVFov;
	float aspectRatio = resolution.x / resolution.y;
	ei::Vec3 nPos {canonicalPos.x * aspectRatio, canonicalPos.y, 1.0f};
	// Go to world space
	ei::Vec3 dirWorld = params.xAxis * nPos.x + params.up * nPos.y + params.viewDir * nPos.z;
	ei::Vec3 dirWorldNormalized = normalize(dirWorld);
	// Get the PDF of the above procedure
	float pixelArea = ei::sq(2 * params.tanVFov) * aspectRatio;
	float pdf = 1.0f / (pixelArea * dirWorldNormalized.z * dirWorldNormalized.z * dirWorldNormalized.z);
	return RaySample{
		params.position + dirWorld * params.near,
		pdf,
		dirWorldNormalized,
		pdf		// W is the same as the PDF by construction
	};
	// TODO: use the far-plane?
}

// Compute pixel position and PDF
// position: a direction in world space.
__host__ __device__ inline ProjectionResult
pinholecam_project(const PinholeParams& params, const ei::Vec2& resolution, const scene::Point& position) {
	ei::Vec3 camToPosDir = position - params.position;
	float w = dot(params.viewDir, camToPosDir);
	// Clip on near plane
	if(w < params.near) return ProjectionResult{};
	// TODO: use the far-plane?

	// Compute screen coordinate for this position
	ei::Vec2 uv{ dot(params.xAxis, camToPosDir), dot(params.up, camToPosDir) };
	uv /= w * params.tanVFov;
	float aspectRatio = resolution.x / resolution.y;
	uv.x /= aspectRatio;

	// On screen?
	if(!(uv.x > -1 && uv.x <= 1 && uv.y > -1 && uv.y <= 1))
		return ProjectionResult{};

	Pixel pixelCoord{ floor((uv * -0.5f + 0.5f) * resolution) };
	// Need to check the boundaries. In rare cases values like uv.x==-0.999999940
	// cause pixel coordinates in equal to the resolution.
	if(pixelCoord.x >= resolution.x) { pixelCoord.x = u32(resolution.x) - 1; }
	if(pixelCoord.y >= resolution.y) { pixelCoord.y = u32(resolution.y) - 1; }

	float cosAtCam = w / len(camToPosDir);
	float pixelArea = ei::sq(2.0f * params.tanVFov) * aspectRatio;
	float pdf = 1.0f / (pixelArea * cosAtCam * cosAtCam * cosAtCam);

	return ProjectionResult{
		pixelCoord,
		pdf,
		pdf
	};
}

// Compute the PDF value only
// direction: a direction in world space.
/*__host__ __device__ float
evaluate_pdf(const PinholeParams& params, const ei::Vec2& resolution, const scene::Direction& direction) {
	// TODO: only if inside frustum
	float aspectRatio = resolution.x / resolution.y;
	float cosAtCam = dot(params.viewDir, direction);
	float pixelArea = ei::sq(2.0f * params.tanVFov) * aspectRatio;
	return 1.0f / (pixelArea * cosAtCam * cosAtCam * cosAtCam);
}*/

}} // namespace mufflon::cameras