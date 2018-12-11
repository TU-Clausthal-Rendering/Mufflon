#pragma once

#include "camera.hpp"
#include "core/export/api.h"
#include "util/types.hpp"
#include "core/math/sampling.hpp"
#include <ei/3dtypes.hpp>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace mufflon {
namespace cameras {

/*
 * The pinhole camera is an infinite sharp perspective camera.
 */
class Focus : public Camera {
public:
	Focus() = default;
	Focus(ei::Vec3 position, ei::Vec3 dir, ei::Vec3 up,
			float focalLength, float focusDist, float lensRad,
			float sensorHeight, float near = 1e-10f,
			float far = 1e10f) :
		Camera(std::move(position), std::move(dir),
			   std::move(up), near, far),
		// TODO
		m_vFov(2.f * std::atan(sensorHeight / (2.f * focalLength))),
		m_tanVFov(std::tan(m_vFov / 2.f)),
		m_focalLength(focalLength),
		m_focusDistance(focusDist),
		m_lensRadius(lensRad),
		m_sensorHeight(sensorHeight)
	{}

	constexpr float get_focus_distance() const noexcept { return m_focusDistance; }
	constexpr float get_focal_length() const noexcept { return m_focalLength; }
	constexpr float get_lens_radius() const noexcept { return m_lensRadius; }
	constexpr float get_sensor_height() const noexcept { return m_sensorHeight; }
	void set_focus_distance(float distance) noexcept { m_focusDistance = distance; }
	void set_focal_length(float length) noexcept { m_focalLength = length; }
	void set_lens_radius(float radius) noexcept { m_lensRadius = radius; }
	void set_sensor_height(float height) noexcept { m_sensorHeight = height; }

	// Get the parameter bundle
	void get_parameter_pack(CameraParams* outBuffer, Device dev, const Pixel& resolution) const final;

	// Get the required size of a parameter bundle.
	std::size_t get_parameter_pack_size() const final;
private:
	Radians m_vFov;			// Vertical field of view in radiant.
	float m_tanVFov;		// Tangents of the vfov halfed
	float m_focalLength;	// Focal length in meter
	float m_focusDistance;	// Focus distance in meter
	float m_lensRadius;		// Lens radius in meter
	float m_sensorHeight;	// Sensor height in meter
};

// A GPU friendly packing of the camera parameters.
// TODO: smaller size and better alignment by packing one of the directions?
struct FocusParams : public CameraParams {
	scene::Point position;
	float tanVFov;
	scene::Direction viewDir;
	float far;
	scene::Direction up;
	__half sensorHalfHeight;
	__half sensorDistance;
	float focusDistance;		// How far away is a sharp point from the lens
	float focalLength;			// Focal length of the lens
	float lensRadius;
	ei::Vec<u16, 2> resolution;	// Output buffer resoultion
};

CUDA_FUNCTION math::PositionSample
focuscam_sample_position(const FocusParams& params, const math::RndSet2& rndSet) {
	// First sample a point on a unit disk; for that, we project the unit square onto it (see PBRT)
	// TODO: is this cheaper than sqrt(r) * sin/cos(theta)?
	// Get a (randomized) position in [-1,1]�
	const ei::Vec2 squarePoint{ 2.f * rndSet.u0 - 1.f, 2.f * rndSet.u1 - 1.f };
	float theta, r;
	if (ei::abs(squarePoint.x) > ei::abs(squarePoint.y)) {
		r = squarePoint.x;
		theta = ei::PI * squarePoint.x / (4.f * squarePoint.y);
	}
	else {
		r = squarePoint.y;
		theta = ei::PI / 2.f - ei::PI * squarePoint.y / (4.f * squarePoint.x);
	}
	// Adjust the sample to be within the lens radius
	const ei::Vec2 diskSample = params.lensRadius * r * ei::Vec2{ cosf(theta), sin(theta) };
	// Transform the sample to camera space
	const ei::Vec3 nPos{ diskSample.x, diskSample.y, 1.0f };
	// Go to world space
	const ei::Vec3 xAxis = ei::cross(params.viewDir, params.up);
	const ei::Vec3 yAxis = ei::cross(params.viewDir, xAxis);
	const ei::Vec3 dirWorld = xAxis * nPos.x + yAxis * nPos.y + params.viewDir * nPos.z;
	const ei::Vec3 pixelPos = params.position + dirWorld * __half2float(params.sensorDistance);
	// Compute the PDF for sampling the given position on the lens
	const AreaPdf lensPosPdf = AreaPdf{ 1.f / ei::PI };

	return math::PositionSample{ pixelPos, lensPosPdf };
}

CUDA_FUNCTION Importon
focuscam_sample_ray(const FocusParams& params, const scene::Point& exitPosWorld,
					const Pixel& pixel, const math::RndSet2& rndSet) {
	// First we sample the sensor position, then we use the sensor and the lens position to compute the direction
	// Get a (randomized) position in [-1,1]�
	const ei::Vec2 subPixel = pixel + ei::Vec2(rndSet.u0, rndSet.u1);
	ei::Vec2 canonicalPos = subPixel / params.resolution * 2.0f - 1.0f;
	// Transform it into a point on the near plane (camera space)
	canonicalPos *= params.tanVFov;
	const float sensorHalfHeight = __half2float(params.sensorHalfHeight);
	const float aspectRatio = params.resolution.x / float(params.resolution.y);
	// Scale to fit the sensor
	const ei::Vec3 nPos{ sensorHalfHeight * ei::Vec2{canonicalPos.x * aspectRatio, canonicalPos.y} };
	// Go to world space
	const ei::Vec3 xAxis = ei::cross(params.viewDir, params.up);
	const ei::Vec3 yAxis = ei::cross(params.viewDir, xAxis);
	const ei::Vec3 pixelPos = nPos.x * xAxis + nPos.y * yAxis;

	// Compute the focal point for the pixel
	// The actual focal point for the direction has to be computed first
	const ei::Vec3 pixelToLens = ei::normalize(exitPosWorld - pixelPos);
	const float cosPixel = ei::dot(pixelToLens, params.viewDir);
	const float sensorDistance = __half2float(params.sensorDistance);
	float pixelSharpDist = sensorDistance;
	if(cosPixel != 1.f)
		pixelSharpDist = sensorDistance + canonicalPos.y / atan(acos(cosPixel));
	// Compute the distance on the other side of the lens (where an object has to be to be sharp)
	const float focusDistance = pixelSharpDist * params.focalLength / (pixelSharpDist - params.focalLength);
	// Compute the focal point for this pixel (cam distance(nearplane) + focal distance along view direction)
	const ei::Vec3 focalPoint = params.position + (sensorDistance + focusDistance) * params.viewDir;
	// Now we can compute the ray direction
	const ei::Vec3 direction = ei::normalize(focalPoint - exitPosWorld);

	// Compute the directional PDF (see PBRT)
	float pixelArea = ei::sq(2.f * params.tanVFov) * aspectRatio;
	const float cosPixelDir = ei::dot(direction, params.viewDir);
	const AngularPdf pixelPdf{ 1.f / (pixelArea * cosPixelDir * cosPixelDir * cosPixelDir) };

	return Importon{
		math::DirectionSample{ direction, pixelPdf },
		static_cast<float>(pixelPdf) // TODO: is this correct?
	};
}

// Compute pixel position and PDF
// position: a direction in world space.
CUDA_FUNCTION ProjectionResult
focuscam_project(const FocusParams& params, const scene::Point& lensPoint, const scene::Direction& excident) {
	// Project the ray back onto the sensor
	// Taken from PBRT
	const float cosTheta = ei::dot(excident, params.viewDir);
	const float focusDistance = params.focusDistance / cosTheta;
	const ei::Vec3 focusPoint = lensPoint - focusDistance * excident;
	const ei::Vec3 sensorPoint = focusPoint - (__half2float(params.sensorDistance) - focusDistance) * params.viewDir;

	// Map to UVs
	const ei::Vec3 xAxis = ei::cross(params.viewDir, params.up);
	const ei::Vec3 yAxis = ei::cross(params.viewDir, xAxis);
	const ei::Vec3 sensorDir = sensorPoint - params.position;
	ei::Vec2 uv{ ei::dot(sensorDir, xAxis), ei::dot(sensorDir, yAxis) };

	// Compute screen coordinate for this position
	const float sensorHalfHeight = __half2float(params.sensorHalfHeight);
	float aspectRatio = params.resolution.x / float(params.resolution.y);
	uv.x *= 1.f / aspectRatio;
	uv *= sensorHalfHeight;

	// On screen?
	if(!(uv.x > -1 && uv.x <= 1 && uv.y > -1 && uv.y <= 1))
		return ProjectionResult{};

	Pixel pixelCoord{ floor((uv * -0.5f + 0.5f) * params.resolution) };
	// Need to check the boundaries. In rare cases values like uv.x==-0.999999940
	// cause pixel coordinates in equal to the resolution.
	if(pixelCoord.x >= params.resolution.x) { pixelCoord.x = u32(params.resolution.x) - 1; }
	if(pixelCoord.y >= params.resolution.y) { pixelCoord.y = u32(params.resolution.y) - 1; }

	float pixelArea = ei::sq(2.0f * params.tanVFov) * aspectRatio;
	float pdf = 1.0f / (pixelArea * cosTheta * cosTheta * cosTheta);

	return ProjectionResult{
		pixelCoord,
		AngularPdf{ pdf },
		pdf
	};
}

// Compute the PDF value only
// direction: a direction in world space.
/*CUDA_FUNCTION float
evaluate_pdf(const PinholeParams& params, const ei::Vec2& resolution, const scene::Direction& direction) {
	// TODO: only if inside frustum
	float aspectRatio = resolution.x / resolution.y;
	float cosAtCam = dot(params.viewDir, direction);
	float pixelArea = ei::sq(2.0f * params.tanVFov) * aspectRatio;
	return 1.0f / (pixelArea * cosAtCam * cosAtCam * cosAtCam);
}*/

}
} // namespace mufflon::cameras