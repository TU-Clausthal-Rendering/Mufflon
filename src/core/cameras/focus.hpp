#pragma once

#include "camera.hpp"
#include "core/export/core_api.h"
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
	Focus(const ei::Vec3* position, const ei::Vec3* dir, const ei::Vec3* up,
		  const u32 pathCount, float focalLength, float focusDist, float lensRad,
		  float sensorHeight, float near = 1e-4f,
		  float far = 1e10f) :
		Camera(CameraModel::FOCUS, position, dir, up, pathCount, near, far),
		m_focalLength(focalLength),
		m_focusDistance(focusDist),
		m_lensRadius(lensRad),
		m_sensorHeight(sensorHeight)
	{}

	float get_focus_distance() const noexcept { return m_focusDistance; }
	float get_focal_length() const noexcept { return m_focalLength; }
	float get_lens_radius() const noexcept { return m_lensRadius; }
	float get_aperture_in_f_stops() const noexcept { return m_focalLength / (2.f * m_lensRadius); }
	float get_sensor_height() const noexcept { return m_sensorHeight; }
	void set_focus_distance(float distance) noexcept {
		m_focusDistance = distance;
		m_isDirty = true;
	}
	void set_focal_length(float length) noexcept {
		m_focalLength = length;
		m_isDirty = true;
	}
	void set_lens_radius(float radius) noexcept {
		m_lensRadius = radius;
		m_isDirty = true;
	}
	void set_aperture_in_f_stops(float fStops) noexcept {
		m_lensRadius = m_focalLength / (2.f * fStops);
		m_isDirty = true;
	}
	void set_sensor_height(float height) noexcept {
		m_sensorHeight = height;
		m_isDirty = true;
	}

	// Get the parameter bundle
	void get_parameter_pack(CameraParams* outBuffer, const Pixel& resolution, const u32 pathIndex) const final;

	// Get the required size of a parameter bundle.
	std::size_t get_parameter_pack_size() const final;
private:
	float m_focalLength;	// Focal length in meter
	float m_focusDistance;	// Focus distance in meter
	float m_lensRadius;		// Lens radius in meter
	float m_sensorHeight;	// Sensor height in meter
};

// A GPU friendly packing of the camera parameters.
// TODO: smaller size and better alignment by packing one of the directions?
struct FocusParams : public CameraParams {
	scene::Point position;
	float sensorDistance;
	scene::Direction viewDir;
	float far;
	scene::Direction up;
	float sensorHalfHeight;
	float focusDistance;		// How far away is a sharp point from the lens
	float focalLength;			// Focal length of the lens
	float lensRadius;
	ei::Vec<u16, 2> resolution;	// Output buffer resoultion
};

inline CUDA_FUNCTION math::PositionSample
focuscam_sample_position(const FocusParams& params, const math::RndSet2& rndSet) {
	// First sample a point on a unit disk; for that, we project the unit square onto it (see PBRT)
	// TODO: is this cheaper than sqrt(r) * sin/cos(theta)?
	// Get a (randomized) position in [-1,1]²
	const ei::Vec2 squarePoint{ 2.f * rndSet.u0 - 1.f, 2.f * rndSet.u1 - 1.f };
	float theta, r;
	if(ei::abs(squarePoint.x) > ei::abs(squarePoint.y)) {
		r = squarePoint.x;
		theta = ei::PI * squarePoint.y / (4.f * squarePoint.x);
	} else {
		r = squarePoint.y;
		theta = ei::PI / 2.f - ei::PI * squarePoint.x / (4.f * squarePoint.y);
	}
	// Adjust the sample to be within the lens radius
	const ei::Vec2 diskSample = params.lensRadius * r * ei::Vec2{ std::cos(theta), std::sin(theta) };
	// Transform the sample to camera space
	const ei::Vec3 nPos{ diskSample.x, diskSample.y, params.sensorDistance };
	// Go to world space
	const ei::Vec3 xAxis = ei::cross(params.viewDir, params.up);
	const ei::Vec3 yAxis = ei::cross(params.viewDir, xAxis);
	const ei::Vec3 dirWorld = xAxis * nPos.x + yAxis * nPos.y + params.viewDir * nPos.z;
	const ei::Vec3 lensPos = params.position + dirWorld;
	// Compute the PDF for sampling the given position on the lens
	const AreaPdf lensPosPdf = AreaPdf{ 1.f / (ei::PI * params.lensRadius * params.lensRadius) };

	return math::PositionSample{ lensPos, lensPosPdf };
}

inline CUDA_FUNCTION Importon
focuscam_sample_ray(const FocusParams& params, const scene::Point& exitPosWorld,
					const Pixel& pixel, const math::RndSet2& rndSet) {
	// First we sample the sensor position, then we use the sensor and the lens position to compute the direction
	// Get a (randomized) position in [-1,1]²
	const ei::Vec2 subPixel = pixel + ei::Vec2(rndSet.u0, rndSet.u1);
	const ei::Vec2 canonicalPos = 2.f * subPixel / params.resolution - 1.0f;
	const float sensorHalfHeight = params.sensorHalfHeight;
	const float aspectRatio = params.resolution.x / float(params.resolution.y);
	// Scale to fit the sensor
	const ei::Vec2 nPos = sensorHalfHeight * ei::Vec2{ canonicalPos.x * aspectRatio, canonicalPos.y };
	// Go to world space
	const ei::Vec3 xAxis = ei::cross(params.viewDir, params.up);
	const ei::Vec3 yAxis = ei::cross(params.viewDir, xAxis);
	const ei::Vec3 sensorPos = params.position - xAxis * nPos.x + yAxis * nPos.y;

	// To compute the ray direction, we compute the intersection
	// point between an unperturbed ray through the lens centre
	// and the focus plane
	const ei::Vec3 lensCentre = params.position + params.viewDir * params.sensorDistance;
	const ei::Vec3 sensorToLensCentre = lensCentre - sensorPos;
	const ei::Vec3 focalPlane = params.position + params.viewDir * params.focusDistance;
	const float sDotV = ei::dot(sensorToLensCentre, params.viewDir);
	mAssert(sDotV != 0.f);
	const float t = ei::dot(focalPlane - sensorPos, params.viewDir) / sDotV;
	const auto focalPoint = sensorPos + sensorToLensCentre * t;
	const auto direction = ei::normalize(focalPoint - exitPosWorld);
	const float cosTheta = ei::dot(direction, params.viewDir);

	// How and why the PDF is like this: see projection code (below)
	const float sensorArea = 4.f * sensorHalfHeight * sensorHalfHeight / aspectRatio;
	const float pdf = params.sensorDistance * params.sensorDistance / (sensorArea * cosTheta * cosTheta * cosTheta);

	return Importon{
		math::DirectionSample{ direction, AngularPdf{ pdf } },
		pdf
	};
}

// Compute pixel position and PDF
// position: a direction in world space.
inline CUDA_FUNCTION ProjectionResult
focuscam_project(const FocusParams& params, const scene::Point& lensPoint,
				 const scene::Direction& excident) {
	const float cosTheta = ei::dot(excident, params.viewDir);
	const float sensorDist = params.sensorDistance;
	// Early-reject if we try to project from the wrong side
	if(cosTheta <= 0.f)
		return ProjectionResult{};

	// Compute the focus point; this is the intersection with the focus plane at focusDistance
	// Since we have the angle with the view direction this is a simple division
	const ei::Vec3 focalPoint = lensPoint + excident * (params.focusDistance / cosTheta);
	// With this we can compute the lens-centre ray
	const ei::Vec3 lensCentre = params.position + params.viewDir * sensorDist;
	// Scale the direction with the sensor distance to improve FP precision
	const ei::Vec3 focalToLensCentre = sensorDist * (lensCentre - focalPoint);
	// The intersection with the sensor plane gives us the sensor point
	const float fDotV = ei::dot(focalToLensCentre, params.viewDir);
	if(fDotV == 0.f)
		return ProjectionResult{};
	const float t = ei::dot(params.position - lensCentre, params.viewDir) / fDotV;
	const ei::Vec3 sensorPoint = lensCentre + t * focalToLensCentre;

	// Map to UVs
	const ei::Vec3 xAxis = ei::cross(params.viewDir, params.up);
	const ei::Vec3 yAxis = ei::cross(params.viewDir, xAxis);
	const ei::Vec3 sensorDir = sensorPoint - params.position;
	// Flip the Y coordinate to flip the image properly
	ei::Vec2 uv{ ei::dot(sensorDir, xAxis), ei::dot(sensorDir, yAxis) };
	// Compute screen coordinate for this position
	const float sensorHalfHeight = params.sensorHalfHeight;
	uv /= sensorHalfHeight;		// Bring UV into [-a,a]x[-1,1]
	const float aspectRatio = params.resolution.y / float(params.resolution.x);
	uv.x *= -aspectRatio;		// Bring UV into [-1,1]x[-1,1]

	// On screen?
	if(!(uv.x > -1 && uv.x <= 1 && uv.y > -1 && uv.y <= 1))
		return ProjectionResult{};

	Pixel pixelCoord{ floor((uv * 0.5f + 0.5f) * params.resolution) };
	// Need to check the boundaries. In rare cases values like uv.x==-0.999999940
	// cause pixel coordinates in equal to the resolution.
	if(pixelCoord.x >= params.resolution.x) { pixelCoord.x = u32(params.resolution.x) - 1; }
	if(pixelCoord.y >= params.resolution.y) { pixelCoord.y = u32(params.resolution.y) - 1; }

	/* The thin-lens model has two sampling events: the origin is determined on the (virtual)
	 * lens and has the uniform area PDF p(A_L) = 1/(π * r²).
	 * The second sampling is slightly more involved: we start by sampling in a given pixel
	 * with the PDF p(A_S) = 1 / A_Sensor = RES_X * RES_Y / (W * H).
	 * To transform this to a directional PDF, we us the solid angle conversion:
	 * p(ω) = RES_X * RES_Y * d² / (W * H * cos(Θ) * cos³(θ)), where d is the distance
	 * between lens and sensor centre, and θ is the angle between view direction and
	 * OUTGOING direction, ie. the vector from the lens sample point into the scene.
	 * The lone cos(θ) comes from the angle conversion, the cos³(θ) comes from the
	 * domain transition and the associated Jacobian.
	 * Now, the lone cos(θ) gets cancelled out when you take into account irradiance
	 * transfer, which falls off with exactly the same cosine. Additionally, the fact that
	 * we sample for one pixel is already compensated for in our renderers and is thus not
	 * part of the result either. Finally, the camera response function is equal to the PDF
	 * for us.
	 */
	const float sensorArea = 4.f * sensorHalfHeight * sensorHalfHeight / aspectRatio;
	const float pdf = params.sensorDistance * params.sensorDistance / (sensorArea * cosTheta * cosTheta * cosTheta);

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


static_assert(sizeof(FocusParams) <= MAX_CAMERA_PARAM_SIZE,
	"MAX_CAMERA_PARAM_SIZE outdated please change the number in the header file.");

}} // namespace mufflon::cameras
