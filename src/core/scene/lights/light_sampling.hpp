#pragma once

#include "lights.hpp"
#include "ei/vector.hpp"
#include "core/scene/types.hpp"
#include <cuda_runtime.h>

namespace mufflon::scene::lights {

struct Photon {
	ei::Vec3 position;
	AreaPdf posPdf;
	ei::Vec3 direction;
	AngularPdf dirPdf;
};

struct NextEventEstimation {
	ei::Vec3 position;
	AreaPdf posPdf;
	ei::Vec3 intensity;
	AngularPdf dirPdf;
};

// TODO
// Sample a light source for either one or many photons
__host__ __device__ inline Photon sample_light(const PointLight& light,
											   float r0, float r1) {
	return {};
}
__host__ __device__ inline Photon sample_light(const SpotLight& light,
											   float r0, float r1) {
	return {};
}
__host__ __device__ inline Photon sample_light(const AreaLightTriangle& light,
											   float r0, float r1) {
	return {};
}
__host__ __device__ inline Photon sample_light(const AreaLightQuad& light,
											   float r0, float r1) {
	return {};
}
__host__ __device__ inline Photon sample_light(const AreaLightSphere& light,
											   float r0, float r1) {
	return {};
}
__host__ __device__ inline Photon sample_light(const DirectionalLight& light,
											   float r0, float r1) {
	return {};
}
template < Device dev >
__host__ __device__ inline Photon sample_light(const EnvMapLight<dev>& light,
											   float r0, float r1) {
	return {};
}

} // namespace mufflon::scene::lights