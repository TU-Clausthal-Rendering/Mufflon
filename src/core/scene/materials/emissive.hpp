#pragma once

#include "core/export/api.h"
#include "core/math/sampling.hpp"
#include "material_definitions.hpp"
#include "core/scene/textures/texture.hpp"

namespace mufflon { namespace scene { namespace materials {

CUDA_FUNCTION MatSampleEmissive fetch(const textures::ConstTextureDevHandle_t<CURRENT_DEV>* textures,
									  const ei::Vec4* texValues,
									  int texOffset,
									  const typename MatEmissive::NonTexParams& params) {
	return MatSampleEmissive{
		Spectrum{texValues[MatEmissive::EMISSION+texOffset]} * params.scale
	};
}

CUDA_FUNCTION math::PathSample sample(const MatSampleEmissive& params,
									  const Direction& incidentTS,
									  Boundary& boundary,
									  const math::RndSet2_1& rndSet,
									  bool adjoint) {
	return math::PathSample{};
}

CUDA_FUNCTION math::EvalValue evaluate(const MatSampleEmissive& params,
									   const Direction& incidentTS,
									   const Direction& excidentTS ,
									   Boundary& boundary) {
	return math::EvalValue{};
}

// The albedo routine
CUDA_FUNCTION Spectrum albedo(const MatSampleEmissive& params) {
	// Return 0 to force layered models to never sample this.
	return Spectrum{0.0f};
}

CUDA_FUNCTION Spectrum emission(const MatSampleEmissive& params, const scene::Direction& geoN, const scene::Direction& excident) {
	return dot(geoN, excident) > 0.0f ? params.radiance : Spectrum{0.0f};
}

template MaterialSampleConcept<MatSampleEmissive>;
template MaterialConcept<MatEmissive>;

}}} // namespace mufflon::scene::materials
