#pragma once

#include "core/export/core_api.h"
#include "core/math/sampling.hpp"
#include "material_definitions.hpp"
#include "core/scene/textures/texture.hpp"

namespace mufflon { namespace scene { namespace materials {

inline CUDA_FUNCTION MatSampleEmissive fetch(const textures::ConstTextureDevHandle_t<CURRENT_DEV>* /*textures*/,
									  const ei::Vec4* texValues,
									  int texOffset,
									  const typename MatEmissive::NonTexParams& params) {
	return MatSampleEmissive{
		Spectrum{texValues[MatEmissive::EMISSION+texOffset]} * params.scale
	};
}

inline CUDA_FUNCTION math::PathSample sample(const MatSampleEmissive& /*params*/,
									  const Direction& /*incidentTS*/,
									  Boundary& /*boundary*/,
									  const math::RndSet2_1& /*rndSet*/,
									  bool /*adjoint*/) {
	return math::PathSample{};
}

inline CUDA_FUNCTION math::BidirSampleValue evaluate(const MatSampleEmissive& /*params*/,
											  const Direction& /*incidentTS*/,
											  const Direction& /*excidentTS*/,
											  Boundary& /*boundary*/) {
	return math::BidirSampleValue{};
}

// The albedo routine
inline CUDA_FUNCTION Spectrum albedo(const MatSampleEmissive& /*params*/) {
	// Return 0 to force layered models to never sample this.
	return Spectrum{0.0f};
}

inline CUDA_FUNCTION math::SampleValue emission(const MatSampleEmissive& params, const scene::Direction& geoN, const scene::Direction& excident) {
	float cosOut = dot(geoN, excident);
	if(cosOut <= 0.0f) return math::SampleValue{};
	return { params.radiance, AngularPdf{cosOut / ei::PI} };
}

inline CUDA_FUNCTION float pdf_max(const MatSampleEmissive& /*params*/) {
	return 0.0f;
}

template class MaterialSampleConcept<MatSampleEmissive>;
template class MaterialConcept<MatEmissive>;

}}} // namespace mufflon::scene::materials
