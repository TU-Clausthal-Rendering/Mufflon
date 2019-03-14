#pragma once

#include "core/export/api.h"
#include "core/math/sampling.hpp"
#include "material_definitions.hpp"
#include "core/scene/textures/texture.hpp"

namespace mufflon { namespace scene { namespace materials {

CUDA_FUNCTION MatSampleLambert fetch(const textures::ConstTextureDevHandle_t<CURRENT_DEV>* textures,
									 const ei::Vec4* texValues,
									 int texOffset,
									 const typename MatLambert::NonTexParams& params) {
	return MatSampleLambert{Spectrum{texValues[MatLambert::ALBEDO + texOffset]}};
}


CUDA_FUNCTION math::PathSample sample(const MatSampleLambert& params,
									  const Direction& incidentTS,
									  Boundary& boundary,
									  const math::RndSet2_1& rndSet,
									  bool adjoint) {
	// Importance sampling for lambert: BRDF * cos(theta)
	Direction excidentTS = math::sample_dir_cosine(rndSet.u0, rndSet.u1).direction;
	// Copy the sign for two sided diffuse
	return math::PathSample {
		Spectrum{params.albedo},
		math::PathEventType::REFLECTED,
		excidentTS * ei::sgn(incidentTS.z),
		AngularPdf(excidentTS.z / ei::PI),
		AngularPdf(ei::abs(incidentTS.z) / ei::PI)
	};
}

CUDA_FUNCTION math::BidirSampleValue evaluate(const MatSampleLambert& params,
											  const Direction& incidentTS,
											  const Direction& excidentTS ,
											  Boundary& boundary) {
	// No transmission - already checked by material, but in a combined model we might get a call
	if(incidentTS.z * excidentTS.z < 0.0f) return math::BidirSampleValue{};
	// Two sided diffuse (therefore the abs())
	return math::BidirSampleValue {
		params.albedo / ei::PI,
		AngularPdf(ei::abs(excidentTS.z) / ei::PI),
		AngularPdf(ei::abs(incidentTS.z) / ei::PI)
	};
}

CUDA_FUNCTION Spectrum albedo(const MatSampleLambert& params) {
	return params.albedo;
}

CUDA_FUNCTION math::SampleValue emission(const MatSampleLambert& params, const scene::Direction& geoN, const scene::Direction& excident) {
	return math::SampleValue{};
}

template MaterialSampleConcept<MatSampleLambert>;
template MaterialConcept<MatLambert>;

}}} // namespace mufflon::scene::materials
