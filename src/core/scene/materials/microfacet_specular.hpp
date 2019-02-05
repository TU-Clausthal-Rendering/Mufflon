#pragma once

#include "core/export/api.h"
#include "core/math/sampling.hpp"
#include "core/scene/textures/interface.hpp"
#include "material_definitions.hpp"
#include "microfacet_base.hpp"

namespace mufflon { namespace scene { namespace materials {

CUDA_FUNCTION MatSampleTorrance fetch(const textures::ConstTextureDevHandle_t<CURRENT_DEV>* textures,
									  const ei::Vec4* texValues,
									  int texOffset,
									  const typename MatTorrance::NonTexParams& params) {
	ei::Vec2 roughness { texValues[MatTorrance::ROUGHNESS+texOffset].x };
	if(get_texture_channel_count(textures[MatTorrance::ROUGHNESS+texOffset]) > 1)
		roughness.y = texValues[MatTorrance::ROUGHNESS+texOffset].y;
	return MatSampleTorrance{
		Spectrum{texValues[MatTorrance::ALBEDO+texOffset]},
		texValues[MatTorrance::ROUGHNESS+texOffset].z,
		roughness, params.ndf
	};
}


// The importance sampling routine
CUDA_FUNCTION math::PathSample sample(const MatSampleTorrance& params,
									  const Direction& incidentTS,
									  Boundary& boundary,
									  const math::RndSet2_1& rndSet,
									  bool) {
	// Importance sampling for the ndf
	math::DirectionSample cavityTS = sample_ndf(params.ndf, params.roughness, rndSet);

	// Find the visible half vector.
	float iDotH;
	Direction halfTS = sample_visible_normal_vcavity(incidentTS, cavityTS.direction, rndSet.i0, iDotH);
	// TODO rotate half vector

	boundary.set_halfTS(halfTS);

	// Reflect the vector 
	Direction excidentTS = (2.0f * iDotH) * halfTS - incidentTS;

	// Get geometry factors for PDF and throughput computation
	float ge = geoshadowing_vcavity(iDotH, excidentTS.z, halfTS.z, params.roughness);
	float gi = geoshadowing_vcavity(iDotH, incidentTS.z, halfTS.z, params.roughness);
	float g = geoshadowing_vcavity_reflection(gi, ge);
	if(ge == 0.0f || gi == 0.0f) // Completely nullyfy the invalid result
		return math::PathSample {};

	// Copy the sign for two sided diffuse
	return math::PathSample {
		Spectrum{params.albedo} * sdiv(g,gi),
		math::PathEventType::REFLECTED,
		excidentTS,
		cavityTS.pdf * sdiv(gi, ei::abs(4.0f * incidentTS.z * halfTS.z)),
		cavityTS.pdf * sdiv(ge, ei::abs(4.0f * excidentTS.z * halfTS.z))
	};
}

// The evaluation routine
CUDA_FUNCTION math::EvalValue evaluate(const MatSampleTorrance& params,
									   const Direction& incidentTS,
									   const Direction& excidentTS,
									   Boundary& boundary) {
	// No transmission
	if(incidentTS.z * excidentTS.z < 0.0f) return math::EvalValue{};

	// Geometry Term
	Direction halfTS = boundary.get_halfTS(incidentTS, excidentTS);
	float iDotH = dot(incidentTS, halfTS);
	float ge = geoshadowing_vcavity(iDotH, excidentTS.z, halfTS.z, params.roughness);
	float gi = geoshadowing_vcavity(iDotH, incidentTS.z, halfTS.z, params.roughness);
	float g = geoshadowing_vcavity_reflection(gi, ge);

	// TODO: rotate halfTS

	// Normal Density Term
	float d = eval_ndf(params.ndf, params.roughness, halfTS);

	// Fresnel is done as layer blending...

	return math::EvalValue {
		params.albedo * sdiv(g * d, ei::abs(4.0f * incidentTS.z * excidentTS.z)),
		ei::abs(excidentTS.z),
		AngularPdf(sdiv(gi * d, ei::abs(4.0f * incidentTS.z))),
		AngularPdf(sdiv(ge * d, ei::abs(4.0f * excidentTS.z)))
	};
}

// The albedo routine
CUDA_FUNCTION Spectrum albedo(const MatSampleTorrance& params) {
	return params.albedo;
}

CUDA_FUNCTION Spectrum emission(const MatSampleTorrance& params, const scene::Direction& geoN, const scene::Direction& excident) {
	return Spectrum{0.0f};
}

template MaterialSampleConcept<MatSampleTorrance>;
template MaterialConcept<MatTorrance>;

}}} // namespace mufflon::scene::materials
