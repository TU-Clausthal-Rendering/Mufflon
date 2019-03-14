#pragma once

#include "core/export/api.h"
#include "core/math/sampling.hpp"
#include "core/scene/textures/interface.hpp"
#include "material_definitions.hpp"
#include "microfacet_base.hpp"

namespace mufflon { namespace scene { namespace materials {

CUDA_FUNCTION MatSampleWalter fetch(const textures::ConstTextureDevHandle_t<CURRENT_DEV>* textures,
									const ei::Vec4* texValues,
									int texOffset,
									const typename MatWalter::NonTexParams& params) {
	ei::Vec2 roughness { texValues[MatWalter::ROUGHNESS+texOffset].x };
	if(get_texture_channel_count(textures[MatWalter::ROUGHNESS+texOffset]) > 1)
		roughness.y = texValues[MatWalter::ROUGHNESS+texOffset].y;
	return MatSampleWalter{
		params.absorption,
		texValues[MatWalter::ROUGHNESS+texOffset].z,
		roughness, params.ndf
	};
}

// The importance sampling routine
CUDA_FUNCTION math::PathSample sample(const MatSampleWalter& params,
									  const Direction& incidentTS,
									  Boundary& boundary,
									  const math::RndSet2_1& rndSet,
									  bool adjoint) {
	// Importance sampling for the ndf
	math::DirectionSample cavityTS = sample_ndf(params.ndf, params.roughness, rndSet);

	// Find the visible half vector.
	float iDotH;
	Direction halfTS = sample_visible_normal_vcavity(incidentTS, cavityTS.direction, rndSet.i0, iDotH);
	// TODO rotate half vector

	boundary.set_halfTS(halfTS);

	// Compute the refraction index
	float n_i = boundary.incidentMedium.get_refraction_index().x;
	float n_e = boundary.otherMedium.get_refraction_index().x;
	float eta = n_i / n_e;
	float etaSq = eta * eta;
	float iDotHabs = ei::abs(iDotH);
	// Snells law
	float eDotHabs = sqrt(ei::max(0.0f, 1.0f - etaSq * (1.0f - iDotHabs * iDotHabs)));
	float eDotH = eDotHabs * -ei::sgn(iDotH); // Opposite to iDotH
	// The refraction vector
	Direction excidentTS = ei::sgn(iDotH) * (eta * iDotHabs - eDotHabs) * halfTS - eta * incidentTS;

	// Get geometry and common factors for PDF and throughput computation
	float ge = geoshadowing_vcavity(eDotH, excidentTS.z, halfTS.z, params.roughness);
	float gi = geoshadowing_vcavity(iDotH, incidentTS.z, halfTS.z, params.roughness);
	float g = geoshadowing_vcavity_transmission(gi, ge);
	if(ge == 0.0f || gi == 0.0f) // Completely nullyfy the invalid result
		return math::PathSample {};

	AngularPdf common = cavityTS.pdf * sdiv(iDotHabs * eDotHabs, ei::sq(n_i * iDotH + n_e * eDotH) * halfTS.z);
	Spectrum throughput { sdiv(g, gi) };
	if(adjoint)
		throughput *= etaSq;

	return math::PathSample {
		throughput,
		math::PathEventType::REFRACTED,
		excidentTS,
		common * sdiv(gi * n_e * n_e, ei::abs(incidentTS.z)),
		common * sdiv(ge * n_i * n_i, ei::abs(excidentTS.z)),
	};
}

// The evaluation routine
CUDA_FUNCTION math::EvalValue evaluate(const MatSampleWalter& params,
									   const Direction& incidentTS,
									   const Direction& excidentTS,
									   Boundary& boundary) {
	// No reflection
	if(incidentTS.z * excidentTS.z > 0.0f) return math::EvalValue{};

	// General terms. For refraction iDotH != eDotH!
	Direction halfTS = boundary.get_halfTS(incidentTS, excidentTS);
	float iDotH = dot(incidentTS, halfTS);
	float eDotH = dot(excidentTS, halfTS);
	float n_i = boundary.incidentMedium.get_refraction_index().x;
	float n_e = boundary.otherMedium.get_refraction_index().x;

	// Geometry Term
	float ge = geoshadowing_vcavity(eDotH, excidentTS.z, halfTS.z, params.roughness);
	float gi = geoshadowing_vcavity(iDotH, incidentTS.z, halfTS.z, params.roughness);
	float g = geoshadowing_vcavity_transmission(gi, ge);

	// Normal Density Term
	float d = eval_ndf(params.ndf, params.roughness, halfTS);

	// Fresnel is done as layer blending...

	float common = sdiv(ei::abs(d * iDotH * eDotH), ei::sq(n_i * iDotH + n_e * eDotH));
	float bsdf = g * common * sdiv(n_e * n_e, ei::abs(incidentTS.z * excidentTS.z));
	return math::EvalValue {
		Spectrum{bsdf},
		ei::abs(excidentTS.z),
		AngularPdf(gi * common * sdiv(n_e * n_e, ei::abs(incidentTS.z))),
		AngularPdf(ge * common * sdiv(n_i * n_i, ei::abs(excidentTS.z)))
	};
}

// The albedo routine
CUDA_FUNCTION Spectrum albedo(const MatSampleWalter& params) {
	// Compute a pseudo value based on the absorption.
	// The problem: the true amount of transmittance depends on the depth of the medium.
	return 1.0f / (Spectrum{1.0f} + params.absorption);
}

CUDA_FUNCTION math::EvalValue emission(const MatSampleWalter& params, const scene::Direction& geoN, const scene::Direction& excident) {
	return math::EvalValue{};
}

template MaterialSampleConcept<MatSampleWalter>;
template MaterialConcept<MatWalter>;

}}} // namespace mufflon::scene::materials
