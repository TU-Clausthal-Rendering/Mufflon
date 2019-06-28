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
		params.shadowing,
		params.ndf,
		roughness
	};
}

// The importance sampling routine
CUDA_FUNCTION math::PathSample sample(const MatSampleWalter& params,
									  const Direction& incidentTS,
									  Boundary& boundary,
									  const math::RndSet2_1& rndSet,
									  bool adjoint) {
	u64 rnd = rndSet.i0;
	float iDotH;
	Direction halfTS;
	AngularPdf cavityPdf;
	if(params.shadowing == ShadowingModel::SMITH) {
		// Find the visible half vector.
		halfTS = sample_visible_normal_smith(params.ndf, incidentTS, params.roughness, rndSet, rnd);
		cavityPdf = AngularPdf(eval_ndf(params.ndf, params.roughness, halfTS));
		iDotH = ei::dot(incidentTS, halfTS);
	} else {
		// Importance sampling for the ndf
		math::DirectionSample cavityTS = sample_ndf(params.ndf, params.roughness, rndSet);

		// Find the visible half vector.
		auto h = sample_visible_normal_vcavity(incidentTS, cavityTS.direction, rnd);
		halfTS = h.halfTS;
		iDotH = h.cosI;
		cavityPdf = cavityTS.pdf;
	}
	boundary.set_halfTS(halfTS);

	// Compute the refraction index
	float n_i = boundary.incidentMedium.get_refraction_index().x;
	float n_e = boundary.otherMedium.get_refraction_index().x;
	float eta = n_i / n_e;
	float etaSq = eta * eta;
	float iDotHabs = ei::abs(iDotH);
	Direction excidentTS;
	float eDotH, eDotHabs;
	// Snells law
	float t = 1.0f - etaSq * (1.0f - iDotHabs * iDotHabs);
	if(t <= 0.0f) { // Total internal reflection
		excidentTS = (2.0f * iDotH) * halfTS - incidentTS;
		eDotH = iDotH;
		eDotHabs = iDotHabs;
	} else {
		eDotHabs = sqrt(t);
		eDotH = eDotHabs * -ei::sgn(iDotH); // Opposite to iDotH
		// The refraction vector
		excidentTS = ei::sgn(iDotH) * (eta * iDotHabs - eDotHabs) * halfTS - eta * incidentTS;
	}

	// Get geometry and common factors for PDF and throughput computation
	float ge, gi;
	if(params.shadowing == ShadowingModel::SMITH) {
		ge = geoshadowing_smith(eDotH, excidentTS, params.roughness, params.ndf);
		gi = geoshadowing_smith(iDotH, incidentTS, params.roughness, params.ndf);
	} else {
		ge = geoshadowing_vcavity(eDotH, excidentTS.z, halfTS.z, params.roughness);
		gi = geoshadowing_vcavity(iDotH, incidentTS.z, halfTS.z, params.roughness);
	}

	if(ge == 0.0f || gi == 0.0f) // Completely nullify the invalid result
		return math::PathSample {};

	float throughput;
	AngularPdf pdfForw, pdfBack;
	math::PathEventType eventType;
	if(t <= 0.0f) {
		eventType = math::PathEventType::REFLECTED;
		float g;
		if(params.shadowing == ShadowingModel::SMITH)
			g = geoshadowing_smith_reflection(gi, ge);
		else
			g = geoshadowing_vcavity_reflection(gi, ge);
		throughput = sdiv(g, gi);
		pdfForw = cavityPdf * sdiv(gi, ei::abs(4.0f * incidentTS.z * halfTS.z));
		pdfBack = cavityPdf * sdiv(ge, ei::abs(4.0f * excidentTS.z * halfTS.z));
	} else {
		eventType = math::PathEventType::REFRACTED;
		float g;
		if(params.shadowing == ShadowingModel::SMITH)
			g = geoshadowing_smith_transmission(gi, ge);
		else
			g = geoshadowing_vcavity_transmission(gi, ge);
		throughput = sdiv(g, gi);
		if(adjoint)
			throughput *= etaSq;

		AngularPdf common = cavityPdf * sdiv(iDotHabs * eDotHabs, ei::sq(n_i * iDotH + n_e * eDotH) * halfTS.z);
		pdfForw = common * sdiv(gi * n_e * n_e, ei::abs(incidentTS.z));
		pdfBack = common * sdiv(ge * n_i * n_i, ei::abs(excidentTS.z));
	}

	return math::PathSample {
		Spectrum { throughput }, eventType, excidentTS,
		pdfForw, pdfBack
	};
}

// The evaluation routine
CUDA_FUNCTION math::BidirSampleValue evaluate(const MatSampleWalter& params,
											  const Direction& incidentTS,
											  const Direction& excidentTS,
											  Boundary& boundary) {
	bool isReflection = incidentTS.z * excidentTS.z > 0.0f;

	// General terms. For refraction iDotH != eDotH!
	Direction halfTS = boundary.get_halfTS(incidentTS, excidentTS);
	float iDotH = dot(incidentTS, halfTS);
	float eDotH = dot(excidentTS, halfTS);
	float n_i = boundary.incidentMedium.get_refraction_index().x;
	float n_e = boundary.otherMedium.get_refraction_index().x;

	// Use snells law to check for total internal reflection
	if(isReflection) {
		float t = ei::sq(n_i / n_e) * (1.0f - iDotH * iDotH);
		// For TIR t must be >= 1 such that sqrt(1-t) would fail
		if(t < 1.0f) return {};
	}

	// Geometry Term
	float ge, gi;
	if(params.shadowing == ShadowingModel::SMITH) {
		ge = geoshadowing_smith(eDotH, excidentTS, params.roughness, params.ndf);
		gi = geoshadowing_smith(iDotH, incidentTS, params.roughness, params.ndf);
	} else {
		ge = geoshadowing_vcavity(eDotH, excidentTS.z, halfTS.z, params.roughness);
		gi = geoshadowing_vcavity(iDotH, incidentTS.z, halfTS.z, params.roughness);
	}
	if(ge == 0.0f || gi == 0.0f)
		return math::BidirSampleValue {};

	// Normal Density Term
	float d = eval_ndf(params.ndf, params.roughness, halfTS);

	// Fresnel is done as layer blending...

	if(isReflection) { // Reflections are possible due to total internal reflection.
		float g;
		if(params.shadowing == ShadowingModel::SMITH) 
			g = geoshadowing_smith_reflection(gi, ge);
		else
			g = geoshadowing_vcavity_reflection(gi, ge);
		return math::BidirSampleValue {
			Spectrum{ sdiv(g * d, 4.0f * incidentTS.z * excidentTS.z) },
			AngularPdf{ sdiv(gi * d, 4.0f * ei::abs(incidentTS.z)) },
			AngularPdf{ sdiv(ge * d, 4.0f * ei::abs(excidentTS.z)) }
		};
	}

	float common = sdiv(ei::abs(d * iDotH * eDotH), ei::sq(n_i * iDotH + n_e * eDotH));
	float g;
	if(params.shadowing == ShadowingModel::SMITH)
		g = geoshadowing_smith_transmission(gi, ge);
	else
		g = geoshadowing_vcavity_transmission(gi, ge);
	float bsdf = g * common * sdiv(n_e * n_e, ei::abs(incidentTS.z * excidentTS.z));
	return math::BidirSampleValue {
		Spectrum{bsdf},
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

CUDA_FUNCTION math::SampleValue emission(const MatSampleWalter& params, const scene::Direction& geoN, const scene::Direction& excident) {
	return math::SampleValue{};
}

CUDA_FUNCTION float pdf_max(const MatSampleWalter& params) {
	return 1.0f / (ei::PI * params.roughness.x * params.roughness.y);
}

template class MaterialSampleConcept<MatSampleWalter>;
template class MaterialConcept<MatWalter>;

}}} // namespace mufflon::scene::materials
