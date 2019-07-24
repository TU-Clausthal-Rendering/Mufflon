#pragma once

#include "core/export/api.h"
#include "core/math/sampling.hpp"
#include "material_definitions.hpp"
#include "core/scene/textures/texture.hpp"

namespace mufflon { namespace scene { namespace materials {

//#define ONLAMBERT_SAMPLING

CUDA_FUNCTION MatSampleOrenNayar
fetch(const textures::ConstTextureDevHandle_t<CURRENT_DEV>* textures,
	  const ei::Vec4* texValues,
	  int texOffset,
	  const typename MatOrenNayar::NonTexParams& params) {
	return MatSampleOrenNayar{
		Spectrum{texValues[MatOrenNayar::ALBEDO + texOffset]},
		params.a, params.b
	};
}

// sinθ -> cosθ or cosθ -> sinθ
CUDA_FUNCTION __forceinline__ float adjTrig(float x) {
	return sqrt(ei::max(0.0f, (1.0f - x) * (1.0f + x)));
}

CUDA_FUNCTION math::BidirSampleValue evaluate(const MatSampleOrenNayar& params,
											  const Direction& incidentTS,
											  const Direction& excidentTS ,
											  Boundary& boundary) {
	// No transmission - already checked by material, but in a combined model we might get a call
	if(incidentTS.z * excidentTS.z < 0.0f) return math::BidirSampleValue{};

	// Two sided diffuse (therefore the abs())
	float cosThetaI = ei::abs(incidentTS.z);
	float cosThetaO = ei::abs(excidentTS.z);
	float sinThetaI = adjTrig(cosThetaI);
	float sinThetaO = adjTrig(cosThetaO);
	float cosDeltaPhi = 0.0f;
	if(sinThetaI > 1e-4f && sinThetaO > 1e-4f) {
		// cos(φ_i-φ_o) = cos(φ_i)cos(φ_o) + sin(φ_i)sin(φ_o)
		float cosPhiI = incidentTS.x / sinThetaI;
		float sinPhiI = incidentTS.y / sinThetaI;
		float cosPhiO = excidentTS.x / sinThetaO;
		float sinPhiO = excidentTS.y / sinThetaO;
		cosDeltaPhi = cosPhiI * cosPhiO + sinPhiI * sinPhiO;
		cosDeltaPhi = ei::max(0.0f, cosDeltaPhi);
	}
	// sin(α) * tan(β) = sin(α) * sin(β) / cos(β)
	float sinTan = sinThetaI * sinThetaO / ei::max(cosThetaI, cosThetaO);
#ifndef ONLAMBERT_SAMPLING
	float pF = params.a / (params.a + params.b * sinThetaI);
	float pB = params.a / (params.a + params.b * sinThetaO);
#endif // ONLAMBERT_SAMPLING
	return math::BidirSampleValue {
		params.albedo * ((params.a + params.b * cosDeltaPhi * sinTan) / ei::PI),
#ifdef ONLAMBERT_SAMPLING
		AngularPdf{ cosThetaO / ei::PI },
		AngularPdf{ cosThetaI / ei::PI }
#else // ONLAMBERT_SAMPLING
		//AngularPdf{ (pF / ei::PI + (1-pF) * 1.5f * cosDeltaPhi * sinThetaO) * cosThetaO },
		//AngularPdf{ (pB / ei::PI + (1-pB) * 1.5f * cosDeltaPhi * sinThetaI) * cosThetaI }
		AngularPdf{ (pF / ei::PI * cosThetaO + (1-pF) * 0.40596962562901f * cosDeltaPhi * sdiv(powf(acosf(cosThetaO), 1.4f), sinThetaO)) },
		AngularPdf{ (pB / ei::PI * cosThetaI + (1-pB) * 0.40596962562901f * cosDeltaPhi * sdiv(powf(acosf(cosThetaI), 1.4f), sinThetaI)) }
#endif // ONLAMBERT_SAMPLING
	};
}

CUDA_FUNCTION math::PathSample sample(const MatSampleOrenNayar& params,
									  const Direction& incidentTS,
									  Boundary& boundary,
									  const math::RndSet2_1& rndSet,
									  bool adjoint) {
#ifdef ONLAMBERT_SAMPLING
	// Importance sampling for lambert: BRDF * cos(theta)
	Direction excidentTS = math::sample_dir_cosine(rndSet.u0, rndSet.u1).direction;
	auto eval = evaluate(params, incidentTS, excidentTS, boundary);
	// Copy the sign for two sided diffuse
	return math::PathSample {
		eval.value * (excidentTS.z / float(eval.pdf.forw)),
		math::PathEventType::REFLECTED,
		excidentTS * ei::sgn(incidentTS.z),
		eval.pdf
	};

#else // ONLAMBERT_SAMPLING

	// New sampling method from Johannes' PHD:
	// Get all the incident sizes which we also need for the evaluation.
	const float cosThetaI = ei::abs(incidentTS.z);
	const float sinThetaI = adjTrig(cosThetaI);
	float sinPhiI = 0.0f, cosPhiI = 1.0f;
	if(sinThetaI > 1e-4f) {
		cosPhiI = incidentTS.x / sinThetaI;
		sinPhiI = incidentTS.y / sinThetaI;
	}
	float cosDeltaPhi, sinThetaO, cosThetaO, sinPhiO, cosPhiO;
	// Decide between Lambert and second-term sampling
	const float pF = params.a / (params.a + params.b * sinThetaI);
	const float rnd = rndSet.i0 / 1.844674407e19f;
	if(rnd < pF) {
		// Sample using cosine (Lambert term)
		cosThetaO = sqrt(rndSet.u0);			// cos(acos(sqrt(x))) = sqrt(x)
		sinThetaO = sqrt(1.0f - rndSet.u0);		// sqrt(1-cos(theta)^2)
		const float phi = rndSet.u1 * 2 * ei::PI;
		sinPhiO = sin(phi);
		cosPhiO = cos(phi);
		// Compute missing excident terms for evaluation
		// cos(φ_i-φ_o) = cos(φ_i)cos(φ_o) + sin(φ_i)sin(φ_o)
		cosDeltaPhi = cosPhiI * cosPhiO + sinPhiI * sinPhiO;
		cosDeltaPhi = ei::max(0.0f, cosDeltaPhi);
	} else {
		// Sample excident quantities
		const float thetaO = ei::PI / 2.0f * powf(rndSet.u0, 1.0f / 2.4f);
		sinThetaO = sin(thetaO);
		cosThetaO = cos(thetaO);
		//sinThetaO = powf(rndSet.u0, 1.0f / 3.0f);
		const float sinDeltaPhi = 2.0f * rndSet.u1 - 1.0f;
		cosDeltaPhi = adjTrig(sinDeltaPhi);
		// Transform local Δφ into tangent space cosφ/sinφ.
		cosPhiO = cosPhiI * cosDeltaPhi - sinPhiI * sinDeltaPhi;
		sinPhiO = sinPhiI * cosDeltaPhi + cosPhiI * sinDeltaPhi;
	}
	Direction excidentTS { sinThetaO * cosPhiO, sinThetaO * sinPhiO, cosThetaO };
	// BRDF value
	const float sinTan = sinThetaI * sinThetaO / ei::max(cosThetaI, cosThetaO);
	const float brdf = (params.a + params.b * cosDeltaPhi * sinTan) / ei::PI;
	// PDFs
	const float pB = params.a / (params.a + params.b * sinThetaO);
	//const float pdfF = (pF / ei::PI + (1-pF) * 1.5f * cosDeltaPhi * sinThetaO) * cosThetaO;
	//const float pdfB = (pB / ei::PI + (1-pB) * 1.5f * cosDeltaPhi * sinThetaI) * cosThetaI;
	const float pdfF = (pF / ei::PI * cosThetaO + (1-pF) * 0.40596962562901f * cosDeltaPhi * powf(acosf(cosThetaO), 1.4f) / sinThetaO);
	const float pdfB = (pB / ei::PI * cosThetaI + (1-pB) * 0.40596962562901f * cosDeltaPhi * powf(acosf(cosThetaI), 1.4f) / sinThetaI);
	return math::PathSample {
		params.albedo * (brdf * cosThetaO / pdfF),
		math::PathEventType::REFLECTED,
		excidentTS * ei::sgn(incidentTS.z),
		{AngularPdf{pdfF}, AngularPdf{pdfB}}
	};

#endif ONLAMBERT_SAMPLING
}

CUDA_FUNCTION Spectrum albedo(const MatSampleOrenNayar& params) {
	return params.albedo;
}

CUDA_FUNCTION math::SampleValue emission(const MatSampleOrenNayar& params, const scene::Direction& geoN, const scene::Direction& excident) {
	return math::SampleValue{};
}

CUDA_FUNCTION float pdf_max(const MatSampleOrenNayar& params) {
	// TODO: proper maximum
	return 1.0f / ei::PI;
}

template MaterialSampleConcept<MatSampleOrenNayar>;
template MaterialConcept<MatOrenNayar>;

#undef ONLAMBERT_SAMPLING

}}} // namespace mufflon::scene::materials
