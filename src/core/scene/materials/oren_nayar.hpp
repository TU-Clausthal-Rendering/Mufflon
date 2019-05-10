#pragma once

#include "core/export/api.h"
#include "core/math/sampling.hpp"
#include "material_definitions.hpp"
#include "core/scene/textures/texture.hpp"

namespace mufflon { namespace scene { namespace materials {

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

CUDA_FUNCTION math::BidirSampleValue evaluate(const MatSampleOrenNayar& params,
											  const Direction& incidentTS,
											  const Direction& excidentTS ,
											  Boundary& boundary) {
	// No transmission - already checked by material, but in a combined model we might get a call
	if(incidentTS.z * excidentTS.z < 0.0f) return math::BidirSampleValue{};

	// Two sided diffuse (therefore the abs())
	float cosThetaI = ei::abs(incidentTS.z);
	float cosThetaO = ei::abs(excidentTS.z);
	float sinThetaI = sqrt(ei::max(0.0f, (1.0f - cosThetaI) * (1.0f + cosThetaI)));
	float sinThetaO = sqrt(ei::max(0.0f, (1.0f - cosThetaO) * (1.0f + cosThetaO)));
	float cosPhi = 0.0f;
	if(sinThetaI > 1e-4f && sinThetaO > 1e-4f)
	{
		// cos(φ_i-φ_o) = cos(φ_i)cos(φ_o) + sin(φ_i)sin(φ_o)
		float cosPhiI = incidentTS.x / sinThetaI;
		float sinPhiI = incidentTS.y / sinThetaI;
		float cosPhiO = excidentTS.x / sinThetaO;
		float sinPhiO = excidentTS.y / sinThetaO;
		cosPhi = cosPhiI * cosPhiO + sinPhiI * sinPhiO;
		cosPhi = ei::max(0.0f, cosPhi);
	}
	// sin(α) * tan(β) = sin(α) * sin(β) / cos(β)
	float sinTan = sinThetaI * sinThetaO
		/ ei::max(cosThetaI, cosThetaO);
	return math::BidirSampleValue {
		params.albedo * ((params.a + params.b * cosPhi * sinTan) / ei::PI),
		AngularPdf(cosThetaO / ei::PI),
		AngularPdf(cosThetaI / ei::PI)
	};
}

CUDA_FUNCTION math::PathSample sample(const MatSampleOrenNayar& params,
									  const Direction& incidentTS,
									  Boundary& boundary,
									  const math::RndSet2_1& rndSet,
									  bool adjoint) {
	// Importance sampling for lambert: BRDF * cos(theta)
	Direction excidentTS = math::sample_dir_cosine(rndSet.u0, rndSet.u1).direction;
	auto eval = evaluate(params, incidentTS, excidentTS, boundary);
	// Copy the sign for two sided diffuse
	return math::PathSample {
		eval.value * (ei::abs(excidentTS.z) / float(eval.pdf.forw)),
		math::PathEventType::REFLECTED,
		excidentTS * ei::sgn(incidentTS.z),
		eval.pdf
	};
}

CUDA_FUNCTION Spectrum albedo(const MatSampleOrenNayar& params) {
	return params.albedo;
}

CUDA_FUNCTION math::SampleValue emission(const MatSampleOrenNayar& params, const scene::Direction& geoN, const scene::Direction& excident) {
	return math::SampleValue{};
}

CUDA_FUNCTION float pdf_max(const MatSampleOrenNayar& params) {
	return 1.0f / ei::PI;
}

template MaterialSampleConcept<MatSampleOrenNayar>;
template MaterialConcept<MatOrenNayar>;

}}} // namespace mufflon::scene::materials
