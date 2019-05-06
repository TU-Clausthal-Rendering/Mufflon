#pragma once

#include "core/export/api.h"
#include "core/math/sampling.hpp"
#include "core/scene/textures/interface.hpp"
#include "material_definitions.hpp"
#include "microfacet_base.hpp"
#include "blend_fresnel.hpp"

namespace mufflon { namespace scene { namespace materials {

CUDA_FUNCTION MatSampleMicrofacet fetch(const textures::ConstTextureDevHandle_t<CURRENT_DEV>* textures,
									const ei::Vec4* texValues,
									int texOffset,
									const typename MatMicrofacet::NonTexParams& params) {
	ei::Vec2 roughness { texValues[MatMicrofacet::ROUGHNESS+texOffset].x };
	if(get_texture_channel_count(textures[MatMicrofacet::ROUGHNESS+texOffset]) > 1)
		roughness.y = texValues[MatMicrofacet::ROUGHNESS+texOffset].y;
	return MatSampleMicrofacet{
		params.absorption,
		texValues[MatMicrofacet::ROUGHNESS+texOffset].z,
		roughness, params.ndf
	};
}

CUDA_FUNCTION math::BidirSampleValue evaluate(const MatSampleMicrofacet& params,
											  const Direction& incidentTS,
											  const Direction& excidentTS,
											  Boundary& boundary);

// The importance sampling routine
CUDA_FUNCTION math::PathSample sample(const MatSampleMicrofacet& params,
									  const Direction& incidentTS,
									  Boundary& boundary,
									  const math::RndSet2_1& rndSet,
									  bool adjoint) {
	// Importance sampling for the ndf
	math::DirectionSample cavityTS = sample_ndf(params.ndf, params.roughness, rndSet);

	// Find the visible half vector.
	//math::RndSet2 r2 { rndSet.i0 };
	u64 rnd = rndSet.i0;
	auto h = sample_visible_normal_vcavity(incidentTS, cavityTS.direction, rnd);
	// TODO rotate half vector
	boundary.set_halfTS(h.halfTS);

	// Compute Fresnel term and refracted cosine
	float n_i = boundary.incidentMedium.get_refraction_index().x;
	float n_t = boundary.otherMedium.get_refraction_index().x;
	float eta = n_i / n_t;
	float iDotHabs = ei::abs(h.cosI);
	Refraction f = fresnel_dielectric(n_i, n_t, iDotHabs);

	// Randomly choose between refraction and reflection proportional to f.f.
	// TIR is handled automatically through f.f = 1 -> independent of random number.
	bool reflect = rnd < math::percentage_of(std::numeric_limits<u64>::max(), f.f);
	float eDotH, eDotHabs;
	Direction excidentTS;
	if(reflect) {
		excidentTS = (2.0f * h.cosI) * h.halfTS - incidentTS;
		eDotH = h.cosI;
		eDotHabs = iDotHabs;
	} else {
		eDotHabs = f.cosTAbs;
		eDotH = eDotHabs * -ei::sgn(h.cosI); // Opposite to iDotH
		// The refraction vector
		excidentTS = ei::sgn(h.cosI) * (eta * iDotHabs - eDotHabs) * h.halfTS - eta * incidentTS;

		Direction htest = boundary.get_halfTS(incidentTS, excidentTS);
		mAssert(ei::approx(htest, h.halfTS));
	}
	mAssert(ei::approx(dot(excidentTS, h.halfTS), eDotH));
	mAssert(h.cosI * incidentTS.z > 0.0f);

	// Get geometry and common factors for PDF and throughput computation
	float ge = geoshadowing_vcavity(eDotH, excidentTS.z, h.halfTS.z, params.roughness);
	float gi = geoshadowing_vcavity(h.cosI, incidentTS.z, h.halfTS.z, params.roughness);
	if(ge == 0.0f || gi == 0.0f) {// Completely nullyfy the invalid result
		auto test = evaluate(params, incidentTS, excidentTS, boundary);
		//if(reflect) mAssert(test.value == 0.0f);
		//if(!reflect) mAssert(test.value == 0.0f);
		return math::PathSample {};
	}

	float throughput;
	AngularPdf pdfForw, pdfBack;
	if(reflect) {
		float g = geoshadowing_vcavity_reflection(gi, ge);
		throughput = sdiv(g, gi);
		AngularPdf common = cavityTS.pdf * sdiv(f.f, 4.0f * h.halfTS.z);
		pdfForw = common * sdiv(gi, ei::abs(incidentTS.z));
		pdfBack = common * sdiv(ge, ei::abs(excidentTS.z));
	} else {
		float g = geoshadowing_vcavity_transmission(gi, ge);
		throughput = sdiv(g, gi);
		if(adjoint)
			throughput *= eta * eta;

		AngularPdf common = cavityTS.pdf * (1.0f - f.f) * sdiv(iDotHabs * eDotHabs, ei::sq(n_i * h.cosI + n_t * eDotH) * h.halfTS.z);
		pdfForw = common * sdiv(gi * n_t * n_t, ei::abs(incidentTS.z));
		pdfBack = common * sdiv(ge * n_i * n_i, ei::abs(excidentTS.z));
	}
	auto test = evaluate(params, incidentTS, excidentTS, boundary);
	float t2 = test.value.x * ei::abs(excidentTS.z) / float(test.pdf.forw);
	mAssert(ei::approx(float(test.pdf.forw), float(pdfForw), 1e-4f));
	mAssert(ei::approx(float(test.pdf.back), float(pdfBack), 1e-4f));
	mAssert(ei::approx(t2, throughput, 1e-4f));

	return math::PathSample {
		Spectrum { throughput },
		reflect ? math::PathEventType::REFLECTED : math::PathEventType::REFRACTED,
		excidentTS, pdfForw, pdfBack
	};
}

// The evaluation routine
CUDA_FUNCTION math::BidirSampleValue evaluate(const MatSampleMicrofacet& params,
											  const Direction& incidentTS,
											  const Direction& excidentTS,
											  Boundary& boundary) {
	bool isReflection = incidentTS.z * excidentTS.z > 0.0f;

	// General terms. For refraction iDotH != eDotH!
	Direction halfTS = boundary.get_halfTS(incidentTS, excidentTS);
	float iDotH = dot(incidentTS, halfTS);
	float eDotH = dot(excidentTS, halfTS);
	float n_i = boundary.incidentMedium.get_refraction_index().x;
	float n_t = boundary.otherMedium.get_refraction_index().x;

	// Geometry Term
	float ge = geoshadowing_vcavity(eDotH, excidentTS.z, halfTS.z, params.roughness);
	float gi = geoshadowing_vcavity(iDotH, incidentTS.z, halfTS.z, params.roughness);
	if(ge == 0.0f || gi == 0.0f)
		return math::BidirSampleValue {};

	// Normal Density Term
	float d = eval_ndf(params.ndf, params.roughness, halfTS);

	// Fresnel (only dielectric allowed)
	float f = fresnel_dielectric(n_i, n_t, ei::abs(iDotH)).f;

	if(isReflection) {
		float g = geoshadowing_vcavity_reflection(gi, ge);
		float common = d * f / 4.0f;
		return math::BidirSampleValue {
			Spectrum{ sdiv(g * common, incidentTS.z * excidentTS.z) },
			AngularPdf{ sdiv(gi * common, ei::abs(incidentTS.z)) },
			AngularPdf{ sdiv(ge * common, ei::abs(excidentTS.z)) }
		};
	}

	float common = sdiv((1-f) * ei::abs(d * iDotH * eDotH), ei::sq(n_i * iDotH + n_t * eDotH));
	float g = geoshadowing_vcavity_transmission(gi, ge);
	float bsdf = g * common * sdiv(n_t * n_t, ei::abs(incidentTS.z * excidentTS.z));
	return math::BidirSampleValue {
		Spectrum{bsdf},
		AngularPdf(gi * common * sdiv(n_t * n_t, ei::abs(incidentTS.z))),
		AngularPdf(ge * common * sdiv(n_i * n_i, ei::abs(excidentTS.z)))
	};
}

// The albedo routine
CUDA_FUNCTION Spectrum albedo(const MatSampleMicrofacet& params) {
	// Compute a pseudo value based on the absorption.
	// The problem: the true amount of transmittance depends on the depth of the medium.
	return 1.0f / (Spectrum{1.0f} + params.absorption);
}

CUDA_FUNCTION math::SampleValue emission(const MatSampleMicrofacet& params, const scene::Direction& geoN, const scene::Direction& excident) {
	return math::SampleValue{};
}

CUDA_FUNCTION float pdf_max(const MatSampleMicrofacet& params) {
	return 1.0f / (ei::PI * params.roughness.x * params.roughness.y);
}

template MaterialSampleConcept<MatSampleMicrofacet>;
template MaterialConcept<MatMicrofacet>;

}}} // namespace mufflon::scene::materials
