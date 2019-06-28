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
		params.shadowing,
		params.ndf,
		roughness
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
	float iDotH;
	Direction halfTS;
	AngularPdf cavityPdf;
	u64 rnd = rndSet.i0 & 0xffffffff00000000;
	if(params.shadowing == ShadowingModel::SMITH) {
		halfTS = sample_visible_normal_smith(params.ndf, incidentTS, params.roughness, rndSet, rnd);
		cavityPdf = AngularPdf(eval_ndf(params.ndf, params.roughness, halfTS));
		cavityPdf *= halfTS.z;
		iDotH = dot(incidentTS, halfTS);
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

	if(halfTS.z <= 0)
		debugBreak;
	// Compute Fresnel term and refracted cosine
	float n_i = boundary.incidentMedium.get_refraction_index().x;
	float n_t = boundary.otherMedium.get_refraction_index().x;
	float eta = n_i / n_t;
	float iDotHabs = ei::abs(iDotH);
	Refraction f = fresnel_dielectric(n_i, n_t, iDotHabs);
	if(f.f <= 0.01f)
		debugBreak;
	//f.f = 1.0f;
	// Randomly choose between refraction and reflection proportional to f.f.
	// TIR is handled automatically through f.f = 1 -> independent of random number.
	float rndf = float(rndSet.i0 & 0x00000000ffffffff) / float(std::numeric_limits<u32>::max()-1);

	//bool reflect = rnd < math::percentage_of(std::numeric_limits<u64>::max(), f.f);
	bool reflect = rndf < f.f;
	float eDotH, eDotHabs;
	Direction excidentTS;
	if(reflect) {
		excidentTS = (2.0f * iDotH) * halfTS - incidentTS;
		if(incidentTS.z * excidentTS.z <= 0.0f)
			return math::PathSample{};
		eDotH = iDotH;
		eDotHabs = iDotHabs;
	} else {
		eDotHabs = f.cosTAbs;
		eDotH = eDotHabs * -ei::sgn(iDotH); // Opposite to iDotH
		// The refraction vector
		excidentTS = ei::sgn(iDotH) * (eta * iDotHabs - eDotHabs) * halfTS - eta * incidentTS;
		if(incidentTS.z * excidentTS.z >= 0.0f)
			return math::PathSample{};

		Direction htest = boundary.get_halfTS(incidentTS, excidentTS);
		(void)htest;
		mAssert(ei::approx(htest, halfTS));
	}
	mAssert(ei::approx(dot(excidentTS, halfTS), eDotH));
	mAssert(iDotH * incidentTS.z > 0.0f);

	// Get geometry and common factors for PDF and throughput computation
	float ge, gi;
	if(params.shadowing == ShadowingModel::SMITH) {
		ge = geoshadowing_smith(excidentTS, params.roughness, params.ndf);
		gi = geoshadowing_smith(incidentTS, params.roughness, params.ndf);
	} else {
		ge = geoshadowing_vcavity(eDotH, excidentTS.z, halfTS.z, params.roughness);
		gi = geoshadowing_vcavity(iDotH, incidentTS.z, halfTS.z, params.roughness);
	}
	if(ge == 0.0f || gi == 0.0f)
		return math::PathSample {};

	float throughput;
	AngularPdf pdfForw, pdfBack;
	if(reflect) {
		float g;
		if(params.shadowing == ShadowingModel::SMITH)
			g = geoshadowing_smith_reflection(gi, ge);
		else
			g = geoshadowing_vcavity_reflection(gi, ge);
		throughput = sdiv(g, gi);
		AngularPdf common = cavityPdf * sdiv(f.f, 4.0f * halfTS.z);
		pdfForw = common * sdiv(gi, ei::abs(incidentTS.z));
		pdfBack = common * sdiv(ge, ei::abs(excidentTS.z));
	} else {
		float g;
		if(params.shadowing == ShadowingModel::SMITH)
			g = geoshadowing_smith_transmission(gi, ge);
		else
			g = geoshadowing_vcavity_transmission(gi, ge);
		throughput = sdiv(g, gi);
		if(adjoint)
			throughput *= eta * eta;

		AngularPdf common = cavityPdf * (1.0f - f.f) * sdiv(iDotHabs * eDotHabs, ei::sq(n_i * iDotH + n_t * eDotH) * halfTS.z);
		pdfForw = common * sdiv(gi * n_t * n_t, ei::abs(incidentTS.z));
		pdfBack = common * sdiv(ge * n_i * n_i, ei::abs(excidentTS.z));
	}

	auto debugEval = evaluate(params, incidentTS, excidentTS, boundary);

	if(!ei::approx((float)debugEval.pdf.back, (float)pdfBack, 1e-3f))
		debugBreak;
	if(!ei::approx((float)debugEval.pdf.forw, (float)pdfForw, 1e-3f))
		debugBreak;
	Spectrum brdf{throughput * (float)pdfForw/abs(excidentTS.z)};
	if(!ei::approx(debugEval.value, brdf, 1e-3f))
		debugBreak;

	//if(!reflect) return  math::PathSample{};

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
	float ge, gi;
	if(params.shadowing == ShadowingModel::SMITH) {
		ge = geoshadowing_smith(excidentTS, params.roughness, params.ndf);
		gi = geoshadowing_smith(incidentTS, params.roughness, params.ndf);
	}
	else {
		ge = geoshadowing_vcavity(eDotH, excidentTS.z, halfTS.z, params.roughness);
		gi = geoshadowing_vcavity(iDotH, incidentTS.z, halfTS.z, params.roughness);
	}
	if(ge == 0.0f || gi == 0.0f)
		return math::BidirSampleValue {};

	// Normal Density Term
	float d = eval_ndf(params.ndf, params.roughness, halfTS);

	// Fresnel (only dielectric allowed)
	float f = fresnel_dielectric(n_i, n_t, ei::abs(iDotH)).f;
	//f = 1.f;
	if(isReflection) {
		float g;
		if(params.shadowing == ShadowingModel::SMITH)
			g = geoshadowing_smith_reflection(gi, ge);
		else
			g = geoshadowing_vcavity_reflection(gi, ge);
		float common = d * f / 4.0f;
		return math::BidirSampleValue {
			Spectrum{ sdiv(g * common, incidentTS.z * excidentTS.z) },
			AngularPdf{ sdiv(gi * common, ei::abs(incidentTS.z)) },
			AngularPdf{ sdiv(ge * common, ei::abs(excidentTS.z)) }
		};
	}

	float common = sdiv((1-f) * ei::abs(d * iDotH * eDotH), ei::sq(n_i * iDotH + n_t * eDotH));
	float g;
	if(params.shadowing == ShadowingModel::SMITH)
		g = geoshadowing_smith_transmission(gi, ge);
	else
		g = geoshadowing_vcavity_transmission(gi, ge);
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

template class MaterialSampleConcept<MatSampleMicrofacet>;
template class MaterialConcept<MatMicrofacet>;

}}} // namespace mufflon::scene::materials
