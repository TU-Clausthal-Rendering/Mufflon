#pragma once

#include "core/export/api.h"
#include "core/math/sampling.hpp"
#include "material_definitions.hpp"

namespace mufflon { namespace scene { namespace materials {

template<class LayerA, class LayerB>
CUDA_FUNCTION typename MatBlend<LayerA, LayerB>::SampleType
fetch(const textures::ConstTextureDevHandle_t<CURRENT_DEV>* textures,
	  const ei::Vec4* texValues,
	  int texOffset,
	  const MatNTPBlend<LayerA, LayerB>& params) {
	return typename MatBlend<LayerA, LayerB>::SampleType{
		fetch(textures, texValues, texOffset, params.a),
		fetch(textures, texValues, texOffset+LayerA::TEX_COUNT, params.b),
		params.factorA,
		params.factorB
	};
}

// The importance sampling routine
template<class LayerASample, class LayerBSample>
CUDA_FUNCTION math::PathSample sample(const MatSampleBlend<LayerASample, LayerBSample>& params,
									  const Direction& incidentTS,
									  Boundary& boundary,
									  math::RndSet2_1 rndSet,
									  bool adjoint) {
	// Determine a probability for each layer.
	float fa = params.factorA * sum(albedo(params.a));
	float fb = params.factorB * sum(albedo(params.b));
	float p = fa / (fa + fb);
	u64 probLayerA = math::percentage_of(std::numeric_limits<u64>::max() - 1, p);

	// Sample and get the pdfs of the second layer.
	math::PathSample sampleVal;
	math::BidirSampleValue otherVal;
	float scaleS, scaleE;
	if(rndSet.i0 < probLayerA) {
		rndSet.i0 = math::rescale_sample(rndSet.i0, 0, probLayerA-1);
		sampleVal = sample(params.a, incidentTS, boundary, rndSet, adjoint);
		if(sampleVal.pdf.forw.is_zero()) return sampleVal; // Discard
		otherVal = evaluate(params.b, incidentTS, sampleVal.excident, boundary);
		scaleS = float(sampleVal.pdf.forw) * params.factorA;
		scaleE = ei::abs(sampleVal.excident.z) * params.factorB;
	} else {
		rndSet.i0 = math::rescale_sample(rndSet.i0, probLayerA, std::numeric_limits<u64>::max());
		sampleVal = sample(params.b, incidentTS, boundary, rndSet, adjoint);
		if(sampleVal.pdf.forw.is_zero()) return sampleVal; // Discard
		otherVal = evaluate(params.a, incidentTS, sampleVal.excident, boundary);
		scaleS = float(sampleVal.pdf.forw) * params.factorB;
		scaleE = ei::abs(sampleVal.excident.z) * params.factorA;
		p = 1.0f - p;
	}

	// Blend values and pdfs.
	float origPdf = float(sampleVal.pdf.forw);
	sampleVal.pdf.forw = AngularPdf{ ei::lerp(float(otherVal.pdf.forw), float(sampleVal.pdf.forw), p) };
	sampleVal.pdf.back = AngularPdf{ ei::lerp(float(otherVal.pdf.back), float(sampleVal.pdf.back), p) };
	sampleVal.throughput = (sampleVal.throughput * scaleS + otherVal.value * scaleE)
						  / float(sampleVal.pdf.forw);
	return sampleVal;
}

// The evaluation routine
template<class LayerASample, class LayerBSample>
CUDA_FUNCTION math::BidirSampleValue evaluate(const MatSampleBlend<LayerASample, LayerBSample>& params,
											  const Direction& incidentTS,
											  const Direction& excidentTS,
											  Boundary& boundary) {
	// Evaluate both sub-layers
	auto valA = evaluate(params.a, incidentTS, excidentTS, boundary);
	auto valB = evaluate(params.b, incidentTS, excidentTS, boundary);
	// Determine the probability choice probability to blend the pdfs correctly.
	float fa = params.factorA * sum(albedo(params.a));
	float fb = params.factorB * sum(albedo(params.b));
	float p = fa / (fa + fb);// TODO: precompute in fetch?
	// Blend their results
	valA.value = valA.value * params.factorA + valB.value * params.factorB;
	valA.pdf.forw = AngularPdf{ ei::lerp(float(valB.pdf.forw), float(valA.pdf.forw), p) };
	valA.pdf.back = AngularPdf{ ei::lerp(float(valB.pdf.back), float(valA.pdf.back), p) };
	return valA;
}

// The albedo routine
template<class LayerASample, class LayerBSample>
CUDA_FUNCTION Spectrum albedo(const MatSampleBlend<LayerASample, LayerBSample>& params) {
	return albedo(params.a) * params.factorA
		 + albedo(params.b) * params.factorB;
}

template<class LayerASample, class LayerBSample>
CUDA_FUNCTION math::SampleValue emission(const MatSampleBlend<LayerASample, LayerBSample>& params, const scene::Direction& geoN, const scene::Direction& excident) {
	// Evaluate both sub-layers
	auto valA = emission(params.a, geoN, excident);
	auto valB = emission(params.b, geoN, excident);
	// Assume that at most one layer is emissive
	mAssert(valA.pdf.is_zero() || valB.pdf.is_zero());
	// TODO: blending and (blended) sampling of emissive models (independent of the other reflection properties)
	valA.value = valA.value * params.factorA + valB.value * params.factorB;
	valA.pdf += valB.pdf;
	return valA;
}

template<class LayerASample, class LayerBSample>
CUDA_FUNCTION float pdf_max(const MatSampleBlend<LayerASample, LayerBSample>& params) {
	// TODO: p based blending as above?
	return ei::max(pdf_max(params.a), pdf_max(params.b));
}

template class MaterialSampleConcept<MatSampleBlend<MatSampleLambert, MatSampleLambert>>;
template class MaterialConcept<MatBlend<MatLambert, MatLambert>>;

}}} // namespace mufflon::scene::materials
