#pragma once

#include "material.hpp"
#include "core/export/api.h"
#include "core/math/sampling.hpp"

namespace mufflon { namespace scene { namespace materials {

// Forward declared:
CUDA_FUNCTION int fetch_subparam(Materials type, const char* subDesc, const UvCoordinate& uvCoordinate, char* subParam);
CUDA_FUNCTION math::EvalValue evaluate_subdesc(Materials type, const char* subParams, const Direction& incidentTS, const Direction& excidentTS, Boundary& boundary, bool adjoint, bool merge);
CUDA_FUNCTION math::PathSample sample_subdesc(Materials type, const char* subParams, const Direction& incidentTS, Boundary& boundary, const math::RndSet2_1& rndSet, bool adjoint);
CUDA_FUNCTION Spectrum albedo(Materials type, const char* subParams);
CUDA_FUNCTION Spectrum emission(Materials type, const char* subParams, const scene::Direction& geoN, const scene::Direction& excident);

struct BlendParameterPack {
	float factorA;
	float factorB;
	u32 offsetB;	// LayerA starts at 'this+1', but LayerB is not implicitly known
	Materials typeA;
	Materials typeB;
};

struct BlendDesc {
	float factorA;
	float factorB;
	u32 offsetB;	// LayerA starts at 'this+1', but LayerB is not implicitly known
	Materials typeA;
	Materials typeB;

	CUDA_FUNCTION int fetch(const UvCoordinate& uvCoordinate, char* outBuffer) const {
		// Fetch the layers recursively
#ifdef __CUDA_ARCH__
	// TODO: CUDA doesn't like recursion...
		return 0;
#else // __CUDA_ARCH__
		const char* layerA = as<char>(this + 1);
		int sizeA = fetch_subparam(typeA, layerA, uvCoordinate, outBuffer + sizeof(BlendParameterPack));
		const char* layerB = as<char>(this) + offsetB;
		int sizeB = fetch_subparam(typeB, layerB, uvCoordinate, outBuffer + sizeof(BlendParameterPack) + sizeA);
		*as<BlendParameterPack>(outBuffer) = BlendParameterPack{
			factorA, factorB, u32(sizeof(BlendParameterPack) + sizeA), typeA, typeB
		};
		return sizeof(BlendParameterPack) + sizeA + sizeB;
#endif // __CUDA_ARCH__
	}
};

// Class for the handling of the Lambertian material.
class Blend : public IMaterial {
public:
	Blend(std::unique_ptr<IMaterial> layerA, float factorA, std::unique_ptr<IMaterial> layerB, float factorB) :
		IMaterial{Materials::BLEND},
		m_layerA{move(layerA)},
		m_layerB{move(layerB)},
		m_factorA{factorA},
		m_factorB{factorB}
	{}

	MaterialPropertyFlags get_properties() const noexcept final {
		return m_layerA->get_properties() | m_layerB->get_properties();
	}

	std::size_t get_descriptor_size(Device device) const final {
		return sizeof(BlendDesc)
			+ m_layerA->get_descriptor_size(device)
			+ m_layerB->get_descriptor_size(device)
			- IMaterial::get_descriptor_size(device);	// Counted twice (in layer A and B)
	}

	std::size_t get_parameter_pack_size() const final {
		return sizeof(BlendParameterPack)
			+ m_layerA->get_parameter_pack_size()
			+ m_layerB->get_parameter_pack_size()
			- IMaterial::get_parameter_pack_size();	// Counted twice (in layer A and B)
	}

	char* get_subdescriptor(Device device, char* outBuffer) const final {
		char* layerBegin = outBuffer + sizeof(BlendDesc);
		char* layerAEnd = m_layerA->get_subdescriptor(device, layerBegin);
		char* layerBEnd = m_layerB->get_subdescriptor(device, layerAEnd);
		*as<BlendDesc>(outBuffer) = BlendDesc{
			m_factorA, m_factorB, u32(layerAEnd - outBuffer), m_layerA->get_type(), m_layerB->get_type()
		};
		return layerBEnd;
	}

	Emission get_emission() const final {
		if(m_layerA->get_properties().is_emissive()) {
			Emission em = m_layerA->get_emission();
			em.scale *= m_factorA;
			return em;
		} else {
			Emission em = m_layerB->get_emission();
			em.scale *= m_factorB;
			return em;
		}
	}

	Medium compute_medium() const final {
		// Forward the medium of layerA. There is no usefull blend operation here,
		// since the medium is not part of the surface, but of its surrounding.
		// The two layers should agree in the medium or explicitly define an order.
		return m_layerA->compute_medium();
	}

	void set_factor_a(float fA) {
		m_factorA = fA;
		m_dirty = true;
	}
	void set_factor_b(float fB) {
		m_factorB = fB;
		m_dirty = true;
	}
	float get_factor_a() const noexcept { return m_factorA; }
	float get_factor_b() const noexcept { return m_factorB; }
	const IMaterial* get_layer_a() const noexcept { return m_layerA.get(); }
	const IMaterial* get_layer_b() const noexcept { return m_layerB.get(); }
private:
	std::unique_ptr<IMaterial> m_layerA;
	std::unique_ptr<IMaterial> m_layerB;
	float m_factorA;
	float m_factorB;
};



// The importance sampling routine
CUDA_FUNCTION math::PathSample
blend_sample(const BlendParameterPack& params,
			 const Direction& incidentTS,
			 Boundary& boundary,
			 math::RndSet2_1 rndSet,
			 bool adjoint) {
#ifdef __CUDA_ARCH__
	// TODO: CUDA doesn't like recursion...
	return math::PathSample{};
#else // __CUDA_ARCH__
	const char* layerA = as<char>(&params + 1);
	const char* layerB = as<char>(&params) + params.offsetB;
	// Determine a probability for each layer.
	float fa = params.factorA * sum(albedo(params.typeA, layerA));
	float fb = params.factorB * sum(albedo(params.typeB, layerB));
	float p = fa / (fa + fb);
	u64 probLayerA = math::percentage_of(std::numeric_limits<u64>::max() - 1, p);

	// Sample and get the pdfs of the second layer.
	math::PathSample sample;
	math::EvalValue otherVal;
	float scaleS, scaleE;
	if(rndSet.i0 < probLayerA) {
		rndSet.i0 = math::rescale_sample(rndSet.i0, 0, probLayerA-1);
		sample = sample_subdesc(params.typeA, layerA, incidentTS, boundary, rndSet, adjoint);
		if(sample.pdfF.is_zero()) return sample; // Discard
		otherVal = evaluate_subdesc(params.typeB, layerB, incidentTS, sample.excident, boundary, adjoint, false);
		scaleS = float(sample.pdfF) * params.factorA;
		scaleE = otherVal.cosOut * params.factorB;
	} else {
		rndSet.i0 = math::rescale_sample(rndSet.i0, probLayerA, std::numeric_limits<u64>::max());
		sample = sample_subdesc(params.typeB, layerB, incidentTS, boundary, rndSet, adjoint);
		if(sample.pdfF.is_zero()) return sample; // Discard
		otherVal = evaluate_subdesc(params.typeA, layerA, incidentTS, sample.excident, boundary, adjoint, false);
		scaleS = float(sample.pdfF) * params.factorB;
		scaleE = otherVal.cosOut * params.factorA;
		p = 1.0f - p;
	}

	// Blend values and pdfs.
	float origPdf = float(sample.pdfF);
	sample.pdfF = AngularPdf{ ei::lerp(float(otherVal.pdfF), float(sample.pdfF), p) };
	sample.pdfB = AngularPdf{ ei::lerp(float(otherVal.pdfB), float(sample.pdfB), p) };
	sample.throughput = (sample.throughput * scaleS + otherVal.value * scaleE)
					  / float(sample.pdfF);
	return sample;
#endif // __CUDA_ARCH__
}

// The evaluation routine
CUDA_FUNCTION math::EvalValue
blend_evaluate(const BlendParameterPack& params,
			   const Direction& incidentTS,
			   const Direction& excidentTS,
			   Boundary& boundary,
			   bool adjoint,
			   bool merge) {
#ifdef __CUDA_ARCH__
	// TODO: CUDA doesn't like recursion...
	return math::EvalValue{};
#else // __CUDA_ARCH__
	// Evaluate both sub-layers
	const char* layerA = as<char>(&params + 1);
	const char* layerB = as<char>(&params) + params.offsetB;
	auto valA = evaluate_subdesc(params.typeA, layerA, incidentTS, excidentTS, boundary, adjoint, merge);
	auto valB = evaluate_subdesc(params.typeB, layerB, incidentTS, excidentTS, boundary, adjoint, merge);
	// Determine the probability choice probability to blend the pdfs correctly.
	float fa = params.factorA * sum(albedo(params.typeA, layerA));
	float fb = params.factorB * sum(albedo(params.typeB, layerB));
	float p = fa / (fa + fb);
	// Blend their results
	valA.value = valA.value * params.factorA + valB.value * params.factorB;
	valA.pdfF = AngularPdf{ ei::lerp(float(valB.pdfF), float(valA.pdfF), p) };
	valA.pdfB = AngularPdf{ ei::lerp(float(valB.pdfB), float(valA.pdfB), p) };
	return valA;
#endif // __CUDA_ARCH__
}

// The albedo routine
CUDA_FUNCTION Spectrum
blend_albedo(const BlendParameterPack& params) {
#ifdef __CUDA_ARCH__
	// TODO: CUDA doesn't like recursion...
	return Spectrum{};
#else // __CUDA_ARCH__
	const char* layerA = as<char>(&params + 1);
	const char* layerB = as<char>(&params) + params.offsetB;
	return albedo(params.typeA, layerA) * params.factorA
		 + albedo(params.typeB, layerB) * params.factorB;
#endif // __CUDA_ARCH__
}

CUDA_FUNCTION Spectrum blend_emission(const BlendParameterPack& params, const scene::Direction& geoN, const scene::Direction& excident) {
#ifdef __CUDA_ARCH__
	// TODO: CUDA doesn't like recursion...
	return Spectrum{};
#else // __CUDA_ARCH__
	const char* layerA = as<char>(&params + 1);
	const char* layerB = as<char>(&params) + params.offsetB;
	return emission(params.typeA, layerA, geoN, excident) * params.factorA
		 + emission(params.typeB, layerB, geoN, excident) * params.factorB;
#endif // __CUDA_ARCH__
}

}}} // namespace mufflon::scene::materials
