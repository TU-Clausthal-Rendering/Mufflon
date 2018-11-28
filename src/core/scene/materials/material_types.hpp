#pragma once

#include "core/scene/types.hpp"

namespace mufflon { namespace scene { namespace materials {

	// Return value of an importance sampler
	struct Sample {
		Spectrum throughput {1.0f};		// BxDF * cosθ / pdfF
		enum class Type: u32 {			// Type of interaction
			INVALID,
			REFLECTED,
			REFRACTED,
		} type = Type::INVALID;
		Direction excident {0.0f};		// The sampled direction
		AngularPdf pdfF {0.0f};			// Sampling PDF in forward direction (current sampler)
		AngularPdf pdfB {0.0f};			// Sampling PDF with reversed incident and excident directions
	};
	
	// Return value of a BxDF evaluation function
	struct EvalValue {
		Spectrum bxdf {0.0f};			// BRDF or BTDF value
		float cosThetaOut {0.0f};		// Outgoing cosine
		AngularPdf pdfF {0.0f};			// Sampling PDF in forward direction 
		AngularPdf pdfB {0.0f};			// Sampling PDF with reversed incident and excident directions
	};

}}} // namespace mufflon::scene::materials