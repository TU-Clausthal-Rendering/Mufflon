#pragma once

#include "core/scene/types.hpp"

namespace mufflon { namespace scene { namespace material {

	/*
	 * A RndSet is a fixed size set of random numbers which may be consumed by a material
	 * sampler. The first two values are standard uniform distributed floating point samples
	 * which should be used to sample the direction.
	 * The third value is open to additional requirements as layer decisions.
	 */
	struct RndSet {
		float u0;	// In [0,1)
		float u1;	// In [0,1)
		u32 i0;		// Full 32 bit random information
	};

	// Return value of an importance sampler
	struct Sample {
		Spectrum throughput {1.0f};		// BxDF * cosθ / pdfF
		Direction excident {0.0f};		// The sampled direction
		float pdfF {0.0f};				// Sampling PDF in forward direction (current sampler)
		float pdfB {0.0f};				// Sampling PDF with reversed incident and excident directions
		enum class Type: u32 {			// Type of interaction
			INVALID,
			REFLECTED,
			REFRACTED,
		} type = Type::INVALID;
	};
	
	// Return value of a BxDF evaluation function
	struct EvalValue {
		Spectrum bxdf {0.0f};			// BRDF or BTDF value
		float cosThetaOut {1.0f};		// Outgoing cosine
		float pdfF {0.0f};				// Sampling PDF in forward direction 
		float pdfB {0.0f};				// Sampling PDF with reversed incident and excident directions
	};

}}} // namespace mufflon::scene::material