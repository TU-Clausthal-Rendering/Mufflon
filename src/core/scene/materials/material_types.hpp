#pragma once

#include "core/scene/types.hpp"

namespace mufflon { namespace scene { namespace materials {

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

		RndSet(ei::Vec2 u01, u32 i0) :
			u0(u01.x), u1(u01.y),
			i0(i0) {}
	};

	// Return value of an importance sampler
	struct Sample {
		Spectrum throughput {1.0f};		// BxDF * cosθ / pdfF
		AngularPdf pdfF {0.0f};			// Sampling PDF in forward direction (current sampler)
		Direction excident {0.0f};		// The sampled direction
		AngularPdf pdfB {0.0f};			// Sampling PDF with reversed incident and excident directions
		enum class Type: u32 {			// Type of interaction
			INVALID,
			REFLECTED,
			REFRACTED,
		} type = Type::INVALID;
	};
	
	// Return value of a BxDF evaluation function
	struct EvalValue {
		Spectrum bxdf {0.0f};			// BRDF or BTDF value
		float cosThetaOut {0.0f};		// Outgoing cosine
		AngularPdf pdfF {0.0f};			// Sampling PDF in forward direction 
		AngularPdf pdfB {0.0f};			// Sampling PDF with reversed incident and excident directions
	};

}}} // namespace mufflon::scene::materials