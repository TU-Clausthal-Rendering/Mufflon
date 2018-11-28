#pragma once

#include "core/scene/types.hpp"

namespace mufflon { namespace math {

	enum class PathEventType: u32 {
		INVALID,						// Discard this sample
		REFLECTED,						// Reflection or any start vertex emit
		REFRACTED,						// Passed a material boundary
	};

	// Return value of an importance sampler
	struct PathSample {
		Spectrum throughput {0.0f};		// BxDF * cosθ / pdfF
		PathEventType type { PathEventType::INVALID };
		scene::Direction excident {0.0f};	// The sampled direction
		AngularPdf pdfF {0.0f};			// Sampling PDF in forward direction (current sampler)
		AngularPdf pdfB {0.0f};			// Sampling PDF with reversed incident and excident directions
	};
	
	// Return value of a BxDF evaluation function
	struct EvalValue {
		Spectrum value {0.0f};			// Flux, Importance, BRDF or BTDF value
		float cosThetaOut {0.0f};		// Outgoing cosine
		AngularPdf pdfF {0.0f};			// Sampling PDF in forward direction 
		AngularPdf pdfB {0.0f};			// Sampling PDF with reversed incident and excident directions
	};

}} // namespace mufflon::math