﻿#pragma once

#include "core/scene/types.hpp"

namespace mufflon { namespace math {

struct DirectionSample {
	scene::Direction direction;
	AngularPdf pdf;
};

struct PositionSample {
	scene::Point position;
	AreaPdf pdf;
};


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


// Return value of a sampler
struct SampleValue {
	Spectrum value {0.0f};			// Flux, Importance, BRDF or BTDF value
	AngularPdf pdf {0.0f};			// Sampling PDF in forward direction 
};
struct BidirSampleValue {
	Spectrum value {0.0f};			// Flux, Importance, BRDF or BTDF value
	AngularPdf pdfF {0.0f};			// Sampling PDF in forward direction 
	AngularPdf pdfB {0.0f};			// Sampling PDF with reversed incident and excident directions
};

// Return value of a BxDF evaluation function
struct EvalValue {
	Spectrum value {0.0f};			// Flux, Importance, BRDF or BTDF value
	float cosOut {0.0f};			// Outgoing cosine (if a surface, 0 or special value otherwise)
	AngularPdf pdfF {0.0f};			// Sampling PDF in forward direction 
	AngularPdf pdfB {0.0f};			// Sampling PDF with reversed incident and excident directions
};

// Check packing
static_assert(sizeof(EvalValue) == 24, "Unexpected packing.");


// Monte Carlo f/p value
struct Throughput {
	Spectrum weight {1.0f};
	float guideWeight {1.0f};	// Custom heuristic to render filtering guides (normals...)
};

}} // namespace mufflon::math