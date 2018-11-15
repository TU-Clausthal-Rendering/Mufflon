#pragma once

#include "util/types.hpp"
#include "core/scene/types.hpp"

namespace mufflon { namespace renderer {

enum class Interaction : u8 {
	VOID,			// The ray missed the scene (no intersection)
	SURFACE,		// A standard material interaction
	CAMERA,			// A camera start vertex
	LIGHT,			// A light start vertex (on hitting lightsources the SURFACE type is set)
};

struct Throughput {
	ei::Vec3 weight;
	float guideWeight;
};

// Collection of parameters produced by a random walk
// TODO: vertex customization?
struct PathHead {
	Throughput throughput;			// General throughput with guide heuristics
	scene::Point position;
	float prevPdfF;					// Forward PDF of the last sampling PDF
	scene::Direction incident;		// May be zero-vector for start points
	float prevPdfB;					// Backward PDF of the last sampling PDF
	// TODO: prevAreaPDF ?
	Interaction type;
};

}} // namespace mufflon::renderer