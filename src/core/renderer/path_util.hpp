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

// Collection of parameters produced by a random walk
// TODO: better alignment for GPU
// TODO: vertex customization?
struct PathHead {
	Interaction type;
	scene::Point position;
	scene::Direction incident;		// May be zero-vector for start points
	float prevPdfF, prevPdfB;		// Forward and backward PDF of the last sampling PDF
	// TODO: prevAreaPDF ?
	ei::Vec3 throughput;			// TODO: general throughput with guide heuristics
};

}} // namespace mufflon::renderer