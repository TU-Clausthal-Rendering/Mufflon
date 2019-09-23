#pragma once

#include "util/int_types.hpp"
#include "render_target.hpp"

namespace mufflon {

enum class Device : unsigned char;

namespace renderer {

// TODO: 'concept'-checks?
struct RadianceTarget {
	static constexpr const char NAME[] = "Radiance";
	using PixelType = float;
	static constexpr u32 NUM_CHANNELS = 3u;

	// Optional: by defining these constants you can enforce the
	// presence of render targets (whether they are visible in e.g.
	// the GUI is another matter; our current read-back makes it
	// such that they also appear turned on in the GUI and consequently
	// will always be written to disk as part of a screenshot command)
	//static constexpr bool REQUIRED = true;
	//static constexpr bool VARIANCE_REQUIRED = true; // Implies REQUIRED

	// Optional: Usually mean and variance will be available for
	// each target. Enabling this constant produces 'iterResult' and
	// mean instead. I.e. the 0st and 1st moment are recorded instead
	// the 1st and 2nd.
	//static constexpr bool REDUCE_MOMENTS = true;
};

struct PositionTarget {
	static constexpr const char NAME[] = "Position";
	using PixelType = float;
	static constexpr u32 NUM_CHANNELS = 3u;
};

struct NormalTarget {
	static constexpr const char NAME[] = "Normal";
	using PixelType = float;
	static constexpr u32 NUM_CHANNELS = 3u;
};

struct AlbedoTarget {
	static constexpr const char NAME[] = "Albedo";
	using PixelType = float;
	static constexpr u32 NUM_CHANNELS = 3u;
};

struct LightnessTarget {
	static constexpr const char NAME[] = "Lightness";
	using PixelType = float;
	static constexpr u32 NUM_CHANNELS = 1u;
};

struct DensityTarget {
	static constexpr const char NAME[] = "Density";
	using PixelType = float;
	static constexpr u32 NUM_CHANNELS = 1u;
	static constexpr bool REDUCE_MOMENTS = true;
};

}} // namespace muffon::renderer
