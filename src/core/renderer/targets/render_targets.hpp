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

	/*template < Device dev >
	static void contribute(cuda::Atomic<dev, PixelType> (&pixel)[NUM_CHANNELS], const ei::Vec3& radiance) {
		cuda::atomic_add<dev>(pixel[0], radiance.x);
		cuda::atomic_add<dev>(pixel[1], radiance.y);
		cuda::atomic_add<dev>(pixel[2], radiance.z);
	}*/
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
};

}} // namespace muffon::renderer