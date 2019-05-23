#pragma once
#include "util/int_types.hpp"

namespace mufflon {	namespace scene { namespace textures {

/*
 * A list of supported texture formats for this renderer.
 * While Hardware and texture formats may support many more, this list is rather short
 * because each format must also be implemented in the CpuTexture.
 * A loader must choose the most appropriate target format which is supported internally.
 * Also, some of the formats cannot be aquired for write mode (the RGB ones) on GPU side.
 *
 * Format semantics:
 * ...XU	Unsigned int per channel with X bits
 * ...XF	Float with X bits
 */
enum class Format : u16 {
	R8U,
	RG8U,
	RGBA8U,
	R16U,
	RG16U,
	RGBA16U,
	R16F,
	RG16F,
	RGBA16F,
	R32F,
	RG32F,
	RGBA32F,

	NUM
};
}}} // namespace mufflon::scene::textures