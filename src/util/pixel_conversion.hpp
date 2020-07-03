#pragma once

#include "assert.hpp"
#include "core/export/texture_data.h"
#include <ei/vector.hpp>
#include <cuda_fp16.h>
#include <cstddef>
#include <cstdint>

namespace mufflon::util {

enum class TextureChannelDataType {
	UINT8,
	UINT16,
	HALF,
	FLOAT
};

inline std::size_t get_channel_count(const TextureFormat format) noexcept {
	switch(format) {
		case TextureFormat::FORMAT_R8U:
		case TextureFormat::FORMAT_R16U:
		case TextureFormat::FORMAT_R16F:
		case TextureFormat::FORMAT_R32F:	return 1u;
		case TextureFormat::FORMAT_RG8U:
		case TextureFormat::FORMAT_RG16U:
		case TextureFormat::FORMAT_RG16F:
		case TextureFormat::FORMAT_RG32F:	return 2u;
		case TextureFormat::FORMAT_RGB8U:
		case TextureFormat::FORMAT_RGB16U:
		case TextureFormat::FORMAT_RGB16F:
		case TextureFormat::FORMAT_RGB32F:	return 3u;
		case TextureFormat::FORMAT_RGBA8U:
		case TextureFormat::FORMAT_RGBA16U:
		case TextureFormat::FORMAT_RGBA16F:
		case TextureFormat::FORMAT_RGBA32F:	return 4u;
		default:							return 0u;
	}
}

inline std::size_t get_channel_size(const TextureFormat format) noexcept {
	switch(format) {
		case TextureFormat::FORMAT_R8U:
		case TextureFormat::FORMAT_RG8U:
		case TextureFormat::FORMAT_RGB8U:
		case TextureFormat::FORMAT_RGBA8U:	return 1u;
		case TextureFormat::FORMAT_R16U:
		case TextureFormat::FORMAT_RG16U:
		case TextureFormat::FORMAT_RGB16U:
		case TextureFormat::FORMAT_RGBA16U:
		case TextureFormat::FORMAT_R16F:
		case TextureFormat::FORMAT_RG16F:
		case TextureFormat::FORMAT_RGB16F:
		case TextureFormat::FORMAT_RGBA16F:	return 2u;
		case TextureFormat::FORMAT_R32F:
		case TextureFormat::FORMAT_RG32F:
		case TextureFormat::FORMAT_RGB32F:
		case TextureFormat::FORMAT_RGBA32F:	return 4u;
		default:							return 0u;
	}
}

inline TextureChannelDataType get_channel_type(const TextureFormat format) noexcept {
	switch(format) {
		case TextureFormat::FORMAT_R8U:
		case TextureFormat::FORMAT_RG8U:
		case TextureFormat::FORMAT_RGB8U:
		case TextureFormat::FORMAT_RGBA8U:	return TextureChannelDataType::UINT8;
		case TextureFormat::FORMAT_R16U:
		case TextureFormat::FORMAT_RG16U:
		case TextureFormat::FORMAT_RGB16U:
		case TextureFormat::FORMAT_RGBA16U:	return TextureChannelDataType::UINT16;
		case TextureFormat::FORMAT_R16F:
		case TextureFormat::FORMAT_RG16F:
		case TextureFormat::FORMAT_RGB16F:
		case TextureFormat::FORMAT_RGBA16F:	return TextureChannelDataType::HALF;
		case TextureFormat::FORMAT_R32F:
		case TextureFormat::FORMAT_RG32F:
		case TextureFormat::FORMAT_RGB32F:
		case TextureFormat::FORMAT_RGBA32F:	return TextureChannelDataType::FLOAT;
		default: mAssertMsg(false, "Unknown texture format"); return TextureChannelDataType{};
	}
}

inline TextureFormat get_format(const TextureChannelDataType type, const std::size_t channels) noexcept {
	switch(type) {
		case TextureChannelDataType::UINT8:
			switch(channels) {
				case 1: return TextureFormat::FORMAT_R8U;
				case 2: return TextureFormat::FORMAT_RG8U;
				case 3: return TextureFormat::FORMAT_RGB8U;
				case 4: return TextureFormat::FORMAT_RGBA8U;
			}
			break;
		case TextureChannelDataType::UINT16:
			switch(channels) {
				case 1: return TextureFormat::FORMAT_R16U;
				case 2: return TextureFormat::FORMAT_RG16U;
				case 3: return TextureFormat::FORMAT_RGB16U;
				case 4: return TextureFormat::FORMAT_RGBA16U;
			}
			break;
		case TextureChannelDataType::HALF:
			switch(channels) {
				case 1: return TextureFormat::FORMAT_R16F;
				case 2: return TextureFormat::FORMAT_RG16F;
				case 3: return TextureFormat::FORMAT_RGB16F;
				case 4: return TextureFormat::FORMAT_RGBA16F;
			}
			break;
		case TextureChannelDataType::FLOAT:
			switch(channels) {
				case 1: return TextureFormat::FORMAT_R32F;
				case 2: return TextureFormat::FORMAT_RG32F;
				case 3: return TextureFormat::FORMAT_RGB32F;
				case 4: return TextureFormat::FORMAT_RGBA32F;
			}
			break;
	}
	mAssertMsg(false, "Unknown texture format");
	return TextureFormat::FORMAT_NUM;
}

inline std::size_t get_format_size(const TextureFormat format) noexcept {
	return get_channel_count(format) * get_channel_size(format);
}

// Reads a pixel from a byte stream of the given format and converts it into RGBA32F
inline ei::Vec4 read_pixel(const char* in, const TextureFormat source) {
	ei::Vec4 value{ 0.f };
	switch(source) {
		case TextureFormat::FORMAT_RGBA8U:
			value.a = static_cast<float>(reinterpret_cast<const std::uint8_t*>(in)[3u]) / 255.f;
			[[fallthrough]];
		case TextureFormat::FORMAT_RGB8U:
			value.b = static_cast<float>(reinterpret_cast<const std::uint8_t*>(in)[2u]) / 255.f;
			[[fallthrough]];
		case TextureFormat::FORMAT_RG8U:
			value.g = static_cast<float>(reinterpret_cast<const std::uint8_t*>(in)[1u]) / 255.f;
			[[fallthrough]];
		case TextureFormat::FORMAT_R8U:
			value.r = static_cast<float>(*reinterpret_cast<const std::uint8_t*>(in)) / 255.f;
			break;
		case TextureFormat::FORMAT_RGBA16U:
			value.a = static_cast<float>(reinterpret_cast<const std::uint16_t*>(in)[3u]) / 255.f;
			[[fallthrough]];
		case TextureFormat::FORMAT_RGB16U:
			value.b = static_cast<float>(reinterpret_cast<const std::uint16_t*>(in)[2u]) / 255.f;
			[[fallthrough]];
		case TextureFormat::FORMAT_RG16U:
			value.g = static_cast<float>(reinterpret_cast<const std::uint16_t*>(in)[1u]) / 255.f;
			[[fallthrough]];
		case TextureFormat::FORMAT_R16U:
			value.r = static_cast<float>(*reinterpret_cast<const std::uint16_t*>(in)) / 255.f;
			break;
		case TextureFormat::FORMAT_RGBA16F:
			value.a = __half2float(reinterpret_cast<const __half*>(in)[3u]);
			[[fallthrough]];
		case TextureFormat::FORMAT_RGB16F:
			value.a = __half2float(reinterpret_cast<const __half*>(in)[2u]);
			[[fallthrough]];
		case TextureFormat::FORMAT_RG16F:
			value.a = __half2float(reinterpret_cast<const __half*>(in)[1u]);
			[[fallthrough]];
		case TextureFormat::FORMAT_R16F:
			value.a = __half2float(*reinterpret_cast<const __half*>(in));
			break;
		case TextureFormat::FORMAT_RGBA32F:
			value.a = reinterpret_cast<const float*>(in)[3u];
			[[fallthrough]];
		case TextureFormat::FORMAT_RGB32F:
			value.b = reinterpret_cast<const float*>(in)[2u];
			[[fallthrough]];
		case TextureFormat::FORMAT_RG32F:
			value.g = reinterpret_cast<const float*>(in)[1u];
			[[fallthrough]];
		case TextureFormat::FORMAT_R32F:
			value.r = *reinterpret_cast<const float*>(in);
			break;
		default:
			mAssertMsg(false, "Unknown pixel format");
			break;
	}
	return value;
}

// Writes a RGBA32F into a byte stream of the given format
inline void write_pixel(char* out, const ei::Vec4 value, const TextureFormat target) {
	// Buffer for maximum size
	char values[sizeof(float) * 4u];
	std::size_t size = 0u;
	switch(target) {
		case TextureFormat::FORMAT_RGBA8U:
			reinterpret_cast<std::uint8_t*>(values)[3u] = static_cast<std::uint8_t>(value.a * 255.f);
			[[fallthrough]];
		case TextureFormat::FORMAT_RGB8U:
			reinterpret_cast<std::uint8_t*>(values)[2u] = static_cast<std::uint8_t>(value.b * 255.f);
			[[fallthrough]];
		case TextureFormat::FORMAT_RG8U:
			reinterpret_cast<std::uint8_t*>(values)[1u] = static_cast<std::uint8_t>(value.g * 255.f);
			[[fallthrough]];
		case TextureFormat::FORMAT_R8U:
			reinterpret_cast<std::uint8_t*>(values)[0u] = static_cast<std::uint8_t>(value.r * 255.f);
			break;
		case TextureFormat::FORMAT_RGBA16U:
			reinterpret_cast<std::uint16_t*>(values)[3u] = static_cast<std::uint16_t>(value.a * 255.f);
			[[fallthrough]];
		case TextureFormat::FORMAT_RGB16U:
			reinterpret_cast<std::uint16_t*>(values)[2u] = static_cast<std::uint16_t>(value.b * 255.f);
			[[fallthrough]];
		case TextureFormat::FORMAT_RG16U:
			reinterpret_cast<std::uint16_t*>(values)[1u] = static_cast<std::uint16_t>(value.g * 255.f);
			[[fallthrough]];
		case TextureFormat::FORMAT_R16U:
			reinterpret_cast<std::uint16_t*>(values)[0u] = static_cast<std::uint16_t>(value.r * 255.f);
			break;
		case TextureFormat::FORMAT_RGBA16F:
			reinterpret_cast<__half*>(values)[3u] = __float2half(value.a);
			[[fallthrough]];
		case TextureFormat::FORMAT_RGB16F:
			reinterpret_cast<__half*>(values)[2u] = __float2half(value.a);
			[[fallthrough]];
		case TextureFormat::FORMAT_RG16F:
			reinterpret_cast<__half*>(values)[1u] = __float2half(value.a);
			[[fallthrough]];
		case TextureFormat::FORMAT_R16F:
			reinterpret_cast<__half*>(values)[0u] = __float2half(value.a);
			break;
		case TextureFormat::FORMAT_RGBA32F:
			reinterpret_cast<float*>(values)[3u] = value.a;
			[[fallthrough]];
		case TextureFormat::FORMAT_RGB32F:
			reinterpret_cast<float*>(values)[2u] = value.b;
			[[fallthrough]];
		case TextureFormat::FORMAT_RG32F:
			reinterpret_cast<float*>(values)[1u] = value.g;
			[[fallthrough]];
		case TextureFormat::FORMAT_R32F:
			reinterpret_cast<float*>(values)[0u] = value.r;
			break;
		default:
			mAssertMsg(false, "Unknown pixel format");
			break;
	}
}

inline void convert_pixel(const char* in, char* out, const TextureFormat source, const TextureFormat target) {
	const auto pixel = read_pixel(in, source);
	write_pixel(out, pixel, target);
}

} // namespace mufflon::util