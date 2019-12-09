#pragma once
#include "core/scene/textures/format.hpp"
#include "util/log.hpp"
#include "gl_wrapper.hpp"

namespace mufflon {
namespace gl {
    struct TextureFormat {
		TextureInternal internal;
		TextureSetFormat setFormat;
		TextureSetType setType;
    };

    inline TextureFormat convertFormat(scene::textures::Format f, bool isSrgb) {
        if(isSrgb) {
			if (f != scene::textures::Format::RGBA8U)
				logError("opengl srgb texture is only supported for RGBA8U format");
			else return { TextureInternal::SRGBA8U, TextureSetFormat::RGBA, TextureSetType::U8 };
        }
        switch (f) { 
		case scene::textures::Format::R8U: return { TextureInternal::R8U, TextureSetFormat::R, TextureSetType::U8 };
        case scene::textures::Format::RG8U: return {TextureInternal::RG8U, TextureSetFormat::RG, TextureSetType::U8 };
        case scene::textures::Format::RGBA8U: return {TextureInternal::RGBA8U, TextureSetFormat::RGBA, TextureSetType::U8 };
        case scene::textures::Format::R16U: return {TextureInternal::R16U, TextureSetFormat::R, TextureSetType::U16 };
        case scene::textures::Format::RG16U: return {TextureInternal::RG16U, TextureSetFormat::RG, TextureSetType::U16 };
        case scene::textures::Format::RGBA16U: return {TextureInternal::RGBA16U, TextureSetFormat::RGBA, TextureSetType::U16 };
        case scene::textures::Format::R16F: return {TextureInternal::R16F, TextureSetFormat::R, TextureSetType::F16 };
        case scene::textures::Format::RG16F: return {TextureInternal::RG16F, TextureSetFormat::RG, TextureSetType::F16 };
        case scene::textures::Format::RGBA16F: return {TextureInternal::RGBA16F, TextureSetFormat::RGBA, TextureSetType::F16 };
        case scene::textures::Format::R32F: return {TextureInternal::R32F, TextureSetFormat::R, TextureSetType::F32 };
        case scene::textures::Format::RG32F: return {TextureInternal::RG32F, TextureSetFormat::RG, TextureSetType::F32 };
        case scene::textures::Format::RGBA32F: return {TextureInternal::RGBA32F, TextureSetFormat::RGBA, TextureSetType::F32 };
        default: mAssert(false); break;
        }
		mAssert(false);
        return TextureFormat{};
    }
}
}
