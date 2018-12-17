#pragma once

#include "material.hpp"
#include "core/export/api.h"
#include "core/math/sampling.hpp"
#include "core/scene/textures/texture.hpp"
#include "core/scene/textures/interface.hpp"

namespace mufflon { namespace scene { namespace materials {

struct EmissiveParameterPack {
	Spectrum radiance;
};

template<Device dev>
struct EmissiveDesc {
	textures::ConstTextureDevHandle_t<dev> emission;
	float scale;

	EmissiveDesc(textures::ConstTextureDevHandle_t<dev> emission,
		float scale) : emission(emission), scale(scale) {}

	CUDA_FUNCTION void fetch(const UvCoordinate& uvCoordinate, char* outBuffer) const {
		*as<EmissiveParameterPack>(outBuffer) = EmissiveParameterPack{
			Spectrum{ sample(emission, uvCoordinate) } * scale
		};
	}
};

// TODO: scale factor
// TODO: apply scale in light-tree creation!

// Lambertian self-emitting surface (uniform radiance in all view directions).
// This class is a special case - it has an BSDF value of 0 and is not sampleable
// instead it is the only class providing a non-zero output on get_emission().
class Emissive : public IMaterial {
public:
	Emissive(TextureHandle emissiveTex, float scale) :
		IMaterial{Materials::LAMBERT},
		m_emission{emissiveTex},
		m_scale(scale)
	{}

	MaterialPropertyFlags get_properties() const noexcept final {
		return MaterialPropertyFlags::EMISSIVE;
	}

	std::size_t get_descriptor_size(Device device) const final {
		device_switch(device, return sizeof(EmissiveDesc<dev>));
		return 0;
	}

	std::size_t get_parameter_pack_size() const final {
		return sizeof(EmissiveParameterPack);
	}

	char* get_descriptor(Device device, char* outBuffer) const final {
		// First write the general descriptor and then append the lambert specific one
		outBuffer = IMaterial::get_descriptor(device, outBuffer);
		device_switch(device,
			*as<EmissiveDesc<dev>>(outBuffer) =
				EmissiveDesc<dev>( m_emission->acquire_const<dev>(), m_scale );
			return outBuffer + sizeof(EmissiveDesc<dev>);
		);
		return nullptr;
	}

	TextureHandle get_emissive_texture() const final { return m_emission; }

	Medium compute_medium() const final {
		// Use some average dielectric refraction index and a maximum absorption
		return Medium{ei::Vec2{1.3f, 0.0f}, Spectrum{std::numeric_limits<float>::infinity()}};
	}
private:
	TextureHandle m_emission;
	float m_scale;
};

// The albedo routine
CUDA_FUNCTION Spectrum
emissive_albedo(const EmissiveParameterPack& params) {
	// Return 0 to force layered models to never sample this.
	return Spectrum{0.0f};
}

}}} // namespace mufflon::scene::materials
