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
	Spectrum scale;

	CUDA_FUNCTION int fetch(const UvCoordinate& uvCoordinate, char* outBuffer) const {
		*as<EmissiveParameterPack>(outBuffer) = EmissiveParameterPack{
			Spectrum{ sample(emission, uvCoordinate) } * scale
		};
		return sizeof(EmissiveParameterPack);
	}
};

// Lambertian self-emitting surface (uniform radiance in all view directions).
// This class is a special case - it has an BSDF value of 0 and is not sampleable
// instead it is the only class providing a non-zero output on get_emission().
class Emissive : public IMaterial {
public:
	Emissive(TextureHandle emissiveTex, Spectrum scale) :
		IMaterial{Materials::EMISSIVE},
		m_emission{emissiveTex},
		m_scale(scale)
	{}

	MaterialPropertyFlags get_properties() const noexcept final {
		return MaterialPropertyFlags::EMISSIVE;
	}

	std::size_t get_descriptor_size(Device device) const final {
		std::size_t s = IMaterial::get_descriptor_size(device);
		device_switch(device, return sizeof(EmissiveDesc<dev>) + s);
		return 0;
	}

	std::size_t get_parameter_pack_size() const final {
		return IMaterial::get_parameter_pack_size()
			+ sizeof(EmissiveParameterPack);
	}

	char* get_descriptor(Device device, char* outBuffer) const final {
		// First write the general descriptor and then append the lambert specific one
		outBuffer = IMaterial::get_descriptor(device, outBuffer);
		device_switch(device,
			(*as<EmissiveDesc<dev>>(outBuffer) =
				 EmissiveDesc<dev>{ m_emission->acquire_const<dev>(), m_scale });
			return outBuffer + sizeof(EmissiveDesc<dev>);
		);
		return nullptr;
	}

	Emission get_emission() const final { return {m_emission, m_scale}; }

	Medium compute_medium() const final {
		// Use some average dielectric refraction index and a maximum absorption
		return Medium{ei::Vec2{1.3f, 0.0f}, Spectrum{std::numeric_limits<float>::infinity()}};
	}
private:
	TextureHandle m_emission;
	Spectrum m_scale;
};

// The albedo routine
CUDA_FUNCTION Spectrum
emissive_albedo(const EmissiveParameterPack& params) {
	// Return 0 to force layered models to never sample this.
	return Spectrum{0.0f};
}

}}} // namespace mufflon::scene::materials
