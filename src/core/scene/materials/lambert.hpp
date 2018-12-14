#pragma once

#include "material.hpp"
#include "core/export/api.h"
#include "core/math/sampling.hpp"
#include "core/scene/handles.hpp"
#include "core/scene/textures/texture.hpp"
#include "core/scene/textures/interface.hpp"

namespace mufflon { namespace scene { namespace materials {

struct LambertParameterPack {
	Spectrum albedo;
};

template<Device dev>
struct LambertDesc {
	textures::ConstTextureDevHandle_t<dev> albedoTex;

	CUDA_FUNCTION void fetch(const UvCoordinate& uvCoordinate, char* outBuffer) const {
		*as<LambertParameterPack>(outBuffer) = LambertParameterPack{
			Spectrum{ sample(albedoTex, uvCoordinate) }
		};
	}
};

// Class for the handling of the Lambertian material.
class Lambert : public IMaterial {
public:
	Lambert(TextureHandle albedo) :
		IMaterial{Materials::LAMBERT},
		m_albedo{albedo}
	{}

	MaterialPropertyFlags get_properties() const noexcept final {
		return MaterialPropertyFlags::REFLECTIVE;
	}

	std::size_t get_descriptor_size(Device device) const final {
		device_switch(device, return sizeof(LambertDesc<dev>));
		return 0;
	}

	std::size_t get_parameter_pack_size() const final {
		return sizeof(LambertParameterPack);
	}

	char* get_descriptor(Device device, char* outBuffer) const final {
		// First write the general descriptor and then append the lambert specific one
		outBuffer = IMaterial::get_descriptor(device, outBuffer);
		device_switch(device,
			*as<LambertDesc<dev>>(outBuffer) =
				LambertDesc<dev>{ m_albedo->acquire_const<dev>() };
			return outBuffer + sizeof(LambertDesc<dev>);
		);
		return nullptr;
	}

	TextureHandle get_emissive_texture() const final { return nullptr; }

	Medium compute_medium() const final {
		// Use some average dielectric refraction index and a maximum absorption
		return Medium{ei::Vec2{1.3f, 0.0f}, Spectrum{std::numeric_limits<float>::infinity()}};
	}
private:
	TextureHandle m_albedo;
};



// The importance sampling routine
CUDA_FUNCTION math::PathSample
lambert_sample(const LambertParameterPack& params,
			   const Direction& incidentTS,
			   const math::RndSet2_1& rndSet) {
	// Importance sampling for lambert: BRDF * cos(theta)
	Direction excidentTS = math::sample_dir_cosine(rndSet.u0, rndSet.u1).direction;
	// Copy the sign for two sided diffuse
	return math::PathSample {
		Spectrum{params.albedo},
		math::PathEventType::REFLECTED,
		excidentTS * ei::sgn(incidentTS.z),
		AngularPdf(excidentTS.z / ei::PI),
		AngularPdf(ei::abs(incidentTS.z) / ei::PI)
	};
}

// The evaluation routine
CUDA_FUNCTION math::EvalValue
lambert_evaluate(const LambertParameterPack& params,
				 const Direction& incidentTS,
				 const Direction& excidentTS) {
	// No transmission - already checked by material, but in a combined model we might get a call
	if(incidentTS.z * excidentTS.z < 0.0f) return math::EvalValue{};
	// Two sided diffuse (therefore the abs())
	return math::EvalValue {
		params.albedo / ei::PI,
		ei::abs(excidentTS.z),
		AngularPdf(ei::abs(excidentTS.z) / ei::PI),
		AngularPdf(ei::abs(incidentTS.z) / ei::PI)
	};
}

// The albedo routine
CUDA_FUNCTION Spectrum
lambert_albedo(const LambertParameterPack& params) {
	return params.albedo;
}

}}} // namespace mufflon::scene::materials
