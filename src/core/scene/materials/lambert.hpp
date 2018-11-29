#pragma once

#include "material_types.hpp"
#include "material.hpp"
#include "export/api.hpp"
#include "core/math/sampling.hpp"
#include "core/scene/handles.hpp"
#include "core/scene/textures/texture.hpp"

namespace mufflon { namespace scene { namespace materials {

template<Device dev>
struct LambertHandlePack : public HandlePack {
	LambertHandlePack(const HandlePack& baseProperties,
					  textures::ConstTextureDevHandle_t<dev> albedoTex) :
		HandlePack(baseProperties),
		albedoTex(albedoTex)
	{}

	textures::ConstTextureDevHandle_t<dev> albedoTex;
};

struct LambertParameterPack : public ParameterPack {
	LambertParameterPack(const ParameterPack& baseParameters, Spectrum albedo) :
		ParameterPack(baseParameters),
		albedo(albedo)
	{}
	Spectrum albedo; // TODO: alignment is bad
};

// Class for the handling of the Lambertian material.
class Lambert :
	public IMaterial {
public:
	// TODO: constructor
	Lambert(TextureHandle albedo);

	std::size_t get_handle_pack_size(Device device) const final;
	std::size_t get_parameter_pack_size() const final { return sizeof(LambertParameterPack); }
	void get_handle_pack(Device device, HandlePack* outBuffer) const final;
	void get_parameter_pack_cpu(const HandlePack* handles, const UvCoordinate& uvCoordinate, ParameterPack* outBuffer) const final;
	Medium compute_medium() const final;
	bool is_emissive() const final { return false; }
	bool is_brdf() const final { return true; }
	bool is_btdf() const final { return false; }
	bool is_halfvector_based() const final { return false; }
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