#pragma once

#include "types.hpp"
#include "material.hpp"

namespace mufflon { namespace scene { namespace material {

template<Device dev>
struct LambertHandlePack : public HandlePack {
	LambertHandlePack(const HandlePack & baseProperties,
					  ConstAccessor<textures::DeviceTextureHandle<dev>> albedoTex) :
		HandlePack(baseProperties),
		albedoTex(albedoTex)
	{}

	ConstAccessor<textures::DeviceTextureHandle<dev>> albedoTex;
};

struct LambertParameterPack : public ParameterPack {
	LambertParameterPack(Spectrum albedo, MediumHandle innerMedium, MediumHandle outerMedium) :
		albedo(albedo)
	{
		type = Materials::LAMBERT;
		this->innerMedium = innerMedium;
		this->outerMedium = outerMedium;
	}
	Spectrum albedo;
};

// Class for the handling of the Lambertian material.
class Lambert :
	public IMaterial {
public:
	std::size_t get_handle_pack_size(Device device) const final;
	std::size_t get_parameter_pack_size() const final { return sizeof(LambertParameterPack); }
	void get_handle_pack(Device device, HandlePack* outBuffer) const final;
	void get_parameter_pack_cpu(const HandlePack* handles, const UvCoordinate& uvCoordinate, ParameterPack* outBuffer) const final;
	bool is_emissive() const final { return false; }
	bool is_brdf() const final { return true; }
	bool is_btdf() const final { return false; }
	bool is_halfvector_based() const final { return false; }
private:
	textures::TextureHandle m_albedo;
};

// The importance sampling routine
__host__ __device__ Sample
lambert_sample(const LambertParameterPack& params,
			   const Direction& incidentTS,
			   const RndSet& rndSet) {
	return Sample{};
}

// The evaluation routine

}}} // namespace mufflon::scene::material