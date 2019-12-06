#pragma once

#include "lights.hpp"
#include "core/export/core_api.h"
#include "util/assert.hpp"
#include "core/scene/handles.hpp"
#include "core/scene/textures/texture.hpp"
#include "core/scene/textures/interface.hpp"
#include "core/concepts.hpp"
#include "core/memory/generic_resource.hpp"
#include <ei/vector.hpp>
#include <variant>

namespace mufflon { namespace scene { namespace lights {

/**
 * This class represents the scene's background.
 * Possibilities range from a plain black background over an envmap
 * to analytical models. All that is required is 1. they need to
 * be sample-able from a direction and 2. they need to be able to
 * compute accumulated flux.
 */
class Background {
public:
	Background(const BackgroundType type);
	Background(const Background&) = delete;
	Background(Background&&) = default;
	Background& operator=(const Background&) = delete;
	Background& operator=(Background&&) = default;
	~Background() = default;

	// Constructors for creating the proper type of background
	static Background black() {
		Background bck{ BackgroundType::COLORED };
		bck.m_params = MonochromParams{ Spectrum{0.f} };
		return bck;
	}
	static Background colored(Spectrum color) {
		Background bck{ BackgroundType::COLORED };
		bck.m_params = MonochromParams{ color };
		// Flux will be computed in acquire_const
		bck.m_flux = Spectrum{ 1.0f };
		return bck;
	}
	static Background envmap(TextureHandle envmap) {
		mAssert(envmap != nullptr);
		Background bck{ BackgroundType::ENVMAP };
		bck.m_params = EnvmapParams{ envmap, nullptr };
		// Flux will be computed together with the SAT
		bck.m_flux = Spectrum{ 1.0f }; // TODO: compute real flux (depends on scene size)
		return bck;
	}
	static Background sky_hosek(const ei::Vec3& sunDir, const float solarRadius,
								const float turbidity, const float albedo) {
		Background bck{ BackgroundType::SKY_HOSEK };
		bck.m_params = SkyParams{ sunDir, solarRadius, turbidity, albedo };
		// Flux will be computed in acquire_const
		bck.m_flux = Spectrum{ 1.0f };
		// Compute the model parameters
		/*bck.m_skyModel.resize(sizeof(HosekSkyModel));
		auto& model = *reinterpret_cast<HosekSkyModel*>(bck.m_skyModel.template acquire<Device::CPU>());
		model = compute_sky_model_params(sunDir, solarRadius, turbidity, albedo);*/
		return bck;
	}

	// Creates a copy of the background suited for the given deviec
	template < Device newDev >
	void synchronize() {
		switch(m_type) {
			case BackgroundType::ENVMAP:
				std::get<EnvmapParams>(m_params).envLight->synchronize<newDev>();
				if(std::get<EnvmapParams>(m_params).summedAreaTable)
					std::get<EnvmapParams>(m_params).summedAreaTable->synchronize<newDev>();
				break;
			case BackgroundType::SKY_HOSEK:	// Fallthrough - nothing to be done
			case BackgroundType::COLORED:	// Fallthrough - nothing to be done
			default:
				break;
		}
	}

	template < Device dev >
	void unload() {
		switch(m_type) {
			case BackgroundType::ENVMAP:
				std::get<EnvmapParams>(m_params).envLight->unload<dev>();
				if(std::get<EnvmapParams>(m_params).summedAreaTable)
					std::get<EnvmapParams>(m_params).summedAreaTable->unload<dev>();
				break;
			case BackgroundType::SKY_HOSEK:	// Fallthrough - nothing to be done
			case BackgroundType::COLORED:	// Fallthrough - nothing to be done
			default:
				break;
		}
	}

	template< Device dev >
	const BackgroundDesc<dev> acquire_const(const ei::Box& bounds);

	BackgroundType get_type() const noexcept {
		return m_type;
	}

	// Monochrom parameters
	void set_monochrom_color(const Spectrum& color) {
		std::get<MonochromParams>(m_params).color = color;
	}
	const Spectrum& get_monochrom_color() const {
		return std::get<MonochromParams>(m_params).color;
	}

	// Envmap parameters
	void set_scale(const Spectrum& color) {
		m_scale = color;
	}
	const Spectrum& get_scale() const {
		return m_scale;
	}
	ConstTextureHandle get_envmap() const {
		return std::get<EnvmapParams>(m_params).envLight;
	}
	TextureHandle get_envmap() {
		return std::get<EnvmapParams>(m_params).envLight;
	}

	// Sky parameters
	void set_sky_sun_direction(const ei::Vec3& dir) {
		std::get<SkyParams>(m_params).sunDir = dir;
	}
	ei::Vec3 get_sky_sun_direction() const {
		return std::get<SkyParams>(m_params).sunDir;
	}
	void set_sky_solar_radius(const float radius) {
		std::get<SkyParams>(m_params).solarRadius = ei::max(0.f, radius);
	}
	float get_sky_solar_radius() const {
		return std::get<SkyParams>(m_params).solarRadius;
	}
	void set_sky_turbidity(const float turbidity) {
		std::get<SkyParams>(m_params).turbidity = ei::max(1.f, ei::min(10.f, turbidity));
	}
	float get_sky_turbidity() const {
		return std::get<SkyParams>(m_params).turbidity;
	}
	void set_sky_albedo(const float albedo) {
		std::get<SkyParams>(m_params).albedo = ei::max(0.f, ei::min(1.f, albedo));
	}
	float get_sky_albedo() const {
		return std::get<SkyParams>(m_params).albedo;
	}

private:
	// In essence, the background is a variant of different types: it can (for now)
	// be either monochromatic, an environment map, or an analytic sky model.
	// This boils down to a tagged union
	struct MonochromParams {
		Spectrum color{ 0.f, 0.f, 0.f };
	};
	struct EnvmapParams {
		TextureHandle envLight{ nullptr };
		std::unique_ptr<textures::Texture> summedAreaTable{ nullptr };
	};
	struct SkyParams {
		ei::Vec3 sunDir{ 0.f, 1.f, 0.f };
		float solarRadius{ 0.00445059f };
		float turbidity{ 1.f };
		float albedo{ 1.f };
	};

	static HosekSkyModel compute_sky_model_params(const SkyParams& params);
	void compute_envmap_flux(const ei::Box& bounds);
	void compute_constant_flux(const ei::Box& bounds);
	void compute_sky_flux(const ei::Box& bounds);

	std::variant<MonochromParams, EnvmapParams, SkyParams> m_params;
	BackgroundType m_type;

	Spectrum m_scale;
	Spectrum m_flux;				// Precomputed value for the flux of the environment light
};

}} // namespace scene::lights

template struct DeviceManagerConcept<scene::lights::Background>;

} // namespace mufflon