#pragma once

#include "lights.hpp"
#include "core/export/api.h"
#include "util/assert.hpp"
#include "core/scene/handles.hpp"
#include "core/scene/textures/texture.hpp"
#include "core/scene/textures/interface.hpp"
#include "core/concepts.hpp"
#include <ei/vector.hpp>

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
	Background() : m_type(BackgroundType::COLORED), m_color(ei::Vec3{ 0.f }) {}
	Background(const Background&) = delete;
	Background(Background&&) = default;
	Background& operator=(const Background&) = delete;
	Background& operator=(Background&&) = default;
	~Background() = default;

	// Constructors for creating the proper type of background
	static Background black() {
		Background bck{ BackgroundType::COLORED };
		bck.m_color = Spectrum { 0.0f };
		bck.m_flux = Spectrum { 0.0f };
		return bck;
	}
	static Background colored(Spectrum color) {
		Background bck{ BackgroundType::COLORED };
		bck.m_color = Spectrum { color };
		// Flux will be computed in acquire_const
		bck.m_flux = Spectrum { 1.0f };
		return bck;
	}
	static Background envmap(TextureHandle envmap) {
		mAssert(envmap != nullptr);
		Background bck{ BackgroundType::ENVMAP };
		bck.m_envLight = envmap;
		// Flux will be computed together with the SAT
		bck.m_flux = Spectrum { 1.0f }; // TODO: compute real flux (depends on scene size)
		bck.m_color = Spectrum { 1.0f }; // Default factor
		return bck;
	}

	void set_scale(const Spectrum& color) {
		m_color = color;
	}
	const Spectrum& get_scale() const noexcept {
		return m_color;
	}

	// Creates a copy of the background suited for the given deviec
	template < Device newDev >
	void synchronize() {
		m_envLight->synchronize<newDev>();
		m_summedAreaTable->synchronize<newDev>();
	}

	template < Device dev >
	void unload() {
		if(m_envLight) m_envLight->unload<dev>();
		if(m_summedAreaTable) m_summedAreaTable->unload<dev>();
	}

	template< Device dev >
	const BackgroundDesc<dev> acquire_const(const ei::Box& bounds);

	constexpr BackgroundType get_type() const noexcept {
		return m_type;
	}

	ConstTextureHandle get_envmap() const noexcept {
		return m_envLight;
	}
	TextureHandle get_envmap() noexcept {
		return m_envLight;
	}

	const Spectrum& get_color() const noexcept {
		return m_color;
	}


private:
	Background(BackgroundType type) : m_type(type) {}

	BackgroundType m_type;
	Spectrum m_color;				// Color for uniform backgrounds OR scale in case of envLights
	// Multiple textures for the environment + sampling
	TextureHandle m_envLight = nullptr;
	std::unique_ptr<textures::Texture> m_summedAreaTable = nullptr;
	Spectrum m_flux;				// Precomputed value for the flux of the environment light

	void compute_envmap_flux(const ei::Box& bounds);
	void compute_constant_flux(const ei::Box& bounds);
};

template DeviceManagerConcept<Background>;

}}} // namespace mufflon::scene::lights