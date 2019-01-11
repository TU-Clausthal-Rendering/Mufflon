﻿#pragma once

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
		bck.m_flux = Spectrum { color }; // TODO: compute real flux (depends on scene size)
		return bck;
	}
	static Background envmap(TextureHandle envmap) {
		mAssert(envmap != nullptr);
		Background bck{ BackgroundType::ENVMAP };
		bck.m_envLight = envmap;
		// Flux will be computed together with the SAT
		bck.m_flux = Spectrum { 1.0f }; // TODO: compute real flux (depends on scene size)
		bck.m_color = Spectrum { 1.0f }; // Default factor, TODO: real factor
		return bck;
	}

	void set_scale(const Spectrum& color) {
		// Rescale flux to the new factor
		m_flux *= color / m_color; // TODO: this can fail if m_color=0. Solution: use 'raw' flux and multiply with color on acquire. Must be in coincidence with colred backs
		m_color = color;
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
	const BackgroundDesc<dev> acquire_const();

	constexpr BackgroundType get_type() noexcept {
		return m_type;
	}

	ConstTextureHandle get_envmap() const noexcept {
		return m_envLight;
	}
	TextureHandle get_envmap() noexcept {
		return m_envLight;
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
};

template DeviceManagerConcept<Background>;

}}} // namespace mufflon::scene::lights