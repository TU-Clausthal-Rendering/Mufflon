#pragma once

#include "lights.hpp"
#include "importance_sampling.hpp"
#include "core/export/api.h"
#include "util/assert.hpp"
#include "core/scene/handles.hpp"
#include "core/scene/textures/texture.hpp"
#include "core/scene/textures/interface.hpp"
#include <ei/vector.hpp>

namespace mufflon { namespace scene { namespace lights {

enum class BackgroundType {
	COLORED,
	ENVMAP
};

/**
 * This class represents the scene's background.
 * Possibilities range from a plain black background over an envmap
 * to analytical models. All that is required is 1. they need to 
 * be sample-able from a direction and 2. they need to be able to
 * compute accumulated flux.
 */
template < Device dev >
class Background {
public:
	static constexpr Device DEVICE = dev;

	Background() : m_type(BackgroundType::COLORED), m_color(ei::Vec3{ 0.f }) {}
	Background(const Background&) = default;
	Background(Background&&) = default;
	Background& operator=(const Background&) = default;
	Background& operator=(Background&&) = default;
	~Background() = default;

	// Constructors for creating the proper type of background
	__host__ static Background black() {
		return colored(ei::Vec3{ 0.f });
	}
	__host__ static Background colored(ei::Vec3 color) {
		Background bck{ BackgroundType::COLORED };
		bck.m_color = color;
		return bck;
	}
	__host__ static Background envmap(TextureHandle envmap, TextureHandle summedAreaTable) {
		mAssert(envmap != nullptr);
		Background bck{ BackgroundType::ENVMAP };
		bck.m_envLight.texHandle = envmap->acquire_const<DEVICE>();
		// Fill the summed area table with life
		create_summed_area_table(envmap, summedAreaTable);
		bck.m_envLight.summedAreaTable = summedAreaTable->acquire_const<DEVICE>();

		// TODO: accumulate flux for envmap
		bck.m_envLight.flux = ei::Vec3{ 0.f };

		return bck;
	}

	// Creates a copy of the background suited for the given deviec
	template < Device newDev >
	__host__ Background<newDev> synchronize(TextureHandle envmapOpt, TextureHandle summedOpt) {
		Background<newDev> newBck{ m_type };
		switch(m_type) {
			case BackgroundType::COLORED:
				newBck.m_color = m_color;
				break;
			case BackgroundType::ENVMAP:
				mAssert(envmap != nullptr);
				newBck.m_envLight.flux = m_envLight.flux;
				newBck.m_envLight.texHandle = envmapOpt->acquire_const<newDev>();
				newBck.m_envLight.summedAreaTable = summedOpt->acquire_const<newDev>();
				break;
		}
		return newBck;
	}

	CUDA_FUNCTION Spectrum get_radiance(const ei::Vec3& direction) const {
		switch(m_type) {
			case BackgroundType::COLORED: return m_color;
			case BackgroundType::ENVMAP: {
				ei::Vec4 sample = textures::sample(m_envLight.texHandle, direction);
				return Spectrum{ sample.x, sample.y, sample.z };
			}
			default: mAssert(false); return {};
		}
	}

	// TODO: sample, connect

	constexpr CUDA_FUNCTION BackgroundType get_type() const noexcept {
		return m_type;
	}

	CUDA_FUNCTION __forceinline__ ei::Vec3 get_flux() const {
		switch(m_type) {
			case BackgroundType::COLORED: return ei::Vec3{ 0.f };
			case BackgroundType::ENVMAP: return m_envLight.flux;
			default: mAssert(false); return {};
		}
	}

	constexpr CUDA_FUNCTION const EnvMapLight<DEVICE>& get_envmap_light() const noexcept {
		return m_envLight;
	}

private:
	// Typed constructor is hidden because different types require different extra parameters
	template < Device >
	friend class Background;
	Background(BackgroundType type) : m_type(type) {}

	BackgroundType m_type;
	Spectrum m_color;
	EnvMapLight<DEVICE> m_envLight;
};

}}} // namespace mufflon::scene::lights