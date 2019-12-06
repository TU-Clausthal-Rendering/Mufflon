#pragma once

#include "util/types.hpp"
#include "core/export/core_api.h"

namespace mufflon { namespace scene { namespace materials {

/**
 * A medium is a volumetric material. We do not support true volume rendering
 * at the moment. Therefore a medium is a rather simple construct which provides
 * uniform absorption and most important the refraction index of the current volume.
 */
class Medium {
public:
	Medium() = default;
	Medium(const ei::Vec2 refrIndex, const Spectrum& absorptionCoeff) :
		m_refractionIndex{refrIndex},
		m_absorptionCoeff{absorptionCoeff}
	{}

	void set_refraction_index(float n) {
		m_refractionIndex.x = n;
		m_refractionIndex.y = 0.0f;
	}
	void set_refraction_index(ei::Vec2 eta) {
		m_refractionIndex = eta;
	}
	void set_absorption_coeff(const Spectrum& coeff) {
		m_absorptionCoeff = coeff;
	}

	CUDA_FUNCTION __forceinline__ const ei::Vec2& get_refraction_index() const {
		return m_refractionIndex;
	}
	CUDA_FUNCTION __forceinline__ const Spectrum& get_absorption_coeff() const {
		return m_absorptionCoeff;
	}

	// Compute the transmission in this medium for the given travel distance.
	CUDA_FUNCTION __forceinline__ Spectrum get_transmission(float distance) const {
		return exp(m_absorptionCoeff * -distance);
	}

	bool operator==(const Medium& rhs) const {
		// Although we work with float we can use ==, because we expect either
		// the exact same definition or treat the medium as a different one.
		return (m_refractionIndex == rhs.m_refractionIndex)
			&& (m_absorptionCoeff == rhs.m_absorptionCoeff);
	}
private:
	// TODO: half packing for GPU friendly alignment?
	ei::Vec2 m_refractionIndex {1.0f, 0.0f};	// Complex refraction index, complex part is 0 for dielectrics
	Spectrum m_absorptionCoeff {0.0f};			// Absorption coefficient λ used as transmission=exp(-d*λ)
};

}}} // namespace mufflon::scene::materials