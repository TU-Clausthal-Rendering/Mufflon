#pragma once

#include "util/types.hpp"

namespace mufflon { namespace scene { namespace materials {

	/**
	 * A medium is a volumetric material. We do not support true volume rendering
	 * at the moment. Therefore a medium is a rather simple construct which provides
	 * uniform absorption and most important the refraction index of the current volume.
	 */
	class Medium {
	public:
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

		const ei::Vec2& get_refraction_index() const {
			return m_refractionIndex;
		}
		const Spectrum& get_absorption_coeff() const {
			return m_absorptionCoeff;
		}

		// Compute the transmission in this medium for the given travel distance.
		Spectrum get_transmission(float distance) const {
			return exp(m_absorptionCoeff * -distance);
		}
	private:
		ei::Vec2 m_refractionIndex {1.0f, 0.0f};	// Complex refraction index, complex part is 0 for dielectrics
		Spectrum m_absorptionCoeff {0.0f};			// Absorption coefficient λ used as transmission=exp(-d*λ)
	};

	using MediumHandle = u16;

}}} // namespace mufflon::scene::materials