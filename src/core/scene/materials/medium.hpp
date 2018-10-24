#pragma once

#include "util/types.hpp"

namespace mufflon { namespace scene { namespace material {

	/**
	 * A medium is a volumetric material. We do not support true volume rendering
	 * at the moment. Therefore a medium is a rather simple construct which provides
	 * uniform absorption and most important the refraction index of the current volume.
	 */
	class Medium {
		void setRefractionIndex(float n) {
			m_refractionIndex.x = n;
			m_refractionIndex.y = 0.0f;
		}
		void setRefractionIndex(ei::Vec2 eta) {
			m_refractionIndex = eta;
		}
		void setAbsorptionCoeff(const Spectrum& coeff) {
			m_absorptionCoeff = coeff;
		}

		ei::Vec2 getRefractionIndex() const {
			return m_refractionIndex;
		}
		const Spectrum& getAbsorptionCoeff() const {
			return m_absorptionCoeff;
		}

		// Compute the transmission in this medium for the given travel distance.
		Spectrum getTransmission(float distance) const {
			return exp(m_absorptionCoeff * -distance);
		}
	private:
		ei::Vec2 m_refractionIndex {1.0f, 0.0f};	// Complex refraction index, complex part is 0 for dielectrics
		Spectrum m_absorptionCoeff {0.0f};			// Absorption coefficient λ used as transmission=exp(-d*λ)
	};

	using MediumHandle = u16;

}}} // namespace mufflon::scene::material