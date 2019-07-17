#pragma once

#include "core/math/sampling.hpp"

namespace mufflon { namespace renderer {

// Simplified version of "5D Covariance Tracing" from Belour et al.
class Footprint2D {
public:
	void init(float sourceArea, float initSolidAngle) {
		// TODO: use 2π scaling like in BSDF for angle?
		m_xx = sourceArea;
		m_aa = initSolidAngle;
		m_ax = 0.0f;
	}

	// Get the variance of positions (unit m²)
	float get_area() const {
		// Could be:???
		//return m_xx;

		// 'Slice' the inverse matrix and then invert again.
		// This means we only need the matrix-inverse of the upper left element
		// which then must inverted as scalar.
		return m_aa == 0.0f ? m_xx
							: m_xx - m_ax * m_ax / m_aa;
							//: (m_xx * m_aa - m_ax * m_ax) / m_aa;
	}

	// Get the variance of the angle (unit sr)
	float get_solid_angle() const {
		// Same as for positions
		return m_xx == 0.0f ? m_aa
							: m_aa - m_ax * m_ax / m_xx;
							//: (m_xx * m_aa - m_ax * m_ax) / m_xx;
	}

	// Project the footprint to a surface.
	void update_projection(float inCosAbs) {
		// Op: Scale pos with 1/cosTheta.
		//  S: | 1/cT 0 |
		//     | 0    1 |
		// V' = Sᵀ V S
		m_xx /= inCosAbs * inCosAbs;
		m_ax /= inCosAbs;
	}

	// Unproject from surface. I.e. when leaving the surface use this one.
	void update_inv_projection(float outCosAbs) {
		// see update_projection...
		m_xx *= outCosAbs * outCosAbs;
		m_ax *= outCosAbs;
	}

	void update_travel(float distance) {
		// According to the paper a travel is an angular shear.
		// IM AM CONFUSED: SHOULDN'T IT BE A SPATIAL ONE?
		// I implement an spatial one here. Reasoning: a point light source
		// should have a growing spatial footprint, but if using an angular shear
		// here + get_area() we would end up with zero.
		//  T: | 1 0 |
		//     | d 1 |
		// V' = Tᵀ V T
		m_xx += 2 * distance * m_ax + distance * distance * m_aa;
		m_ax += distance * m_aa;
	}

	void update_curvature(float mean_curvature) {
		// CONTRARY TO THE PAPER: IMPLEMENTED THIS ONE AS ANGULAR SHEAR.
		// See above.
		//  C: | 1 c |
		//     | 0 1 |
		// V' = Cᵀ V C
		m_aa += 2 * mean_curvature * m_ax + mean_curvature * mean_curvature * m_xx;
		m_ax += mean_curvature * m_xx;
	}

	// Update a BSDF event by using the value of its PDF
	void update_sampling(AngularPdf pdf) {
		// Assumption: pdf is Gaussian around the sampled value
		// -> value can be inverted for standard deviation.
	//	float aa = 1.0f / (2 * ei::PI * float(pdf));
		float aa = 1.0f / float(pdf);
		//  B: | 0 0  |
		//     | 0 aa |
		// Applying a BSDF is a convolution -> V' = V + B.
		m_aa += aa;
		// However: the paper does a lot of wired inversion stuff here.
		// V' = (V⁺ + B⁺)⁺     where ⁺ is the pseudo-inverse.
		// It has somehow to do with the frequency space, but I cannot figure
		// out why all these inversions should be necessary.
	}

	// Full update to aggregate all above functions into a single one.
	void update(float outCosAbs, AngularPdf pdf, float mean_curvature) {
//		update_projection(cosInAbs);
		update_sampling(pdf);
		update_curvature(mean_curvature);
		update_inv_projection(outCosAbs);
	}

	Footprint2D add_segment(float pdf, bool orthographic, float mean_curvature,
							float outCosAbs, float distance, float inCosAbs) const {
		Footprint2D f = *this;
		if(orthographic) {
			f.m_xx += 1.0f / pdf;
		} else {
			f.update_sampling(AngularPdf{pdf});
			f.update_curvature(mean_curvature);
			f.update_inv_projection(outCosAbs);
			f.update_travel(distance);
		}
		f.update_projection(inCosAbs);
		return f;
	}
private:
	float m_xx = 0.0f;		// Variance of the position
	float m_aa = 0.0f;		// Variance of the angle
	float m_ax = 0.0f;		// Correlation position-angle
};

}} // namespace mufflon::renderer