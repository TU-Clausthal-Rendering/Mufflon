#pragma once

#include "core/math/sampling.hpp"

namespace mufflon { namespace renderer {

// Simplified version of "5D Covariance Tracing" from Belour et al.
class Footprint2DCov {
public:
	void init(float sourceArea, float initSolidAngle) {
		m_xx = sourceArea;
		m_aa = 2 * ei::PI * initSolidAngle;	// TODO: use 2π scaling like in BSDF for angle?
		m_ax = 0.0f;
	}

	// Get the variance of positions (unit m²)
	float get_area() const {
		// Could be:???
		//return m_xx;

		// 'Slice' the inverse matrix and then invert again.
		// This means we only need the matrix-inverse of the upper left element
		// which then must be inverted as scalar.
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
		//  H: | 1 H |
		//     | 0 1 |
		// V' = Hᵀ V H
		m_aa += 2 * mean_curvature * m_ax + mean_curvature * mean_curvature * m_xx;
		m_ax += mean_curvature * m_xx;
	////	m_aa += mean_curvature * mean_curvature * m_xx;
	}

	// Update a BSDF event by using the value of its PDF
	void update_sampling(AngularPdf pdf) {
		// Assumption: pdf is Gaussian around the sampled value
		// -> value can be inverted for standard deviation.
		float aa = 1.0f / (2 * ei::PI * float(pdf));
	//	float aa = 1.0f / float(pdf);
		//  B: | 0 0  |
		//     | 0 aa |
		// Applying a BSDF is a convolution -> V' = V + B.
		m_aa += aa;// + 1e-4f;
		// However: the paper does a lot of wired inversion stuff here.
		// V' = (V⁺ + B⁺)⁺     where ⁺ is the pseudo-inverse.
		// It has somehow to do with the frequency space, but I cannot figure
		// out why all these inversions should be necessary.
		/*float detA = m_xx * m_aa - m_ax * m_ax;
		float ixx = detA != 0.0f ? m_aa / detA : (m_xx == 0.0f ? 0.0f : 1.0f / m_xx);
		float iax = detA != 0.0f ? -m_ax / detA : (m_xx == 0.0f ? 0.0f : -1.0f / m_ax);
		float iaa = detA != 0.0f ? m_xx / detA : (m_aa == 0.0f ? 0.0f : 1.0f / m_aa);
		iaa += 1.0f / aa;
		float detIA = ixx * iaa - iax * iax;
		m_xx = detIA != 0.0f ? iaa / detIA : (ixx == 0.0f ? 0.0f : 1.0f / ixx);
		m_ax = detIA != 0.0f ? -iax / detIA : (ixx == 0.0f ? 0.0f : -1.0f / iax);
		m_aa = detIA != 0.0f ? ixx / detIA : (iaa == 0.0f ? 0.0f : 1.0f / iaa);//*/
	//	m_aa = 2.0f / (1.0f / m_aa + (2 * ei::PI * float(pdf)));
	}

	// Full update to aggregate all above functions into a single one.
	void update(float outCosAbs, AngularPdf pdf, float mean_curvature) {
//		update_projection(cosInAbs);
		update_sampling(pdf);
		update_curvature(mean_curvature);
		update_inv_projection(outCosAbs);
	}

	Footprint2DCov add_segment(float pdf, bool orthographic, float mean_curvature,
							float outCosAbs, float distance, float inCosAbs) const {
		Footprint2DCov f = *this;
		if(orthographic) {
			f.m_xx = 1.0f / pdf;
		} else {
			f.update_sampling(AngularPdf{pdf});
			f.update_curvature(-mean_curvature);
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

class FootprintV0 {
public:
	void init(float sourceArea, float initSolidAngle) {
		// TODO: use 2π scaling like in BSDF for angle?
		m_x = sqrt(sourceArea);
		m_a = sqrt(initSolidAngle);
	}

	float get_area() const {
		return m_x * m_x;
	}

	FootprintV0 add_segment(float pdf, bool orthographic, float mean_curvature,
							float outCosAbs, float distance, float inCosAbs) const {
		FootprintV0 f = *this;
		if(orthographic) {
			f.m_x = 1.0f / sqrt(pdf);
		} else {
			f.m_x += m_a * distance;
			f.m_a += 1.0f / sqrt(pdf);
		}
		return f;
	}
private:
	float m_x = 0.0f;
	float m_a = 0.0f;
};

class FootprintV1 {
public:
	void init(float sourceArea, float initSolidAngle) {
		// TODO: use 2π scaling like in BSDF for angle?
		m_x = sqrt(sourceArea);
		m_a = sqrt(2 * ei::PI * initSolidAngle);
	}

	float get_area() const {
		return m_x * m_x;
	}

	FootprintV1 add_segment(float pdf, bool orthographic, float mean_curvature,
							float outCosAbs, float distance, float inCosAbs) const {
		FootprintV1 f = *this;
		if(orthographic) {
			f.m_x = 1.0f / sqrt(pdf);
		} else {
			f.m_a += 1.0f / sqrt(2 * ei::PI * pdf)
				//+ ei::abs(mean_curvature) * m_x;
				- mean_curvature * m_x / (2 * ei::PI);
			// Leave tangent space
			f.m_x *= sqrt(outCosAbs);
			// Travel
			f.m_x += f.m_a * distance;
			if(f.m_x < 0.0f) {
				f.m_x = -f.m_x;
				f.m_a = -f.m_a;
			}
			// Enter tangent space
			f.m_x /= sqrt(inCosAbs);
		}
		return f;
	}
private:
	float m_x = 0.0f;
	float m_a = 0.0f;
};


//using Footprint2D = Footprint2DCov;
using Footprint2D = FootprintV1;

}} // namespace mufflon::renderer