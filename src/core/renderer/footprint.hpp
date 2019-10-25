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
							: ei::abs(m_xx - m_ax * m_ax / m_aa);
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
//		m_aa += 2 * distance * m_ax + distance * distance * m_xx;
//		m_ax += distance * m_xx;
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
							   float prevInCos, float prevOutCos, float eta, float distance,
							   float inCos) const {
		Footprint2DCov f = *this;
		if(orthographic) {
			f.m_xx = 1.0f / pdf;
		} else {
			f.update_sampling(AngularPdf{pdf});
			f.update_curvature(mean_curvature * prevInCos);
		//	f.update_inv_projection(ei::abs(prevOutCos));
			f.update_travel(distance);
		}
	//	f.update_projection(ei::abs(inCos));
		if(std::isnan(f.m_aa) || std::isnan(f.m_ax) || std::isnan(f.m_xx))
			__debugbreak();
		return f;
	}
private:
	float m_xx = 0.0f;		// Variance of the position
	float m_aa = 0.0f;		// Variance of the angle
	float m_ax = 0.0f;		// Correlation position-angle
};

class FootprintV0 {
public:
	__host__ __device__ void init(float sourceArea, float initSolidAngle, float pChoice) {
		m_x = sqrt(sourceArea);
		m_a = sqrt(initSolidAngle);
		m_P = pChoice;
	}

	__host__ __device__ float get_area() const {
		return m_x * m_x / m_P;
	}

	__host__ __device__ float get_solid_angle() const {
		return m_a * m_a / m_P;
	}

	__host__ __device__ FootprintV0 add_segment(float pdf, bool orthographic, float mean_curvature,
							float prevInCos, float prevOutCos, float eta, float distance,
							float inCos, float pRoulette) const {
		FootprintV0 f = *this;
		if(orthographic) {
			f.m_x = 1.0f / sqrt(pdf);
		} else {
			f.m_a += 1.0f / sqrt(pdf);
			f.m_x += f.m_a * distance;
		}
		f.m_P *= pRoulette;
		return f;
	}
private:
	float m_x = 0.0f;
	float m_a = 0.0f;
	float m_P = 1.0f;
};

class FootprintV0Sq {
public:
	__host__ __device__ void init(float sourceArea, float initSolidAngle, float pChoice) {
		m_x = sourceArea;
		m_a = initSolidAngle;
		m_P = pChoice;
	}

	__host__ __device__ float get_area() const {
		return m_x / m_P;
	}

	__host__ __device__ FootprintV0Sq add_segment(float pdf, bool orthographic, float mean_curvature,
							float prevInCos, float prevOutCos, float eta, float distance,
							float inCos, float pRoulette) const {
		FootprintV0Sq f = *this;
		if(orthographic) {
			f.m_x = 1.0f / pdf;
		} else {
			f.m_a += 1.0f / pdf;
			f.m_x += f.m_a * distance * distance;
		}
		f.m_P *= pRoulette;
		return f;
	}
private:
	float m_x = 0.0f;
	float m_a = 0.0f;
	float m_P = 1.0f;
};

class FootprintV1 {
public:
	void init(float sourceArea, float initSolidAngle) {
		m_x = sqrt(sourceArea);
		m_a = sqrt(initSolidAngle);
	}

	float get_area() const {
		return m_x * m_x;
	}

	FootprintV1 add_segment(float pdf, bool orthographic, float mean_curvature,
							float prevInCos, float prevOutCos, float eta, float distance,
							float inCos) const {
		FootprintV1 f = *this;
		if(orthographic) {
			f.m_x = 1.0f / sqrt(pdf);
		} else {
			f.m_a += 1.0f / sqrt(pdf) + ei::abs(mean_curvature) * m_x;
			f.m_x += f.m_a * distance;
		}
		return f;
	}
private:
	float m_x = 0.0f;
	float m_a = 0.0f;
};

class FootprintV2 {
public:
	void init(float sourceArea, float initSolidAngle, float pChoice) {
		m_xx = sourceArea;
		m_a = sqrt(initSolidAngle);
		m_P = pChoice;
	}

	float get_area() const {
		return m_xx;
	}

	// eta = ratio of refraction indices n_in / n_out
	FootprintV2 add_segment(float pdf, bool orthographic, float mean_curvature,
							float prevInCos, float prevOutCos, float eta, float distance,
							float inCos, float pRoulette) const {
		FootprintV2 f = *this;
		if(orthographic) {
			f.m_xx = 1.0f / (pdf * ei::abs(inCos));
			//f.m_x = 1.0f / sqrt(pdf);
		} else {
		//	const float xNoProj = sqrt(f.m_xx * ei::abs(prevInCos));
		//	const float Hs = mean_curvature * xNoProj * (1 + 1.0f / ei::abs(prevInCos)) * ei::sign(prevInCos);
		//	const float Hs = mean_curvature * xNoProj * ei::sign(prevInCos);
		//	const float Hs = mean_curvature * sqrt(f.m_xx) * ei::sign(prevInCos);
			const float Hs = mean_curvature * atan(sqrt(f.m_xx)) * ei::sign(prevInCos);
			if(prevInCos * prevOutCos < 0.0f && ei::abs(eta) > 1e-5f) {
				// Refraction
			//	float da = Hs * sqrt(2 * ei::PI);
			//	f.m_a = (f.m_a + Hs) * eta - Hs;
			//	f.m_a = (f.m_a + Hs * ei::abs(prevInCos / prevOutCos)) * eta - Hs;
			//	f.m_a = f.m_a + Hs * ei::abs(prevInCos / prevOutCos) * eta - Hs;
			//	f.m_a = (f.m_a + Hs * ei::abs(prevInCos)) * eta - Hs * ei::abs(prevOutCos);
			//	f.m_a = eta * f.m_a + (eta * prevInCos + prevOutCos) * Hs;
			} else {
				// Reflection
				//f.m_a = f.m_a + 2 * mean_curvature * sqrt(f.m_xx) * ei::sign(prevOutCos);
				//f.m_a = f.m_a + 2 * mean_curvature * sqrt(f.m_xx / ei::abs(prevInCos)) * ei::sign(prevOutCos);
				//f.m_a = f.m_a + 2 * mean_curvature * sqrt(f.m_xx) * (prevOutCos);
			//	f.m_a = f.m_a + 2 * Hs;
				//f.m_a += 2.0f * Hs * prevInCos;
			}
			f.m_a += ei::abs(Hs);
			// Leave tangent space
			f.m_xx *= ei::abs(prevOutCos);
			// BRDF
			f.m_a += 1.0f / (1e-6f + sqrt(pdf));
			// Travel
			f.m_xx += f.m_a * ei::abs(f.m_a) * distance * distance;
			if(f.m_xx < 0.0f) {
				f.m_xx = -f.m_xx;
				f.m_a = -f.m_a;
			}
			// Enter tangent space
			f.m_xx /= ei::abs(inCos) + 1e-6f;
		//	if(f.m_xx > 1e20f || isnan(f.m_xx))
		//		__debugbreak();
		}
		f.m_P *= pRoulette;
		return f;
	}
private:
	float m_xx = 0.0f;
	float m_a = 0.0f;
	float m_P = 1.0f;
};

class FootprintV2Sq {
	static float ssqrt(float x) { return ei::sgn(x) * sqrt(ei::abs(x)); }
public:
	void init(float sourceArea, float initSolidAngle, float pChoice) {
		m_xx = sourceArea;
		m_aa = initSolidAngle;
		m_P = pChoice;
	}

	float get_area() const {
		return m_xx;
	}

	// eta = ratio of refraction indices n_in / n_out
	FootprintV2Sq add_segment(float pdf, bool orthographic, float mean_curvature,
							  float prevInCos, float prevOutCos, float eta, float distance,
							  float inCos, float pRoulette) const {
		FootprintV2Sq f = *this;
		if(orthographic) {
			f.m_xx = 1.0f / (pdf * ei::abs(inCos));
			//f.m_x = 1.0f / sqrt(pdf);
		} else {
			const float Hs = mean_curvature * sqrt(f.m_xx) * ei::sign(prevInCos);
			if(prevInCos * prevOutCos < 0.0f) {
				// Refraction
			//	f.m_aa = f.m_aa * eta * eta + (Hs * eta - Hs) * ei::abs(Hs * eta - Hs);
				f.m_aa = ssqrt(f.m_aa) * eta + (eta * prevInCos + prevOutCos) * sqrt(m_xx);
			//	const float a = -ei::sign(prevOutCos) * sqrt(m_xx) * mean_curvature;
			//	f.m_aa = (sqrt(f.m_aa) + a) * eta - a;
				f.m_aa *= ei::abs(f.m_aa);
			} else {
				// Reflection
			//	f.m_aa += 4.0f * Hs * ei::abs(Hs);
				f.m_aa = ssqrt(f.m_aa) + 2 * mean_curvature * prevInCos * sqrt(f.m_xx);
			//	f.m_aa = sqrt(f.m_aa) + mean_curvature * sqrt(f.m_xx);
			//	f.m_aa = sqrt(f.m_aa) + 2 * mean_curvature * sqrt(f.m_xx);
				f.m_aa *= ei::abs(f.m_aa);
			}
			//f.m_aa += ei::PI * 2.0f * mean_curvature * mean_curvature * m_xx;
			//f.m_aa += 4.0f * mean_curvature * mean_curvature * m_xx;
			//f.m_aa += 8.0f * mean_curvature * mean_curvature * m_xx;
			//f.m_aa += ei::PI * 4.0f * mean_curvature * mean_curvature * m_xx;
			//f.m_aa += ei::PI * 4.0f * mean_curvature * mean_curvature * m_xx * prevOutCos;
			//f.m_aa += 4.0f * mean_curvature * mean_curvature * m_xx * prevOutCos * prevOutCos;
			//f.m_aa += mean_curvature * ei::abs(mean_curvature) * m_xx * ei::sgn(prevOutCos);
			//f.m_aa += 4.0f * Hs * ei::abs(Hs);
			//if(f.m_aa < 0.0f) f.m_aa = 0.0f;
			// Leave tangent space
			f.m_xx *= ei::abs(prevOutCos);
			//f.m_aa *= ei::abs(prevOutCos);
			// BRDF
			f.m_aa += 1.0f / (pdf + 1e-6f);
		//	f.m_aa += 2*ei::PI / pdf;
		//	f.m_aa = ei::sq(sqrt(f.m_aa) + 1.0f / sqrt(pdf));
			// Travel
			f.m_xx += f.m_aa * distance * distance;
			//float corr = distance * distance / (distance * distance + f.m_xx);
			//float corr = sqrt(distance * distance / (distance * distance + f.m_xx));
			//float corr = sqrt(distance * distance / (distance * distance + f.m_xx)) / ei::abs(prevOutCos);
			//f.m_xx = f.m_xx + f.m_aa * distance * distance * corr;
			//float corr = sqrt(f.m_aa * f.m_xx) * distance;
			//f.m_xx += f.m_aa * distance * distance - corr;
			//f.m_xx += f.m_aa * distance * distance / ei::abs(prevOutCos);
			if(f.m_xx < 0.0f) {
				f.m_xx = -f.m_xx;
				f.m_aa = -f.m_aa;
			}
			// Enter tangent space
			f.m_xx /= ei::abs(inCos) + 1e-6f;
			//f.m_aa /= ei::abs(inCos);
		}
		f.m_P *= pRoulette;
		return f;
	}
private:
	float m_xx = 0.0f;
	float m_aa = 0.0f;
	float m_P = 1.0f;
};


//using Footprint2D = Footprint2DCov;
using Footprint2D = FootprintV0;

}} // namespace mufflon::renderer