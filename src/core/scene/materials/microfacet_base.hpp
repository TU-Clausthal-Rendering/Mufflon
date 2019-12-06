#pragma once

#include "core/export/core_api.h"
#include "util/types.hpp"
#include "core/math/sampling.hpp"
#include "core/math/functions.hpp"
#include "medium.hpp"
#include <cuda_runtime.h>
#include <ei/vector.hpp>
#include <math.h>

namespace mufflon { namespace scene { namespace materials {


inline CUDA_FUNCTION Direction compute_half_vector(const Medium& inMedium, const Medium& exMedium, const Direction& incidentTS, const Direction& excidentTS) {
	// There are two different preconditions: reflection and refraction.
	// For reflection the good old normalize(in + out) is sufficient.
	// For refraction we need normalize(in * n_i + out * n_e) with the
	// refraction indices n. Luckily n_i=n_e for reflection, so we need
	// no case distinction.
	// Also, an additional invariant to render two-sided materials is
	// that the halfTS has a positive z axis (aligned with normal).
	Direction halfTS = inMedium.get_refraction_index().x * incidentTS
		             + exMedium.get_refraction_index().x * excidentTS;
	float l = len(halfTS) * ei::sgn(halfTS.z); // Half vector always on side of the normal
	return sdiv(halfTS, l);
}

/*
 * Cached half vector. Not every model needs to compute a half vector,
 * but it is possible that blend/fresnel models require it multiple times.
 * The same models require access to the refraction indices -> useful to
 * have both at hand
 */
struct Boundary { // Interface would be a good name too, but 'interface' as variable would clash with a keyword
public:
	const Medium& incidentMedium;
	const Medium& otherMedium;
	inline CUDA_FUNCTION Boundary(const Medium& incidentMedium, const Medium& otherMedium) :
		incidentMedium(incidentMedium), otherMedium(otherMedium)
	{}
	// Get the half vector. If necessary it will be computed.
	inline CUDA_FUNCTION Direction get_halfTS(const Direction& incidentTS, const Direction& excidentTS) noexcept {
		if(halfIsComputed) return halfTS; // Cache hit.
		const Medium& excidentMedium = incidentTS.z * excidentTS.z > 0.0f ? incidentMedium : otherMedium;
		halfTS = compute_half_vector(incidentMedium, excidentMedium, incidentTS, excidentTS);
		halfIsComputed = true;
		return halfTS;
	}
	// Set the half vector from another source (e.g. from sampling)
	inline CUDA_FUNCTION void set_halfTS(const Direction& newHalfTS) {
		halfTS = newHalfTS;
		halfIsComputed = true;
	}
private:
	Direction halfTS {0.0f};
	bool halfIsComputed {false};
};



enum class NDF : i16 {
	GGX,		// Anisotropic GGX
	BECKMANN,	// Anisotropic Beckmann-Spizzichino
	//COSINE
};

enum class ShadowingModel : i16 {
	VCAVITY,
	SMITH
};

// Geometry term (Selfshadow) for V-cavity model.
CUDA_FUNCTION __forceinline__ float geoshadowing_vcavity(float wDotH, float wDotN, float hDotN, const ei::Vec2 & /*roughness*/) {
	// For refraction the half vector can be the other way around than the surface. I.e. the
	// microfacet would be seen from back-side, which is impossible.
	if(wDotH * wDotN <= 0.0) return 0.0;
	// V-Cavity/Cook-Torrance
	return ei::min(1.0f, 2.0f * ei::abs(sdiv(hDotN * wDotN, wDotH)));
	// Smith for GGX
	//return sdiv(2.0f, 1.0f + sqrt(1.0f + _roughness.x*_roughness.y * (sdiv(1.0f, wDotN*wDotN) - 1.0f)));
}

// Geometry term (Selfshadow) for Smith model.
CUDA_FUNCTION __forceinline__ float geoshadowing_smith(float wDotH, const ei::Vec3& w, const ei::Vec2& roughness, NDF ndf) {
	// For refraction the half vector can be the other way around than the surface. I.e. the
	// microfacet would be seen from back-side, which is impossible.
	if(wDotH * w.z <= 0.0) return 0.0;
	switch (ndf) {
		case NDF::GGX: {
			// H14Understanding P.86
			// alpha0 = sqrt(x²alphax²/sqrt(1-z²)²+y²alphay²/sqrt(1-z²)²)
			// = 1/sqrt(1-z²)*sqrt(x²alphax²+y²alphay²)
			// tan(theta) = sin(theta)/cos(theta) = sqrt(1-z²)/z
			// a = 1/(alpha0*tan(theta))
			// = sqrt(x²alphax²+y²alphay²)/z
			// 1/a² = sqrt(x²alphax²+y²alphay²)/z²
			float oneOverASquared = sdiv(ei::sq(w.x * roughness.x) + ei::sq(w.y * roughness.y), w.z * w.z);
			// G = 1/(1+(-1+sqrt(1+1/a²))/2)
			// = 2/(1+sqrt(1+1/a²))
			return 2.0f / (1.0f + sqrt(1.f + oneOverASquared));
		}
		case NDF::BECKMANN: {
			// http://graphicrants.blogspot.com/2013/08/specular-brdf-reference.html
			float c = sdiv(w.z, sqrt(ei::sq(w.x * roughness.x) + ei::sq(w.y * roughness.y)));
			c = ei::abs(c);
			return c >= 1.6f ? 1.0f : (3.535f + 2.181f*c) / (1.0f/c + 2.276f + 2.577f*c);
		}
	}
	mAssert(false);
	return 1.f;
}

namespace detail {
	// Johannes' personal hack for appearance of multibounce model (compensates energy loss)
	CUDA_FUNCTION __forceinline__ float modify_G_multi_bounce(float g)
	{
		return g < 0.11f ? powf(g+0.74299714456847f, 8.0f) - 0.092874643071059f
								: ((1.139400367544197f * g - 3.317658394008854f) * g + 3.317115685385115f) * g - 0.13885765892046f;//*/
		//return g;
	}
}

// V-cavity geometry term for refraction (transmitted)
CUDA_FUNCTION __forceinline__ float geoshadowing_vcavity_transmission(float Gi, float Ge)
{
	//return Gi * Ge; // Smith
	return ei::max(0.0f, detail::modify_G_multi_bounce(Gi) + detail::modify_G_multi_bounce(Ge) - 1.0f);
}

// Smith geometry term for refraction (transmitted)
CUDA_FUNCTION __forceinline__ float geoshadowing_smith_transmission(float Gi, float Ge) {
	return Gi * Ge; // Smith
}

// V-cavity geometry term for reflection
CUDA_FUNCTION __forceinline__ float geoshadowing_vcavity_reflection(float Gi, float Ge)
{
	//return Gi * Ge; // Smith
	return detail::modify_G_multi_bounce( ei::min(Gi, Ge) );
}

// Smith geometry term for reflection
CUDA_FUNCTION __forceinline__ float geoshadowing_smith_reflection(float Gi, float Ge) {
	return Gi * Ge; // Smith
}

// Get a normal of the cavity proportional to their visibility.
// Also returns iDotH for the new half vector.
struct VCavitySampleResult { Direction halfTS; float cosI; };

CUDA_FUNCTION __forceinline__ VCavitySampleResult
sample_visible_normal_vcavity(const Direction& incidentTS, const Direction& cavityTS, u64& rndChoice) {
	// Transform to tangent space and produce V-cavity adjoint normal.
	// See "Importance Sampling Microfacet-Based BSDFs using the Distribution of Visible Normals".
	Direction adjHalf { -cavityTS.x, -cavityTS.y, cavityTS.z };
	float iDotHOut = dot(incidentTS, cavityTS);
	float iDotHadj = dot(incidentTS, adjHalf);
	// Choose one of the normals randomly
	float t = incidentTS.z > 0.0f ? ei::max(0.0f, iDotHadj) / (ei::max(0.0f, iDotHOut) + ei::max(0.0f, iDotHadj))
								  : ei::max(0.0f,-iDotHadj) / (ei::max(0.0f,-iDotHOut) + ei::max(0.0f,-iDotHadj));
	u64 probAdj = math::percentage_of(std::numeric_limits<u64>::max(), t);
	if(rndChoice < probAdj) {
		rndChoice = math::rescale_sample(rndChoice, 0, probAdj-1);
		// Use adjoint half vector instead
		iDotHOut = iDotHadj;
		return { adjHalf, iDotHadj };
	} else {
		rndChoice = math::rescale_sample(rndChoice, probAdj, std::numeric_limits<u64>::max());
		return { cavityTS, iDotHOut };
	}
}

// Get the sampled slopes for the smith model with beckmann distribution
inline CUDA_FUNCTION ei::Vec2 sample_visible_slopes_beckmann(float sinTheta, float cosTheta, float tanTheta, const ei::Vec2& /*roughness*/, const math::RndSet2& rnd, u64& rnd2) {
	// Special case (normal incidence)
	if(tanTheta < 0.0001) {
		const float r = sqrt(-log(rnd.u0));
		const float phi = 6.28318530718f * rnd.u1;
		const float slopeX = r * cosf(phi);
		const float slopeY = r * sinf(phi);
		return {slopeX, slopeY};
	}
	// precomputations
	const float a = 1.0f / tanTheta;
	const float erfA = erf(a);
	const float expA2 = exp(-a * a);
	const float SQRT_PI_INV = 0.56418958354f;
	const float lambda = 0.5f*(erfA - 1.f) + 0.5f*SQRT_PI_INV*expA2 / a;
	const float g1 = sdiv(1.0f , 1.0f + lambda); // masking
	const float c = 1.0f - g1 * erfA;

	// sample slope X
	float slopeX;
	u64 ci = math::percentage_of(std::numeric_limits<u64>::max(), c);
	if(rnd2 < ci) {
		const float w1 = 0.5f * SQRT_PI_INV * sinTheta * expA2;
		const float w2 = cosTheta * (0.5f - 0.5f*erfA);
		const float p = w1 / (w1 + w2);

		u64 pi = math::percentage_of(ci, p);
		if(rnd2 < pi) {
			rnd2 = math::rescale_sample(rnd2, 0, pi - 1);
			slopeX = -sqrt(-log(rnd.u0*expA2));
		} else {
			rnd2 = math::rescale_sample(rnd2, pi, ci - 1);
			slopeX = math::erfInv(rnd.u0 - 1.0f - rnd.u0 * erfA);
		}
	} else {
		slopeX = math::erfInv((-1.0f + 2.0f*rnd.u0)*erfA);
		const float p = (-slopeX * sinTheta + cosTheta) / (2.0f*cosTheta);
		u64 pi = ci + math::percentage_of(std::numeric_limits<u64>::max() - ci, p);
		if(rnd2 < pi) 
			rnd2 = math::rescale_sample(rnd2, ci, pi - 1);
		else {
			rnd2 = math::rescale_sample(rnd2, pi, std::numeric_limits<u64>::max());
			slopeX = -slopeX;
		}
	}
	// sample slope Y
	const float slopeY = math::erfInv(ei::clamp(2.0f*rnd.u1 - 1.0f, -0.99999f, 0.99999f));
	mAssert(!isnan(slopeX) && !isnan(slopeY));
	return { slopeX, slopeY };
}

// Get the sampled slopes for the smith model with ggx distribution
inline CUDA_FUNCTION ei::Vec2 sample_visible_slopes_ggx(float tanTheta, const math::RndSet2& rnd, u64& rnd2) {
	mAssert(tanTheta >= 0.0f);
	// Special case (normal incidence)
	if(tanTheta < 0.0001) {
		const float r = sqrtf(sdiv(rnd.u0, (1.f-rnd.u0)));
		const float phi = 6.28318530718f * rnd.u1;
		const float slopeX = r * cosf(phi);
		const float slopeY = r * sinf(phi);
		return { slopeX, slopeY };
	}
	const float a1 = 1.0f / tanTheta;
	const float g1 = 2.f / (1.f + sqrtf(1.0f + 1.0f / (a1 * a1)));
	// sample slope_x
	const float a = 2.0f*rnd.u0 / g1 - 1.0f;
	const float tmp = sdiv(1.0f , a * a - 1.0f);
	const float d = tmp * sqrtf(tanTheta * tanTheta - (a * a - tanTheta * tanTheta) / tmp);
	const float slopeX1 = tanTheta * tmp - d;
	const float slopeX2 = tanTheta * tmp + d;
	const float slopeX = (a < 0.f || slopeX2 > 1.0f / tanTheta) ? slopeX1 : slopeX2;
	// sample slope_y
	float s;
	if(rnd2 & (1ull << 63)) {
		rnd2 *= 2; 
		s = 1.0f;
	} else {
		rnd2 *= 2;
		s = -1.0f;
	}
	const float z = sdiv((rnd.u1 * (rnd.u1 * (rnd.u1 * 0.27385f - 0.73369f) + 0.46341f)), 
		(rnd.u1 * (rnd.u1 * (rnd.u1 * 0.093073f + 0.309420f) - 1.000000f) + 0.597999f));
	const float slopeY = s * z * sqrtf(1.0f + slopeX * slopeX);
	return { slopeX, slopeY };
}

// Get a normal of the cavity proportional to their visibility.
inline CUDA_FUNCTION Direction sample_visible_normal_smith(const NDF ndf,const Direction& incidentTS, const ei::Vec2& roughness, const math::RndSet2& rnd, u64& rnd2) {
	// Stretch incidentTS
	Direction inStretched = incidentTS * ei::Vec3{ roughness , 1.0f };
	inStretched = normalize(inStretched);
	if(incidentTS.z < 0.0f)
		inStretched = -inStretched;

	const float sinTheta = sqrtf((1.0f - inStretched.z) * (1.0f + inStretched.z));
	const float tanTheta = sinTheta / inStretched.z;
	const float cosPhi = sinTheta > 1e-5f ? inStretched.x / sinTheta : 0.0f;
	const float sinPhi = sinTheta > 1e-5f ? inStretched.y / sinTheta : 0.0f;

	// sample
	ei::Vec2 slopes{ 0.f };
	switch (ndf) {
		case NDF::BECKMANN: {
			slopes = sample_visible_slopes_beckmann(sinTheta, inStretched.z, tanTheta, roughness, rnd, rnd2);
		} break;
		case NDF::GGX: {
			slopes = sample_visible_slopes_ggx(tanTheta, rnd, rnd2);
		} break;
	}
	// rotate
	float tmp = cosPhi * slopes.x - sinPhi * slopes.y;
	slopes.y = sinPhi * slopes.x + cosPhi * slopes.y;
	slopes.x = tmp;
	// Unstrech
	slopes.x *= roughness.x;
	slopes.y *= roughness.y;
	// Compute normal
	return normalize(Direction{-slopes.x, -slopes.y, 1.0f});
}

// Evaluate any of the supported ndfs
inline CUDA_FUNCTION float eval_ndf(NDF ndf, ei::Vec2 roughness, const Direction& halfTS) {
	switch(ndf) {
		case NDF::GGX: {
			// Anisotropic GGX: http://graphicrants.blogspot.de/2013/08/specular-brdf-reference.html,
			// https://hal.inria.fr/hal-00942452v1/document "Understanding the Masking-Shadowing Function in Microfacet-Based BRDFs"
			// 1/(pi α_x α_y) * 1/((x⋅h/α_x)² + (y⋅h/α_y)² + (n⋅h)²)²
			float norm = ei::max(ei::PI * roughness.x * roughness.y, 1e-30f);
			float cosThetaSq = halfTS.z * halfTS.z;
			float rx = sdiv(halfTS.x, roughness.x);
			float ry = sdiv(halfTS.y, roughness.y);
			float tmp = rx * rx + ry * ry + cosThetaSq;
			return sdiv(1.0f, norm * tmp * tmp);
		}
		case NDF::BECKMANN: {
			// 1/(π α_x α_y (n⋅h)⁴) exp(((n⋅h)²-1) / (n⋅h)² * ((x⋅h/α_x sin(θh))² + (y⋅h/α_y sin(θh))²))
			// = 1/(π α_x α_y (n⋅h)⁴) exp( -((x⋅h/α_x)² + (y⋅h/α_y)²) / (n⋅h)² )
			float norm = ei::PI * roughness.x * roughness.y;
			float rx = sdiv(halfTS.x, roughness.x);
			float ry = sdiv(halfTS.y, roughness.y);
			float nDotHSq = halfTS.z * halfTS.z;
			float e = exp(-(rx*rx + ry*ry) / nDotHSq);
			return sdiv(e, norm * nDotHSq * nDotHSq);
		}
	}
	return 0.0f;
}

inline CUDA_FUNCTION math::DirectionSample sample_ndf(NDF ndf, ei::Vec2 roughness, const math::RndSet2& rnd) {
	switch(ndf) {
		case NDF::GGX: {
			float phi = 2.0f * ei::PI * rnd.u0;
			float e = sqrt(rnd.u1 / (1.0f - rnd.u1));
			float slopeX = e * cos(phi); // Partially slope (missing roughness)
			float slopeY = e * sin(phi);

			float norm = ei::PI * roughness.x * roughness.y;
			float tmp = 1.0f + slopeX * slopeX + slopeY * slopeY;
			// PDF of slopes is 1 / (norm * tmp * tmp)

			slopeX *= roughness.x; // Complete slopes
			slopeY *= roughness.y;
			Direction dir = normalize(ei::Vec3(-slopeX, -slopeY, 1.0f));

			// Transform the PDF of slopes into a PDF of normals by the Jacobian
			// 1 / dot(dir, normal)^3. Here, the normal is (0,0,1).
			AngularPdf pdf { 1.0f / ei::max(norm * tmp * tmp * dir.z * dir.z * dir.z, 1e-20f) };

			return math::DirectionSample{dir, pdf};
		}
		case NDF::BECKMANN: {
			// Using slope based sampling (Heitz 2014 Importance Sampling Microfacet-Based BSDFs
			// Using the Distribution of Visible Normals, Supplemental 2).
			// The exponential in the Beck. distr. is sampled using the Box-Muller transform.
			float phi = 2 * ei::PI * rnd.u0;
			float xi = rnd.u1 + 1e-20f;
			float e = sqrt(-log(xi));
			float slopeX = e * cos(phi);
			float slopeY = e * sin(phi);
			Direction dir = normalize(ei::Vec3(-roughness.x * slopeX, -roughness.y * slopeY, 1.0f));

			// PDF = 1/(π α_x α_y) exp(-(sX/α_x)²-(s/α_y)²) / (n⋅h)³
			//     = 1/(π α_x α_y) exp(-(slopeX² + slopeY²)) / (n⋅h)³
			//     = 1/(π α_x α_y) exp(-e²) / (n⋅h)³
			//     = 1/(π α_x α_y) xi / (n⋅h)³
			// The / (n⋅h)³ is from the Jacobian slope space -> normalized direction
			AngularPdf pdf { xi / ei::max(ei::PI * roughness.x * roughness.y * dir.z * dir.z * dir.z, 1e-20f) };

			return math::DirectionSample{dir, pdf};
		}
		default:
			mAssertMsg(false, "NDF not supported.");
	}
	return math::DirectionSample{};
}

}}} // namespace mufflon::scene::materials
