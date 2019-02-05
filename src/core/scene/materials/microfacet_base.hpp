#pragma once

#include "core/export/api.h"
#include "util/types.hpp"
#include "core/math/sampling.hpp"
#include "medium.hpp"
#include <cuda_runtime.h>
#include <ei/vector.hpp>

namespace mufflon { namespace scene { namespace materials {


CUDA_FUNCTION Direction compute_half_vector(const Medium& inMedium, const Medium& exMedium, const Direction& incidentTS, const Direction& excidentTS) {
	// There are two different preconditions: reflection and refraction.
	// For reflection the good old normalize(in + out) is sufficient.
	// For refraction we need normalize(in * n_i + out * n_e) with the
	// refraction indices n. Luckily n_i=n_e for reflection, so we need
	// no case distinction.
	// Also, an additional invariant to rander two-sided materials is
	// that the halfTS has a positiv z axis (aligned with normal).
	Direction halfTS = inMedium.get_refraction_index().x * incidentTS
		             + exMedium.get_refraction_index().x * excidentTS;
	float l = len(halfTS) * ei::sgn(halfTS.z); // Half vector always on side of the normal
	return sdiv(halfTS, l);
}

/*
 * Cached half vector. Not every model needs to compute a half vector,
 * but it is possible that blend/fresnel models require it multiple times.
 * The same models require access to the refraction indices -> usefull to
 * have both at hand
 */
struct Boundary { // Interface would be a good name too, but 'interface' as variable would clash with a keyword
public:
	const Medium& incidentMedium;
	const Medium& otherMedium;
	CUDA_FUNCTION Boundary(const Medium& incidentMedium, const Medium& otherMedium) :
		incidentMedium(incidentMedium), otherMedium(otherMedium)
	{}
	// Get the half vector. If necessary it will be computed.
	CUDA_FUNCTION Direction get_halfTS(const Direction& incidentTS, const Direction& excidentTS) noexcept {
		if(halfIsComputed) return halfTS; // Cache hit.
		const Medium& excidentMedium = incidentTS.z * excidentTS.z > 0.0f ? incidentMedium : otherMedium;
		halfTS = compute_half_vector(incidentMedium, excidentMedium, incidentTS, excidentTS);
		halfIsComputed = true;
		return halfTS;
	}
	// Set the half vector from another source (e.g. from sampling)
	CUDA_FUNCTION void set_halfTS(const Direction& newHalfTS) {
		halfTS = newHalfTS;
		halfIsComputed = true;
	}
private:
	Direction halfTS {0.0f};
	bool halfIsComputed {false};
};



enum class NDF {
	GGX,		// Anisotropic GGX
	BECKMANN,	// Anisotropic Beckmann-Spizzichino
	//COSINE
};

// Geometry term (Selfshadow) for V-cavity model.
CUDA_FUNCTION __forceinline__ float geoshadowing_vcavity(float wDotH, float wDotN, float hDotN, const ei::Vec2 & _roughness)
{
	// For refraction the half vector can be the other way around than the surface. I.e. the
	// microfacet would be seen from back-side, which is impossible.
	if(wDotH * wDotN <= 0.0) return 0.0;
	// V-Cavity/Cook-Torrance
	return ei::min(1.0f, 2.0f * ei::abs(sdiv(hDotN * wDotN, wDotH)));
	// Smith for GGX
	//return sdiv(2.0f, 1.0f + sqrt(1.0f + _roughness.x*_roughness.y * (sdiv(1.0f, wDotN*wDotN) - 1.0f)));
}

namespace detail {
	// Johannes' personal hack for appearance of multibounce model (compensates energy loss)
	CUDA_FUNCTION __forceinline__ float modify_G_multi_bounce(float _g)
	{
		return _g < 0.11f ? powf(_g+0.74299714456847f, 8.0f) - 0.092874643071059f
								: ((1.139400367544197f * _g - 3.317658394008854f) * _g + 3.317115685385115f) * _g - 0.13885765892046f;//*/
		//return _g;
	}
}

// V-cavity geometry term for refraction (transmitted)
CUDA_FUNCTION __forceinline__ float geoshadowing_vcavity_transmission(float Gi, float Ge)
{
	//return Gi * Ge; // Smith
	return ei::max(0.0f, detail::modify_G_multi_bounce(Gi) + detail::modify_G_multi_bounce(Ge) - 1.0f);
}

// V-cavity geometry term for reflection
CUDA_FUNCTION __forceinline__ float geoshadowing_vcavity_reflection(float Gi, float Ge)
{
	//return Gi * Ge; // Smith
	return detail::modify_G_multi_bounce( ei::min(Gi, Ge) );
}

// Get a normal of the cavity proportional to their visibility.
// Also returns iDotH for the new half vector.
CUDA_FUNCTION __forceinline__ Direction sample_visible_normal_vcavity(const Direction& incidentTS, const Direction& cavityTS, u64 rndChoice, float& iDotHOut) {
	// Transform to tangent space and produce V-cavity adjoint normal.
	// See "Importance Sampling Microfacet-Based BSDFs using the Distribution of Visible Normals".
	Direction adjHalf { -cavityTS.x, -cavityTS.y, cavityTS.z };
	iDotHOut = dot(incidentTS, cavityTS);
	float iDotHadj = dot(incidentTS, adjHalf);
	// Choose one of the normals randomly
	float t = incidentTS.z > 0.0f ? ei::max(0.0f, iDotHadj) / (ei::max(0.0f, iDotHOut) + ei::max(0.0f, iDotHadj))
								  : ei::max(0.0f,-iDotHadj) / (ei::max(0.0f,-iDotHOut) + ei::max(0.0f,-iDotHadj));
	constexpr float MAX_RND = float(std::numeric_limits<u64>::max()); // TODO: successor
	float rnd = rndChoice / MAX_RND;
	if(rnd < t) {
		// Use adjoint half vector instead
		iDotHOut = iDotHadj;
		return adjHalf;
	} else
		return cavityTS;
}



// Evaluate any of the supported ndfs
CUDA_FUNCTION float eval_ndf(NDF ndf, ei::Vec2 roughness, const Direction& halfTS) {
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

CUDA_FUNCTION math::DirectionSample sample_ndf(NDF ndf, ei::Vec2 roughness, const math::RndSet2& rnd) {
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
