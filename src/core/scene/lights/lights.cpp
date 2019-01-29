#include "lights.hpp"
#include "core/math/rng.hpp"
#include "core/math/sampling.hpp"
#include "core/scene/textures/interface.hpp"

namespace mufflon::scene::lights {

Spectrum get_flux(const PointLight& light) {
	return light.intensity * 4.f * ei::PI;
}


Spectrum get_flux(const SpotLight& light) {
	// Flux for the PBRT spotlight version with falloff ((cos(t)-cos(t_w))/(cos(t_f)-cos(t_w)))^4
	float cosFalloff = __half2float(light.cosFalloffStart);
	float cosWidth = __half2float(light.cosThetaMax);
	return light.intensity * (2.f * ei::PI * (1.f - cosFalloff) + (cosFalloff - cosWidth) / 5.f);
}


Spectrum get_flux(const AreaLightTriangle<Device::CPU>& light) {
	float area = len(cross(light.posV[1u], light.posV[2u])) / 2.0f;
	// Sample the radiance over the entire triangle region.
	math::GoldenRatio2D gen(*reinterpret_cast<u32*>(&area));	// Use the area as seed
	Spectrum radianceSum{ 0.0f };
	for(int i = 0; i < 128; ++i) {// TODO: adaptive sample count?
		const ei::Vec2 u = math::sample_uniform(gen.next());
		const ei::Vec2 bary = math::sample_barycentric(u.x, u.y);
		const UvCoordinate uv = light.uvV[0u] + light.uvV[1u] * bary.x + light.uvV[2u] * bary.y;
		radianceSum += Spectrum{sample(light.radianceTex, uv)};
	}
	radianceSum /= 128;
	return radianceSum * area * 2 * ei::PI;
}


Spectrum get_flux(const AreaLightQuad<Device::CPU>& light) {
	float area = light.compute_area();
	// Sample the radiance over the entire triangle region.
	math::GoldenRatio2D gen(*reinterpret_cast<u32*>(&area));	// Use the area as seed
	Spectrum radianceSum{ 0.0f };
	for(int i = 0; i < 128; ++i) {// TODO: adaptive sample count?
		const ei::Vec2 u = math::sample_uniform(gen.next());
		const UvCoordinate uv = light.uvV[0u] + light.uvV[1u] * u.y + light.uvV[2u] * u.x + light.uvV[3u] * (u.x * u.y);
		radianceSum += Spectrum{sample(light.radianceTex, uv)};
	}
	radianceSum /= 128;
	return radianceSum * area * 2 * ei::PI; // TODO: put area inside the loop and use the local area/density (cross product of tangents at the point)
}


Spectrum get_flux(const AreaLightSphere<Device::CPU>& light) {
	float area = ei::surface(ei::Sphere{ light.position, light.radius });
	// Sample the radiance over the entire triangle region.
	math::GoldenRatio2D gen(*reinterpret_cast<u32*>(&area));	// Use the area as seed
	Spectrum radianceSum{ 0.0f };
	for(int i = 0; i < 128; ++i) {// TODO: adaptive sample count?
		const ei::Vec2 u = math::sample_uniform(gen.next());
		// Get lon-lat, but in domain [0,1]
		float theta = acos(u.x * 2.0f - 1.0f) / ei::PI;
		float phi = u.y;
		radianceSum += Spectrum{sample(light.radianceTex, UvCoordinate{phi, theta})};
	}
	radianceSum /= 128;
	return radianceSum * area * 2 * ei::PI;
}


Spectrum get_flux(const DirectionalLight& light,
				  const ei::Vec3& aabbDiag) {
	mAssert(aabbDiag.x > 0 && aabbDiag.y > 0 && aabbDiag.z > 0);
	float surface = aabbDiag.y*aabbDiag.z*std::abs(light.direction.x)
		+ aabbDiag.x*aabbDiag.z*std::abs(light.direction.y)
		+ aabbDiag.x*aabbDiag.y*std::abs(light.direction.z);
	return light.irradiance * surface;
}


template < Device dev >
float AreaLightQuad<dev>::compute_area() const {
	/*const ei::Vec3 e03 = points[3] - points[0];
	const ei::Vec3 e12 = points[2] - points[1];
	const ei::Vec3 e01 = points[1] - points[0];
	const ei::Vec3 e32 = points[2] - points[3];
	// Uniform
	constexpr int SAMPLES_PER_DIM = 32;
	float area = 0;
	for(int i = 0; i <= SAMPLES_PER_DIM; ++i) {
		float s = i / float(SAMPLES_PER_DIM);
		for(int j = 0; j <= SAMPLES_PER_DIM; ++j) {
			float t = j / float(SAMPLES_PER_DIM);
			ei::Vec3 tangentX = ei::lerp(e03, e12, t);
			ei::Vec3 tangentY = ei::lerp(e01, e32, s);
			area += len(cross(tangentX, tangentY));
		}
	}
	area /= ei::sq(SAMPLES_PER_DIM+1);*/

	/* Adaptive, quasi Monte Carlo */
	// Get area of two opposing triangles.
	const ei::Vec3 n0 = cross(posV[1u], posV[2u]);
	const ei::Vec3 n1 = cross(posV[1u] + posV[3u], posV[2u] + posV[3u]);
	const float len0 = len(n0); // = tri area*2
	const float len1 = len(n1); // = tri area*2
	float area = (len0 + len1) / 2.0f;
	// This is the exact area, if the quad is planar.
	if(dot(n0, n1) / (len0 * len1) < 0.99999f) {
		// Get more samples for numeric integration
		int count = 2;
		float diff = 0;
		math::GoldenRatio2D seq(235);
		do {
			math::RndSet2 st( seq.next() );
			const ei::Vec3 tangentX = posV[1u] + st.u0 * posV[3u];
			const ei::Vec3 tangentY = posV[2u] + st.u1 * posV[3u];
			const float sample = len(cross(tangentX, tangentY));
			diff = sample - area;
			area += diff / (++count);
		} while(ei::abs(diff / area) > 1e-2f);
	}
	return area;
}
template struct AreaLightQuad<Device::CPU>;
template struct AreaLightQuad<Device::CUDA>;

} // mufflon::scene::lights