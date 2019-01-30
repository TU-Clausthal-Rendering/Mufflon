#include "background.hpp"
#include "texture_sampling.hpp"
#include "core/scene/textures/interface.hpp"

namespace mufflon::scene::lights {

template< Device dev >
const BackgroundDesc<dev> Background::acquire_const(const ei::Box& bounds) {
	switch(m_type) {
		case BackgroundType::COLORED:
			compute_constant_flux(bounds);
			return BackgroundDesc<dev>{ {}, {}, m_type, m_color, m_flux * m_color };
		case BackgroundType::ENVMAP:
			mAssert(m_envLight != nullptr);
			if(!m_summedAreaTable) { // Flux and SAT are not computed?
				m_summedAreaTable = create_summed_area_table(m_envLight);
				m_summedAreaTable->mark_changed(Device::CPU);
				compute_envmap_flux(bounds);
			}
			return BackgroundDesc<dev>{
				m_envLight->acquire_const<dev>(),
				m_summedAreaTable->acquire_const<dev>(),
				m_type, m_color, m_flux * m_color
			};
		default: mAssert(false); return {};
	}
}

template const BackgroundDesc<Device::CPU> Background::acquire_const<Device::CPU>(const ei::Box&);
template const BackgroundDesc<Device::CUDA> Background::acquire_const<Device::CUDA>(const ei::Box&);

Spectrum irradiance_to_flux(const Spectrum& irradiance, const ei::Vec3& aabbDiag, const Direction& direction) {
	float surface = aabbDiag.y*aabbDiag.z*std::abs(direction.x)
				  + aabbDiag.x*aabbDiag.z*std::abs(direction.y)
				  + aabbDiag.x*aabbDiag.y*std::abs(direction.z);
	return irradiance * surface;
}

void Background::compute_envmap_flux(const ei::Box& bounds) {
	// Acquire some domain information
	ei::Vec3 aabbDiag = bounds.max - bounds.min;
	const int width = static_cast<int>(m_envLight->get_width());
	const int height = static_cast<int>(m_envLight->get_height());
	const int layers = static_cast<int>(m_envLight->get_num_layers());
	auto envTex = m_envLight->acquire_const<Device::CPU>();
	const float avgPixelSolidAngle = 4 * ei::PI / (width * height);
	// Integrate over all pixels
	m_flux = Spectrum { 0.0f };
	if(layers == 6) {
		// Cubemap
		for(int l = 0; l < layers; ++l) {
			for(int y = 0; y < height; ++y) {
				for(int x = 0; x < width; ++x) {
					const Spectrum radiance = Spectrum{ textures::read(envTex, Pixel{ x, y }, l) } * m_color;
					// Get the direction and length to the point on the cube
					UvCoordinate uv { (x + 0.5f) / float(width), (y + 0.5f) / float(height) };
					Point cubePos = textures::cubemap_uv_to_surface(uv, l);
					float lsq = lensq(cubePos);
					float length = sqrt(lsq);
					// Given the length we can compute the solid angle
					const Spectrum irradiance = radiance * (avgPixelSolidAngle * lsq * length / 24.0f);
					m_flux += irradiance_to_flux(irradiance, aabbDiag, cubePos / length);
				}
			}
		}
	} else {
		// Polarmap
		for(int y = 0; y < height; ++y) {
			const float sinTheta = sinf(ei::PI * (y + 0.5f) / float(height));
			for(int x = 0; x < width; ++x) {
				const Spectrum radiance = Spectrum{ textures::read(envTex, Pixel{ x, y }) } * m_color;
				// Get the direction
				const float phi = 2 * ei::PI * (x + 0.5f) / float(width);
				const Direction direction { sinTheta * cos(phi),
											sqrt(1.0f - ei::sq(sinTheta)), // cosTheta
											sinTheta * sin(phi) };
				// Given the length we can compute the solid angle factor
				const Spectrum irradiance = radiance * (avgPixelSolidAngle / (2.0f * ei::PI * ei::PI * sinTheta));
				m_flux += irradiance_to_flux(irradiance, aabbDiag, direction);
			}
		}
	}
}

void Background::compute_constant_flux(const ei::Box& bounds) {
	ei::Vec3 aabbDiag = bounds.max - bounds.min;
	// Flux is radiance * solidAngle * area.
	// The radiance gets mulitplied in the descriptor construction
	const float solidAngle = 4 * ei::PI;
	// integrate(integrate(A * abs(cos(theta)) * sin(theta), theta, 0, %pi), phi, 0, 2*%pi);
	// = 2 * π * A
	// with A being a face of the box (abs(cos(theta)) means we integrate both opposite sides).
	const float area = 2 * ei::PI * (aabbDiag.x * aabbDiag.y + aabbDiag.x * aabbDiag.z + aabbDiag.y * aabbDiag.z);
	m_flux = Spectrum { solidAngle * area };
}


} // namespace mufflon::scene::lights
