#include "sphere.hpp"

namespace mufflon::scene {

void SphereArray::push_back(const Sphere &sphere) {
	m_positions.push_back(sphere.position);
	m_radii.push_back(sphere.radius);
	m_matIndices.push_back(sphere.matIndex);
}

} // namespace mufflon::scene