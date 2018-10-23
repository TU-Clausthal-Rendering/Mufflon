#include "sphere.hpp"

namespace mufflon::scene::geometry {

Spheres::SphereHandle Spheres::add(const Point& point, float radius, MaterialIndex idx) {
	SphereHandle hdl(m_sphereData.size());
	m_attributes.resize(m_sphereData.size() + 1u);
	m_sphereData.back().m_radPos.position = point;
	m_sphereData.back().m_radPos.radius = radius;
	m_matIndex.back() = idx;
	return hdl;
}

Spheres::BulkReturn Spheres::add_bulk(std::size_t count, std::istream& radPosStream) {
	SphereHandle hdl(m_sphereData.size());
	// Resize all attributes to fit the number of spheres we want to add
	this->resize(m_sphereData.size() + count);
	// Read the radii and positions
	radPosStream.read(reinterpret_cast<char*>(m_sphereData.data()), 
					  sizeof(Sphere) * count);
	std::size_t readRadPos = static_cast<std::size_t>(radPosStream.gcount()) / sizeof(Sphere);

	return {hdl, readRadPos};
}


} // namespace mufflon::scene::geometry