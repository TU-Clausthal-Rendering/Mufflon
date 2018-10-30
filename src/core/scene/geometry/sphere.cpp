#include "sphere.hpp"
#include <cuda_runtime.h>

namespace mufflon::scene::geometry {

Spheres::SphereHandle Spheres::add(const Point& point, float radius, MaterialIndex idx) {
	std::size_t newIndex = m_attributes.get_size();
	SphereHandle hdl(newIndex);
	m_attributes.resize(newIndex + 1u);
	auto posRadAccessor = m_sphereData.aquire<>();
	auto matIndexAccessor = m_matIndex.aquire<>();
	(*posRadAccessor)[newIndex].m_radPos.position = point;
	(*posRadAccessor)[newIndex].m_radPos.radius = radius;
	(*matIndexAccessor)[newIndex] = idx;
	return hdl;
}

Spheres::BulkReturn Spheres::add_bulk(std::size_t count, std::istream& radPosStream) {
	std::size_t start = m_attributes.get_size();
	SphereHandle hdl(start);
	m_attributes.resize(start + count);
	std::size_t readRadPos = m_sphereData.restore(radPosStream, start, count);
	return {hdl, readRadPos};
}

} // namespace mufflon::scene::geometry