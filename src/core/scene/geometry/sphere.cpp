#include "sphere.hpp"
#include <cuda_runtime.h>

namespace mufflon::scene::geometry {

Spheres::Spheres() :
	m_attributes(),
	m_sphereData(m_attributes.aquire(m_attributes.add<Sphere>("radius-position"))),
	m_matIndex(m_attributes.aquire(m_attributes.add<MaterialIndex>("materialIdx"))) {
	// Invalidate bounding box
	m_boundingBox.min = {
		std::numeric_limits<float>::max(),
		std::numeric_limits<float>::max(),
		std::numeric_limits<float>::max()
	};
	m_boundingBox.max = {
		std::numeric_limits<float>::min(),
		std::numeric_limits<float>::min(),
		std::numeric_limits<float>::min()
	};
}

Spheres::SphereHandle Spheres::add(const Point& point, float radius) {
	std::size_t newIndex = m_attributes.get_size();
	SphereHandle hdl(newIndex);
	m_attributes.resize(newIndex + 1u);
	auto posRadAccessor = m_sphereData.aquire<>();
	(*posRadAccessor)[newIndex].m_radPos.position = point;
	(*posRadAccessor)[newIndex].m_radPos.radius = radius;
	// Expand bounding box
	m_boundingBox = ei::Box{ m_boundingBox, ei::Box{ei::Sphere{ point, radius }} };
	return hdl;
}

Spheres::SphereHandle Spheres::add(const Point& point, float radius, MaterialIndex idx) {
	SphereHandle hdl = this->add(point, radius);
	(*m_matIndex.aquire<>())[hdl] = idx;
	return hdl;
}

Spheres::BulkReturn Spheres::add_bulk(std::size_t count, std::istream& radPosStream) {
	std::size_t start = m_attributes.get_size();
	SphereHandle hdl(start);
	m_attributes.resize(start + count);
	std::size_t readRadPos = m_sphereData.restore(radPosStream, start, count);
	// Expand bounding box
	const Sphere* radPos = *m_sphereData.aquireConst();
	for(std::size_t i = start; i < start + readRadPos; ++i) {
		m_boundingBox.max = ei::max(radPos[i].m_radPos.position + radPos[i].m_radPos.radius,
									m_boundingBox.max);
		m_boundingBox.min = ei::min(radPos[i].m_radPos.position - radPos[i].m_radPos.radius,
									m_boundingBox.min);
	}
	return { hdl, readRadPos };
}

Spheres::BulkReturn Spheres::add_bulk(std::size_t count, std::istream& radPosStream,
									  const ei::Box& boundingBox) {
	std::size_t start = m_attributes.get_size();
	SphereHandle hdl(start);
	m_attributes.resize(start + count);
	std::size_t readRadPos = m_sphereData.restore(radPosStream, start, count);
	// Expand bounding box
	m_boundingBox = ei::Box(m_boundingBox, boundingBox);
	return { hdl, readRadPos };
}

} // namespace mufflon::scene::geometry