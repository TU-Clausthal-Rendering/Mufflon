#include "sphere.hpp"
#include <cuda_runtime.h>

namespace mufflon::scene::geometry {

Spheres::Spheres() :
	m_attributes(),
	m_sphereData(m_attributes.add<ei::Sphere>("spheres")),
	m_matIndex(m_attributes.add<MaterialIndex>("materialIdx")) {
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

Spheres::Spheres(Spheres&& sphere) :
	m_attributes(std::move(sphere.m_attributes)),
	m_sphereData(std::move(sphere.m_sphereData)),
	m_matIndex(std::move(sphere.m_matIndex)),
	m_boundingBox(std::move(sphere.m_boundingBox)){
	sphere.m_attribBuffer.for_each([&](auto& buffer) {
		using ChangedBuffer = std::decay_t<decltype(buffer)>;
		m_attribBuffer.get<ChangedBuffer>() = buffer;
		buffer.size = 0u;
		buffer.buffer = ArrayDevHandle_t<ChangedBuffer::DEVICE, ArrayDevHandle_t<ChangedBuffer::DEVICE, void>>{};
	});
}

Spheres::~Spheres() {
	m_attribBuffer.for_each([&](auto& buffer) {
		using ChangedBuffer = std::decay_t<decltype(buffer)>;
		if(buffer.size != 0)
			Allocator<ChangedBuffer::DEVICE>::free(buffer.buffer, buffer.size);
	});
}

Spheres::SphereHandle Spheres::add(const Point& point, float radius) {
	std::size_t newIndex = m_attributes.get_size();
	SphereHandle hdl(newIndex);
	m_attributes.resize(newIndex + 1u);
	auto posRadAccessor = get_spheres().aquire<>();
	(*posRadAccessor)[newIndex].center = point;
	(*posRadAccessor)[newIndex].radius = radius;
	// Expand bounding box
	m_boundingBox = ei::Box{ m_boundingBox, ei::Box{ei::Sphere{ point, radius }} };
	return hdl;
}

Spheres::SphereHandle Spheres::add(const Point& point, float radius, MaterialIndex idx) {
	SphereHandle hdl = this->add(point, radius);
	(*get_mat_indices().aquire<>())[hdl] = idx;
	return hdl;
}

Spheres::BulkReturn Spheres::add_bulk(std::size_t count, util::IByteReader& radPosStream) {
	std::size_t start = m_attributes.get_size();
	SphereHandle hdl(start);
	m_attributes.resize(start + count);
	auto& spheres = get_spheres();
	std::size_t readRadPos = spheres.restore(radPosStream, start, count);
	// Expand bounding box
	const ei::Sphere* radPos = *spheres.aquireConst();
	for(std::size_t i = start; i < start + readRadPos; ++i)
		m_boundingBox = ei::Box{ m_boundingBox, ei::Box{radPos[i]} };
	return { hdl, readRadPos };
}

Spheres::BulkReturn Spheres::add_bulk(std::size_t count, util::IByteReader& radPosStream,
									  const ei::Box& boundingBox) {
	std::size_t start = m_attributes.get_size();
	SphereHandle hdl(start);
	m_attributes.resize(start + count);
	auto& spheres = get_spheres();
	std::size_t readRadPos = spheres.restore(radPosStream, start, count);
	// Expand bounding box
	m_boundingBox = ei::Box(m_boundingBox, boundingBox);
	return { hdl, readRadPos };
}

} // namespace mufflon::scene::geometry