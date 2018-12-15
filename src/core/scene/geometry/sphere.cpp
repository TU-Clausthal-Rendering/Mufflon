#include "sphere.hpp"
#include <cuda_runtime.h>

namespace mufflon::scene::geometry {

Spheres::Spheres() :
	m_attributes(),
	m_spheresHdl(m_attributes.add_attribute<ei::Sphere>("spheres")),
	m_matIndicesHdl(m_attributes.add_attribute<MaterialIndex>("materialIdx")) {
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
	m_spheresHdl(std::move(sphere.m_spheresHdl)),
	m_matIndicesHdl(std::move(sphere.m_matIndicesHdl)),
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
	std::size_t newIndex = m_attributes.get_attribute_elem_count();
	SphereHandle hdl(newIndex);
	m_attributes.resize(newIndex + 1u);
	ei::Sphere* spheres = m_attributes.acquire<Device::CPU, ei::Sphere>(m_spheresHdl);
	spheres[newIndex].center = point;
	spheres[newIndex].radius = radius;
	m_attributes.mark_changed(Device::CPU, m_spheresHdl);
	// Expand bounding box
	m_boundingBox = ei::Box{ m_boundingBox, ei::Box{ei::Sphere{ point, radius }} };
	return hdl;
}

Spheres::SphereHandle Spheres::add(const Point& point, float radius, MaterialIndex idx) {
	SphereHandle hdl = this->add(point, radius);
	m_attributes.acquire<Device::CPU, u16>(m_spheresHdl)[hdl] = idx;
	m_attributes.mark_changed(Device::CPU, m_matIndicesHdl);
	return hdl;
}

Spheres::BulkReturn Spheres::add_bulk(std::size_t count, util::IByteReader& radPosStream) {
	std::size_t start = m_attributes.get_attribute_elem_count();
	SphereHandle hdl(start);
	m_attributes.reserve(start + count);
	std::size_t readRadPos = m_attributes.restore(m_spheresHdl, radPosStream, start, count);
	// Expand bounding box
	const ei::Sphere* radPos = m_attributes.acquire_const<Device::CPU, ei::Sphere>(m_spheresHdl);
	for(std::size_t i = start; i < start + readRadPos; ++i)
		m_boundingBox = ei::Box{ m_boundingBox, ei::Box{radPos[i]} };
	return { hdl, readRadPos };
}

Spheres::BulkReturn Spheres::add_bulk(std::size_t count, util::IByteReader& radPosStream,
									  const ei::Box& boundingBox) {
	std::size_t start = m_attributes.get_attribute_elem_count();
	SphereHandle hdl(start);
	m_attributes.reserve(start + count);
	std::size_t readRadPos = m_attributes.restore(m_spheresHdl, radPosStream, start, count);
	// Expand bounding box
	m_boundingBox = ei::Box(m_boundingBox, boundingBox);
	return { hdl, readRadPos };
}

std::size_t Spheres::add_bulk(std::string_view name, const SphereHandle& startSphere,
							  std::size_t count, util::IByteReader& attrStream) {
	if(startSphere >= m_attributes.get_attribute_elem_count())
		return 0u;
	if(startSphere + count > m_attributes.get_attribute_elem_count())
		m_attributes.reserve(startSphere + count);
	return m_attributes.restore(name, attrStream, startSphere, count);
}

std::size_t Spheres::add_bulk(AttributePool::AttributeHandle hdl, const SphereHandle& startSphere,
							  std::size_t count, util::IByteReader& attrStream) {
	if(startSphere >= m_attributes.get_attribute_elem_count())
		return 0u;
	if(startSphere + count > m_attributes.get_attribute_elem_count())
		m_attributes.reserve(startSphere + count);
	return m_attributes.restore(hdl, attrStream, startSphere, count);
}
} // namespace mufflon::scene::geometry