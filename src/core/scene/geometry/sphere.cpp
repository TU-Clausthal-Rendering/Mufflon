#include "sphere.hpp"
#include "core/scene/descriptors.hpp"
#include <cuda_runtime.h>
#include "util/punning.hpp"

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
		-std::numeric_limits<float>::max(),
		-std::numeric_limits<float>::max(),
		-std::numeric_limits<float>::max()
	};
}

Spheres::Spheres(const Spheres& sphere) :
	m_attributes(sphere.m_attributes),
	m_spheresHdl(sphere.m_spheresHdl),
	m_matIndicesHdl(sphere.m_matIndicesHdl),
	m_boundingBox(sphere.m_boundingBox),
	m_uniqueMaterials(sphere.m_uniqueMaterials)
{
	sphere.m_attribBuffer.for_each([&](auto& buffer) {
		using ChangedBuffer = std::decay_t<decltype(buffer)>;
		auto& attribBuffer = m_attribBuffer.template get<ChangedBuffer>();
		attribBuffer.size = buffer.size;
		if(buffer.size == 0u || buffer.buffer == ArrayDevHandle_t<ChangedBuffer::DEVICE, ArrayDevHandle_t<ChangedBuffer::DEVICE, void>>{}) {
			attribBuffer.buffer = ArrayDevHandle_t<ChangedBuffer::DEVICE, ArrayDevHandle_t<ChangedBuffer::DEVICE, void>>{};
		} else {
			attribBuffer.buffer = Allocator<ChangedBuffer::DEVICE>::template alloc_array<ArrayDevHandle_t<ChangedBuffer::DEVICE, void>>(buffer.size);
			copy<ArrayDevHandle_t<ChangedBuffer::DEVICE, void>>(attribBuffer.buffer, buffer.buffer, 0, sizeof(ArrayDevHandle_t<ChangedBuffer::DEVICE, void>) * buffer.size);
		}
	});
}

Spheres::Spheres(Spheres&& sphere) :
	m_attributes(std::move(sphere.m_attributes)),
	m_spheresHdl(std::move(sphere.m_spheresHdl)),
	m_matIndicesHdl(std::move(sphere.m_matIndicesHdl)),
	m_boundingBox(std::move(sphere.m_boundingBox)),
	m_uniqueMaterials(std::move(sphere.m_uniqueMaterials))
{
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
			Allocator<ChangedBuffer::DEVICE>::template free<ArrayDevHandle_t<ChangedBuffer::DEVICE, void>>(buffer.buffer, buffer.size);
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
	m_attributes.acquire<Device::CPU, MaterialIndex>(m_matIndicesHdl)[hdl] = idx;
	m_attributes.mark_changed(Device::CPU, m_matIndicesHdl);
	m_uniqueMaterials.emplace(idx);
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

std::size_t Spheres::add_bulk(StringView name, const SphereHandle& startSphere,
							  std::size_t count, util::IByteReader& attrStream) {
	return this->add_bulk(m_attributes.get_attribute_handle(name), startSphere,
						  count, attrStream);
}

std::size_t Spheres::add_bulk(SphereAttributeHandle hdl, const SphereHandle& startSphere,
							  std::size_t count, util::IByteReader& attrStream) {
	if(startSphere >= m_attributes.get_attribute_elem_count())
		return 0u;
	if(startSphere + count > m_attributes.get_attribute_elem_count())
		m_attributes.reserve(startSphere + count);
	std::size_t numRead = m_attributes.restore(hdl, attrStream, startSphere, count);
	// Update material table in case this load was about materials
	if(hdl == m_matIndicesHdl) {
		MaterialIndex* materials = m_attributes.acquire<Device::CPU, MaterialIndex>(hdl);
		for(std::size_t i = startSphere; i < startSphere+numRead; ++i)
			m_uniqueMaterials.emplace(materials[i]);
	}
	return numRead;
}

void Spheres::transform(const ei::Mat3x4& transMat, const ei::Vec3& scale) {
	if (this->get_sphere_count() == 0) return;
	// Invalidate bounding box
	m_boundingBox.min = {
		std::numeric_limits<float>::max(),
		std::numeric_limits<float>::max(),
		std::numeric_limits<float>::max()
	};
	m_boundingBox.max = {
		-std::numeric_limits<float>::max(),
		-std::numeric_limits<float>::max(),
		-std::numeric_limits<float>::max()
	};
	// Transform mesh
	ei::Mat3x3 rotation(transMat);
	ei::Vec3 translation(transMat[3], transMat[7], transMat[11]);
	ei::Sphere* spheres = m_attributes.acquire<Device::CPU, ei::Sphere>(m_spheresHdl);
	for (size_t i = 0; i < this->get_sphere_count(); i++) {
		mAssert(scale.x == scale.y && scale.y == scale.z);
		spheres[i].radius *= scale.x;
		spheres[i].center = rotation * spheres[i].center;
		spheres[i].center += translation;
		m_boundingBox.max = ei::max(util::pun<ei::Vec3>(spheres[i].center + ei::Vec3(spheres[i].radius)), m_boundingBox.max);
		m_boundingBox.min = ei::min(util::pun<ei::Vec3>(spheres[i].center - ei::Vec3(spheres[i].radius)), m_boundingBox.min);
	}
	m_attributes.mark_changed(Device::CPU, m_spheresHdl);
	// TODO: Apply transformation to UV Coordinates
}

// Gets the descriptor with only default attributes (position etc)
template < Device dev >
SpheresDescriptor<dev> Spheres::get_descriptor() {
	this->synchronize<dev>();
	return SpheresDescriptor<dev>{
		static_cast<u32>(this->get_sphere_count()),
		0u,
		this->acquire_const<dev, ei::Sphere>(this->get_spheres_hdl()),
		this->acquire_const<dev, u16>(this->get_material_indices_hdl()),
		ArrayDevHandle_t<dev, ArrayDevHandle_t<dev, void>>{}
	};
}
// Updates the descriptor with the given set of attributes
template < Device dev >
void Spheres::update_attribute_descriptor(SpheresDescriptor<dev>& descriptor,
										  const std::vector<const char*>& attribs) {
	this->synchronize<dev>();
	// Collect the attributes; for that, we iterate the given Attributes and
	// gather them on CPU side (or rather, their device pointers); then
	// we copy it to the actual device
	AttribBuffer<dev>& attribBuffer = m_attribBuffer.get<AttribBuffer<dev>>();
	if(attribs.size() > 0) {
		// Resize the attribute array if necessary
		if(attribBuffer.size < attribs.size()) {
			if(attribBuffer.size == 0)
				attribBuffer.buffer = Allocator<dev>::template alloc_array<ArrayDevHandle_t<dev, void>>(attribs.size());
			else
				attribBuffer.buffer = Allocator<dev>::template realloc<ArrayDevHandle_t<dev, void>>(attribBuffer.buffer, attribBuffer.size,
																	   attribs.size());
			attribBuffer.size = attribs.size();
		}

		std::vector<void*> cpuAttribs(attribs.size());
		for(const char* name : attribs)
			cpuAttribs.push_back(m_attributes.acquire<Device::CPU, char>(name));
		copy<void>(attribBuffer.buffer, cpuAttribs.data(), sizeof(const char*) * attribs.size());
	} else if(attribBuffer.size != 0) {
		attribBuffer.buffer = Allocator<dev>::template free<ArrayDevHandle_t<dev, void>>(attribBuffer.buffer, attribBuffer.size);
	}
	descriptor.numAttributes = static_cast<u32>(attribs.size());
	descriptor.attributes = attribBuffer.buffer;
}

template SpheresDescriptor<Device::CPU> Spheres::get_descriptor<Device::CPU>();
template SpheresDescriptor<Device::CUDA> Spheres::get_descriptor<Device::CUDA>();
template SpheresDescriptor<Device::OPENGL> Spheres::get_descriptor<Device::OPENGL>();
template void Spheres::update_attribute_descriptor<Device::CPU>(SpheresDescriptor<Device::CPU>& descriptor,
																 const std::vector<const char*>&);
template void Spheres::update_attribute_descriptor<Device::CUDA>(SpheresDescriptor<Device::CUDA>& descriptor,
																 const std::vector<const char*>&);
template void Spheres::update_attribute_descriptor<Device::OPENGL>(SpheresDescriptor<Device::OPENGL>& descriptor,
																 const std::vector<const char*>&);
} // namespace mufflon::scene::geometry