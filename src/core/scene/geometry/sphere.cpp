#include "sphere.hpp"
#include "core/scene/descriptors.hpp"
#include "core/scene/tessellation/tessellater.hpp"
#include "core/scene/scenario.hpp"
#include <cuda_runtime.h>
#include "util/punning.hpp"

namespace mufflon::scene::geometry {

Spheres::Spheres() :
	m_attributes(),
	m_spheresHdl(this->template add_attribute<ei::Sphere>("spheres")),
	m_matIndicesHdl(this->template add_attribute<MaterialIndex>("materials")) {
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
	m_boundingBox(sphere.m_boundingBox)
{
	sphere.m_attribBuffer.for_each([&](auto& buffer) {
		using ChangedBuffer = std::decay_t<decltype(buffer)>;
		auto& attribBuffer = m_attribBuffer.template get<ChangedBuffer>();
		attribBuffer.size = buffer.size;
		if(buffer.size == 0u || buffer.buffer == ArrayDevHandle_t<ChangedBuffer::DEVICE, ArrayDevHandle_t<ChangedBuffer::DEVICE, void>>{}) {
			attribBuffer.buffer = ArrayDevHandle_t<ChangedBuffer::DEVICE, ArrayDevHandle_t<ChangedBuffer::DEVICE, void>>{};
		} else {
			attribBuffer.buffer = Allocator<ChangedBuffer::DEVICE>::template alloc_array<ArrayDevHandle_t<ChangedBuffer::DEVICE, void>>(buffer.size);
			copy(attribBuffer.buffer, buffer.buffer, sizeof(ArrayDevHandle_t<ChangedBuffer::DEVICE, void>) * buffer.size);
		}
	});
}

Spheres::Spheres(Spheres&& sphere) :
	m_attributes(std::move(sphere.m_attributes)),
	m_spheresHdl(std::move(sphere.m_spheresHdl)),
	m_matIndicesHdl(std::move(sphere.m_matIndicesHdl)),
	m_boundingBox(std::move(sphere.m_boundingBox))
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
	m_attributes.mark_changed(Device::CPU);
	// Expand bounding box
	m_boundingBox = ei::Box{ m_boundingBox, ei::Box{ei::Sphere{ point, radius }} };
	return hdl;
}

Spheres::SphereHandle Spheres::add(const Point& point, float radius, MaterialIndex idx) {
	SphereHandle hdl = this->add(point, radius);
	m_attributes.acquire<Device::CPU, MaterialIndex>(m_matIndicesHdl)[hdl] = idx;
	m_attributes.mark_changed(Device::CPU);
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

std::size_t Spheres::add_bulk(SphereAttributeHandle hdl, const SphereHandle& startSphere,
							  std::size_t count, util::IByteReader& attrStream) {
	if(startSphere >= m_attributes.get_attribute_elem_count())
		return 0u;
	if(startSphere + count > m_attributes.get_attribute_elem_count())
		m_attributes.reserve(startSphere + count);
	std::size_t numRead = m_attributes.restore(hdl, attrStream, startSphere, count);
	return numRead;
}

void Spheres::transform(const ei::Mat3x4& transMat) {
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
	ei::Sphere* spheres = m_attributes.acquire<Device::CPU, ei::Sphere>(m_spheresHdl);
	const float scale = ei::len(ei::Vec3(transMat, 0u, 0u));
	for (size_t i = 0; i < this->get_sphere_count(); i++) {
		mAssertMsg(scale == ei::len(ei::Vec3(transMat, 0u, 1u))
				   && scale == ei::len(ei::Vec3(transMat, 0u, 1u)),
				   "The scales in the transformation matrix must be all equal!");
		spheres[i].radius *= scale;
		spheres[i].center = ei::transform(spheres[i].center, transMat);
		m_boundingBox.max = ei::max(util::pun<ei::Vec3>(spheres[i].center + ei::Vec3(spheres[i].radius)), m_boundingBox.max);
		m_boundingBox.min = ei::min(util::pun<ei::Vec3>(spheres[i].center - ei::Vec3(spheres[i].radius)), m_boundingBox.min);
	}
	m_attributes.mark_changed(Device::CPU);
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
										  const std::vector<AttributeIdentifier>& attribs) {
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
				attribBuffer.buffer = Allocator<dev>::template realloc<ArrayDevHandle_t<dev, void>>(attribBuffer.buffer,
																									attribBuffer.size,
																									attribs.size());
			attribBuffer.size = attribs.size();
		}

		std::vector<ArrayDevHandle_t<dev, void>> cpuAttribs(attribs.size());
		for(const auto& ident : attribs)
			cpuAttribs.push_back(this->template acquire<dev, void>(ident));
		copy<ArrayDevHandle_t<dev, void>>(attribBuffer.buffer, cpuAttribs.data(), sizeof(cpuAttribs.front()) * attribs.size());
	} else if(attribBuffer.size != 0) {
		attribBuffer.buffer = Allocator<dev>::template free<ArrayDevHandle_t<dev, void>>(attribBuffer.buffer, attribBuffer.size);
	}
	descriptor.numAttributes = static_cast<u32>(attribs.size());
	descriptor.attributes = attribBuffer.buffer;
}


void Spheres::displace(tessellation::TessLevelOracle& oracle, const Scenario& scenario) {
	(void)oracle;
	(void)scenario;
	// There is no displacement we can perform for a perfect sphere (yet)
}

void Spheres::tessellate(tessellation::TessLevelOracle& oracle, const Scenario* scenario,
						 const bool usePhong) {
	(void)oracle;
	(void)scenario;
	// There is no tessellation we can/have to perform for a perfect sphere (yet)
}

template SpheresDescriptor<Device::CPU> Spheres::get_descriptor<Device::CPU>();
template SpheresDescriptor<Device::CUDA> Spheres::get_descriptor<Device::CUDA>();
template SpheresDescriptor<Device::OPENGL> Spheres::get_descriptor<Device::OPENGL>();
template void Spheres::update_attribute_descriptor<Device::CPU>(SpheresDescriptor<Device::CPU>& descriptor,
																const std::vector<AttributeIdentifier>&);
template void Spheres::update_attribute_descriptor<Device::CUDA>(SpheresDescriptor<Device::CUDA>& descriptor,
																 const std::vector<AttributeIdentifier>&);
template void Spheres::update_attribute_descriptor<Device::OPENGL>(SpheresDescriptor<Device::OPENGL>& descriptor,
																 const std::vector<AttributeIdentifier>&);
} // namespace mufflon::scene::geometry