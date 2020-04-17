#pragma once

#include "attribute_sizes.hpp"
#include "attribute_handles.hpp"
#include "util/tagged_tuple.hpp"
#include "core/memory/residency.hpp"
#include "core/scene/geometry/polygon_mesh.hpp"

#include <optional>
#include <vector>

namespace mufflon::util {

class IByteReader;
class IByteWriter;

} // namespace mufflon::util

namespace mufflon::scene {

struct VertexAttributeHandle final : public AttributeHandle {
	VertexAttributeHandle(const AttributeIdentifier& ident, const std::uint32_t index) : AttributeHandle(ident, index) {}
	VertexAttributeHandle(const AttributeHandle hdl) : AttributeHandle{ hdl } {}
};
struct FaceAttributeHandle final : public AttributeHandle {
	FaceAttributeHandle(const AttributeIdentifier& ident, const std::uint32_t index) : AttributeHandle(ident, index) {}
	FaceAttributeHandle(const AttributeHandle hdl) : AttributeHandle{ hdl } {}
};
struct SphereAttributeHandle final : public AttributeHandle {
	SphereAttributeHandle(const AttributeIdentifier& ident, const std::uint32_t index) : AttributeHandle(ident, index) {}
	SphereAttributeHandle(const AttributeHandle hdl) : AttributeHandle{ hdl } {}
};

/**
 * Attribute pool for multi-device attributes.
 * Holds all of its memory on every device.
 * Has management semantics, albeit with a slightly different interface in that it works on
 * individual attributes as well.
 */
class AttributePool {
public:
	AttributePool() = default;
	AttributePool(const AttributePool& pool);
	AttributePool(AttributePool&& pool) noexcept;
	AttributePool& operator=(const AttributePool&) = delete;
	AttributePool& operator=(AttributePool&&) noexcept;
	~AttributePool();

	// Adds a new attribute
	AttributeHandle add_attribute(const AttributeIdentifier& ident);
	std::optional<AttributeHandle> find_attribute(const AttributeIdentifier& ident) const;
	void remove(AttributeHandle handle);

	// Causes force-unload on actual reserve
	// Capacity is in terms of elements, not bytes
	void reserve(std::size_t capacity);

	// Resizes the attribute, leaves the memory uninitialized
	// Force-unloads non-CPU pools if reserve necessary
	void resize(std::size_t size);

	// Shrinks the memory to fit the element count on all devices
	// Does not unload any device memory
	void shrink_to_fit();

	// Copies over all attributes of one element slot to another on the specified device
	template < Device dev >
	void copy(const std::size_t from, const std::size_t to);
	template < Device dev >
	void copy(AttributePool& fromPool, const std::size_t from, const std::size_t to);

	template < Device dev, class T >
	ArrayDevHandle_t<dev, T> acquire(AttributeHandle hdl) {
		mAssert(hdl.index < m_attributes.size());
		mAssert(!m_attributes[hdl.index].erased);
		this->template synchronize<dev>();
		return as<ArrayDevHandle_t<dev, T>, ArrayDevHandle_t<dev, char>>(
			m_pools.template get<PoolHandle<dev>>().handle.get() + m_attributes[hdl.index].poolOffset);
	}

	template < Device dev, class T >
	ConstArrayDevHandle_t<dev, T> acquire_const(AttributeHandle hdl) {
		mAssert(hdl.index < m_attributes.size());
		mAssert(!m_attributes[hdl.index].erased);
		this->template synchronize<dev>();
		return as<ArrayDevHandle_t<dev, T>, ArrayDevHandle_t<dev, char>>(
			m_pools.template get<PoolHandle<dev>>().handle.get() + m_attributes[hdl.index].poolOffset);
	}

	template < Device dev >
	void synchronize();

	template < Device dev >
	void unload();

	void mark_changed(Device dev);

	// Loads the attribute from a byte stream, starting at elem start
	// Resizes the attributes if necessary
	// Returns the number of read instances.
	std::size_t restore(AttributeHandle hdl, util::IByteReader& attrStream,
						std::size_t start, std::size_t count);

	// Store the attribute to a byte stream, starting at elem start
	std::size_t store(AttributeHandle hdl, util::IByteWriter& attrStream,
					  std::size_t start, std::size_t count);

	std::size_t get_attribute_elem_count() const noexcept {
		return m_attribElemCount;
	}

	std::size_t get_attribute_elem_capacity() const noexcept {
		return m_attribElemCapacity;
	}
private:
	// Bookkeeping for attributes
	struct AttribInfo {
		std::size_t elemSize = 0u;
		std::size_t poolOffset = 0u;
		StringView name;
		// Stores whether the attribute has been erased and can be overwritten
		bool erased = false;
	};

	template < Device dev >
	struct PoolHandle {
		static constexpr Device DEVICE = dev;
		unique_device_ptr<dev, char[]> handle;
	};

	std::size_t insert_attribute_at_first_empty(AttribInfo&& info);
	
	std::size_t m_attribElemCount = 0u;
	std::size_t m_attribElemCapacity = 0u;
	std::size_t m_poolSize = 0u;
	util::TaggedTuple<
		PoolHandle<Device::CPU>, 
		PoolHandle<Device::CUDA>,
		PoolHandle<Device::OPENGL>> m_pools = {};

	std::vector<AttribInfo> m_attributes;
};

} // namespace mufflon::scene
