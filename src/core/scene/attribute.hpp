#pragma once

#include "core/memory/allocator.hpp"
#include "core/memory/residency.hpp"
#include "core/scene/geometry/polygon_mesh.hpp"
#include "util/byte_io.hpp"
#include "util/tagged_tuple.hpp"
#include <cstddef>
#include <functional>
#include <climits>
#include <stdexcept>
#include <string>
#include "util/string_view.hpp"
#include <type_traits>
#include <map>
#include <vector>
#include "core/memory/dyntype_memory.hpp"

namespace mufflon { namespace scene {
	// TODO: move to geometry namespace?

// Represents a handle to an attribute (basically a glorified index)
struct AttributeHandle {
	std::size_t index = std::numeric_limits<std::size_t>::max();
	bool operator==(const AttributeHandle& rhs) const noexcept {
		return index == rhs.index;
	}
};
// Overload aliases
struct FaceAttributeHandle : public AttributeHandle {
	static constexpr bool IS_FACE = true;
	FaceAttributeHandle(const AttributeHandle& attr) : AttributeHandle{attr} {};
	FaceAttributeHandle(std::size_t index) : AttributeHandle{index} {};
};
struct VertexAttributeHandle : public AttributeHandle {
	static constexpr bool IS_FACE = false;
	VertexAttributeHandle(const AttributeHandle& attr) : AttributeHandle{attr} {};
	VertexAttributeHandle(std::size_t index) : AttributeHandle{index} {};
};
struct SphereAttributeHandle : public AttributeHandle {
	SphereAttributeHandle(const AttributeHandle& attr) : AttributeHandle{attr} {};
	SphereAttributeHandle(std::size_t index) : AttributeHandle{index} {};
};

/**
 * Attribute pool for multi-device attributes, shared with OpenMesh.
 * Only holds non-CPU memory on its own, CPU-side attributes have their memory in OpenMesh.
 * Has management semantics, albeit with a slightly different interface in that it works on
 * individual attributes as well.
 * IsFace: defines if the pool is for vertex or face attributes
 */
template < bool IsFace >
class OpenMeshAttributePool {
	template < Device dev >
	struct AcquireHelper {};

	template <> 
	struct AcquireHelper< Device::CPU > {
		AcquireHelper(OpenMeshAttributePool<IsFace>& parent) : parent(parent) {}

		template <class T>
		ArrayDevHandle_t<Device::CPU, T> acquire(size_t index) {
			return as<T*, char*>(parent.m_attributes[index].accessor(parent.m_mesh));
		}

		OpenMeshAttributePool<IsFace>& parent;
	};

	template <>
	struct AcquireHelper< Device::CUDA > {
		AcquireHelper(OpenMeshAttributePool<IsFace>& parent) : parent(parent) {}

		template <class T>
		ArrayDevHandle_t<Device::CUDA, T> acquire(size_t index) {
			return as<T*, char*>(&parent.m_cudaPool[parent.m_attributes[index].poolOffset]);
		}

		OpenMeshAttributePool<IsFace>& parent;
	};

	template <>
	struct AcquireHelper< Device::OPENGL > {
		AcquireHelper(OpenMeshAttributePool<IsFace>& parent) : parent(parent) {}

		template <class T>
		ArrayDevHandle_t<Device::OPENGL, T> acquire(size_t index) {
			return ArrayDevHandle_t<Device::OPENGL, T>(0);
		}

		OpenMeshAttributePool<IsFace>& parent;
	};

	friend AcquireHelper < Device::CPU >;
	friend AcquireHelper < Device::CUDA >;
	friend AcquireHelper < Device::OPENGL >;
public:
	static constexpr bool IS_FACE = IsFace;
	template < class T >
	using PropertyHandleType = std::conditional_t<IS_FACE, OpenMesh::FPropHandleT<T>, OpenMesh::VPropHandleT<T>>;
//	using AttributeHandle = std::conditional_t<IS_FACE, FaceAttributeHandle, VertexAttributeHandle>;

	OpenMeshAttributePool(geometry::PolygonMeshType &mesh);
	OpenMeshAttributePool(const OpenMeshAttributePool&) = delete;
	OpenMeshAttributePool& operator=(const OpenMeshAttributePool&) = delete;
	OpenMeshAttributePool& operator=(OpenMeshAttributePool&&) = delete;
	OpenMeshAttributePool(OpenMeshAttributePool&& pool);
	~OpenMeshAttributePool();

	// Copies over name maps etc.
	void copy(const OpenMeshAttributePool<IsFace>& pool);

	// Adds a new attribute; force-unload non-CPU pools
	template < class T >
	AttributeHandle add_attribute(std::string name) {
		if(m_nameMap.find(name) != m_nameMap.end())
			throw std::runtime_error("Attribute '" + name + "' already exists");
		this->unload<Device::CPU>();
		// Create the OpenMesh-Property
		PropertyHandleType<T> propHandle;
		m_mesh.add_property(propHandle, name);
		if(!propHandle.is_valid())
			throw std::runtime_error("Failed to add property '" + name + "' to OpenMesh");
		// Create the accessor...
		AttribInfo info{
			sizeof(T),
			m_poolSize,
			[propHandle](geometry::PolygonMeshType& mesh) {
				auto& prop = mesh.property(propHandle);
				return reinterpret_cast<char*>(prop.data_vector().data());
			}
		};
		m_poolSize += sizeof(T) * m_attribElemCapacity;
		// ...and map the name to the index
		m_nameMap.emplace(std::move(name), m_attributes.size());
		m_attributes.push_back(std::move(info));
		return AttributeHandle{ m_attributes.size() - 1u };
	}

	// Adds an attribute that OpenMesh supposedly already has
	template < class T >
	AttributeHandle register_attribute(PropertyHandleType<T> propHdl) {
		auto& prop = m_mesh.property(propHdl);
		if(m_nameMap.find(prop.name()) != m_nameMap.end())
			throw std::runtime_error("Registered attribute '" + prop.name() + "' already exists");
		this->unload<Device::CUDA>();
		// Create the accessor...
		AttribInfo info{
			sizeof(T),
			m_poolSize,
			[propHdl](geometry::PolygonMeshType& mesh) {
				auto& prop = mesh.property(propHdl);
				return reinterpret_cast<char*>(prop.data_vector().data());
			}
		};
		m_poolSize += sizeof(T) * m_attribElemCapacity;
		// ...and map the name to the index
		m_nameMap.emplace(prop.name(), m_attributes.size());
		m_attributes.push_back(std::move(info));
		return AttributeHandle{ m_attributes.size() - 1u };
	}

	// Reserving memory force-unloads other devices
	// Capacity is in terms of elements, not bytes
	void reserve(std::size_t capacity);

	// Resizes the attribute, leaves the memory uninitialized
	// Force-unloads non-CPU pools if reserve necessary
	void resize(std::size_t size);

	// Shrinks the memory to fit the element count on all devices
	// Does not unload any device memory
	// Also performs garbage-collection for OpenMesh
	void shrink_to_fit();

	template < Device dev, class T >
	ArrayDevHandle_t<dev, T> acquire(AttributeHandle hdl) {
		mAssert(hdl.index < m_attributes.size());
		// Mark both the specific attribute flags and the flags that indicate a change is present
		AcquireHelper<dev> helper(*this);
		return helper.template acquire<T>(hdl.index);
		/*switch (dev) {
			case Device::CPU: return reinterpret_cast<ArrayDevHandle_t<dev, T>>((void*)m_attributes[hdl.index].accessor(m_mesh));
			case Device::CUDA: return reinterpret_cast<ArrayDevHandle_t<dev, T>>((void*)&m_cudaPool[m_attributes[hdl.index].poolOffset]);
			default: return ArrayDevHandle_t<dev, T>(0);
		}*/
	}

	template < Device dev, class T >
	ConstArrayDevHandle_t<dev, T> acquire_const(AttributeHandle hdl) {
		mAssert(hdl.index < m_attributes.size());
		this->synchronize<dev>();
		AcquireHelper<dev> helper(*const_cast<OpenMeshAttributePool<IsFace>*>(this));
		return helper.template acquire<T>(hdl.index);
		/*switch (dev) {
			case Device::CPU: return reinterpret_cast<ConstArrayDevHandle_t<dev, T>>((void*)m_attributes[hdl.index].accessor(m_mesh));
			case Device::CUDA: return reinterpret_cast<ConstArrayDevHandle_t<dev, T>>((void*)&m_cudaPool[m_attributes[hdl.index].poolOffset]);
			default: return ConstArrayDevHandle_t<dev, T>(0);
		}*/
	}

	template < Device dev, class T >
	ArrayDevHandle_t<dev, T> acquire(StringView name) {
		return acquire<dev, T>(get_attribute_handle(name));
	}

	template < Device dev, class T >
	ConstArrayDevHandle_t<dev, T> acquire_const(StringView name) {
		return acquire_const<dev, T>(get_attribute_handle(name));
	}

	template < Device dev >
	void synchronize();

	template < Device dev >
	void unload();

	void mark_changed(Device dev);

	// Loads the attribute from a byte stream, starting at elem start
	// Resizes the attributes if necessary
	std::size_t restore(AttributeHandle hdl, util::IByteReader& attrStream,
						std::size_t start, std::size_t count);
	std::size_t restore(StringView name, util::IByteReader& attrStream,
						std::size_t start, std::size_t count) {
		return this->restore(get_attribute_handle(name), attrStream, start, count);
	}

	// Store the attribute to a byte stream, starting at elem start
	std::size_t store(AttributeHandle hdl, util::IByteWriter& attrStream,
					  std::size_t start, std::size_t count);
	std::size_t store(StringView name, util::IByteWriter& attrStream,
					  std::size_t start, std::size_t count) {
		return this->store(get_attribute_handle(name), attrStream, start, count);
	}

	std::size_t get_attribute_count() const noexcept {
		return m_attributes.size();
	}

	std::size_t get_attribute_elem_count() const noexcept {
		return m_attribElemCount;
	}

	std::size_t get_attribute_elem_capacity() const noexcept {
		return m_attribElemCapacity;
	}

	// Resolves a name to an attribute
	AttributeHandle get_attribute_handle(StringView name);
private:
	// Bookkeeping for attributes
	struct AttribInfo {
		std::size_t elemSize = 0u;
		std::size_t poolOffset = 0u;
		std::function<char*(geometry::PolygonMeshType&)> accessor;
	};

	geometry::PolygonMeshType &m_mesh;	// References the OpenMesh mesh
	std::map<std::string, std::size_t, std::less<>> m_nameMap;
	std::size_t m_attribElemCount = 0u;
	std::size_t m_attribElemCapacity = 0u;
	std::size_t m_poolSize = 0u;
	ArrayDevHandle_t<Device::CUDA, char> m_cudaPool = nullptr;
	// TODO: OpenGL pool?

	std::vector<AttribInfo> m_attributes;
	bool m_openMeshSynced = false;
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
	AttributePool& operator=(const AttributePool&) = delete;
	AttributePool& operator=(AttributePool&&) = delete;
	AttributePool(AttributePool&& pool);
	~AttributePool();

	// Adds a new attribute
	template < class T >
	AttributeHandle add_attribute(std::string name) {
		if(m_nameMap.find(name) != m_nameMap.end())
			throw std::runtime_error("Attribute '" + name + "' already exists");
		// Create the OpenMesh-Property
		// Create the accessor...
		AttribInfo info{
			sizeof(T),
			m_poolSize
		};
		m_poolSize += sizeof(T) * m_attribElemCapacity;
		// ...and map the name to the index
		m_nameMap.emplace(std::move(name), m_attributes.size());
		m_attributes.push_back(std::move(info));
		return AttributeHandle{ m_attributes.size() - 1u };
	}

	// Causes force-unload on actual reserve
	// Capacity is in terms of elements, not bytes
	void reserve(std::size_t capacity);

	// Resizes the attribute, leaves the memory uninitialized
	// Force-unloads non-CPU pools if reserve necessary
	void resize(std::size_t size);

	// Shrinks the memory to fit the element count on all devices
	// Does not unload any device memory
	void shrink_to_fit();

	template < Device dev, class T >
	ArrayDevHandle_t<dev, T> acquire(AttributeHandle hdl) {
		mAssert(hdl.index < m_attributes.size());
		this->synchronize<dev>();
		return as<ArrayDevHandle_t<dev, T>, ArrayDevHandle_t<dev, char>>(
			m_pools.template get<PoolHandle<dev>>().handle + m_attributes[hdl.index].poolOffset);
	}

	template < Device dev, class T >
	ConstArrayDevHandle_t<dev, T> acquire_const(AttributeHandle hdl) {
		mAssert(hdl.index < m_attributes.size());
		this->synchronize<dev>();
		return as<ArrayDevHandle_t<dev, T>, ArrayDevHandle_t<dev, char>>(
			m_pools.template get<PoolHandle<dev>>().handle + m_attributes[hdl.index].poolOffset);
	}

	template < Device dev, class T >
	ArrayDevHandle_t<dev, T> acquire(StringView name) {
		return acquire<dev, T>(get_attribute_handle(name));
	}

	template < Device dev, class T >
	ConstArrayDevHandle_t<dev, T> acquire_const(StringView name) {
		return acquire_const<dev, T>(get_attribute_handle(name));
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
	/*std::size_t restore(StringView name, util::IByteReader& attrStream,
						std::size_t start, std::size_t count) {
		return this->restore(get_attribute_handle(name), attrStream, start, count);
	}*/

	// Store the attribute to a byte stream, starting at elem start
	std::size_t store(AttributeHandle hdl, util::IByteWriter& attrStream,
					  std::size_t start, std::size_t count);
	/*std::size_t store(StringView name, util::IByteWriter& attrStream,
					  std::size_t start, std::size_t count) {
		return this->store(get_attribute_handle(name), attrStream, start, count);
	}*/

	std::size_t get_attribute_count() const noexcept {
		return m_attributes.size();
	}

	std::size_t get_attribute_elem_count() const noexcept {
		return m_attribElemCount;
	}

	std::size_t get_attribute_elem_capacity() const noexcept {
		return m_attribElemCapacity;
	}

	// Resolves a name to an attribute
	AttributeHandle get_attribute_handle(StringView name);
private:
	// Bookkeeping for attributes
	struct AttribInfo {
		std::size_t elemSize = 0u;
		std::size_t poolOffset = 0u;
	};

	template < Device dev >
	struct PoolHandle {
		static constexpr Device DEVICE = dev;
		ArrayDevHandle_t<dev, char> handle = ArrayDevHandle_t<dev, char>{};
	};

	std::map<std::string, std::size_t, std::less<>> m_nameMap;
	std::size_t m_attribElemCount = 0u;
	std::size_t m_attribElemCapacity = 0u;
	std::size_t m_poolSize = 0u;
	util::TaggedTuple<
		PoolHandle<Device::CPU>, 
		PoolHandle<Device::CUDA>,
		PoolHandle<Device::OPENGL>> m_pools = {};

	std::vector<AttribInfo> m_attributes;
};

}} // namespace mufflon::scene