#pragma once

#include "core/memory/allocator.hpp"
#include "core/memory/residency.hpp"
#include "core/scene/geometry/polygon_mesh.hpp"
#include "util/byte_io.hpp"
#include "util/flag.hpp"
#include "util/tagged_tuple.hpp"
#include <cstddef>
#include <functional>
#include <climits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <map>
#include <vector>

namespace mufflon { namespace scene {

/**
 * Attribute pool for multi-device attributes, shared with OpenMesh.
 * Only holds non-CPU memory on its own, CPU-side attributes have their memory in OpenMesh.
 * Has management semantics, albeit with a slightly different interface in that it works on
 * individual attributes as well.
 * IsFace: defines if the pool is for vertex or face attributes
 */
template < bool IsFace >
class OpenMeshAttributePool {
public:
	static constexpr bool IS_FACE = IsFace;
	template < class T >
	using PropertyHandleType = std::conditional_t<IS_FACE, OpenMesh::FPropHandleT<T>, OpenMesh::VPropHandleT<T>>;

	// Represents a handle to an attribute (basically a glorified index)
	struct AttributeHandle {
		std::size_t index = std::numeric_limits<std::size_t>::max();
	};

	OpenMeshAttributePool(geometry::PolygonMeshType &mesh) : m_mesh(mesh) {}
	OpenMeshAttributePool(const OpenMeshAttributePool&) = delete;
	OpenMeshAttributePool& operator=(const OpenMeshAttributePool&) = delete;
	OpenMeshAttributePool& operator=(OpenMeshAttributePool&&) = delete;
	OpenMeshAttributePool(OpenMeshAttributePool&& pool) :
		m_mesh(pool.m_mesh),
		m_nameMap(std::move(pool.m_nameMap)),
		m_attribElemCount(pool.m_attribElemCount),
		m_attribElemCapacity(pool.m_attribElemCapacity),
		m_poolSize(pool.m_poolSize),
		m_cudaPool(pool.m_cudaPool),
		m_dirty(pool.m_dirty),
		m_attributes(std::move(pool.m_attributes)) {
		m_cudaPool = ArrayDevHandle_t<Device::CUDA, char>{};
	}
	~OpenMeshAttributePool() {
		if(m_cudaPool)
			m_cudaPool = Allocator<Device::CUDA>::free(m_cudaPool, m_poolSize);
	}

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
			},
			util::DirtyFlags<Device>{}
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
			},
			util::DirtyFlags<Device>{}
		};
		m_poolSize += sizeof(T) * m_attribElemCapacity;
		// ...and map the name to the index
		m_nameMap.emplace(prop.name(), m_attributes.size());
		m_attributes.push_back(std::move(info));
		return AttributeHandle{ m_attributes.size() - 1u };
	}

	// Reserving memory force-unloads other devices
	// Capacity is in terms of elements, not bytes
	void reserve(std::size_t capacity) {
		if(capacity <= m_attribElemCapacity)
			return;

		this->unload<Device::CUDA>();
		if constexpr (IS_FACE)
			m_mesh.reserve(m_mesh.n_vertices(), m_mesh.n_edges(), capacity);
		else
			m_mesh.reserve(capacity, m_mesh.n_edges(), m_mesh.n_faces());

		std::size_t currOffset = 0u;
		// Adjust pool offsets for the attributes
		for (auto& attrib : m_attributes) {
			attrib.poolOffset = currOffset;
			currOffset += attrib.elemSize * capacity;
		}

		m_poolSize = currOffset;
		m_attribElemCapacity = capacity;
	}

	// Resizes the attribute, leaves the memory uninitialized
	// Force-unloads non-CPU pools if reserve necessary
	void resize(std::size_t size) {
		this->reserve(size);
		if constexpr(IS_FACE)
			m_mesh.resize(m_mesh.n_vertices(), m_mesh.n_edges(), size);
		else
			m_mesh.resize(size, m_mesh.n_edges(), m_mesh.n_faces());
		m_attribElemCount = size;
	}

	// Shrinks the memory to fit the element count on all devices
	// Does not unload any device memory
	// Also performs garbage-collection for OpenMesh
	void shrink_to_fit() {
		m_mesh.garbage_collection();
		if(m_attribElemCount == m_attribElemCapacity)
			return;
		if constexpr (IS_FACE)
			m_mesh.resize(m_mesh.n_vertices(), m_mesh.n_edges(), m_attribElemCount);
		else
			m_mesh.resize(m_attribElemCount, m_mesh.n_edges(), m_mesh.n_faces());

		if(m_attribElemCount != 0) {
			std::size_t bytes = m_attribElemCount * m_poolSize / m_attribElemCapacity;
			if(m_cudaPool)
				m_cudaPool = Allocator<Device::CUDA>::realloc(m_cudaPool, m_poolSize, bytes);
			m_poolSize = bytes;
		} else {
			if(m_cudaPool)
				m_cudaPool = Allocator<Device::CUDA>::free(m_cudaPool, m_poolSize);
			m_poolSize = 0u;
		}
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

	template < Device dev, class T >
	ArrayDevHandle_t<dev, T> acquire(AttributeHandle hdl) {
		mAssert(hdl.index < m_attributes.size());
		// Mark both the specific attribute flags and the flags that indicate a change is present
		m_attributes[hdl.index].dirty.mark_changed(dev);
		m_dirty.mark_changed(dev);
		switch (dev) {
			case Device::CPU: return reinterpret_cast<T*>(m_attributes[hdl.index].accessor(m_mesh));
			case Device::CUDA: return reinterpret_cast<T*>(&m_cudaPool[m_attributes[hdl.index].poolOffset]);
			default: return nullptr;
		}
	}

	template < Device dev, class T >
	ConstArrayDevHandle_t<dev, T> acquire_const(AttributeHandle hdl) {
		mAssert(hdl.index < m_attributes.size());
		this->synchronize<dev>(hdl);
		switch (dev) {
			case Device::CPU: return reinterpret_cast<const T*>(m_attributes[hdl.index].accessor(m_mesh));
			case Device::CUDA: return reinterpret_cast<const T*>(&m_cudaPool[m_attributes[hdl.index].poolOffset]);
			default: return nullptr;
		}
	}

	template < Device dev, class T >
	ArrayDevHandle_t<dev, T> acquire(std::string_view name) {
		return acquire<dev, T>(get_attribute_handle(name));
	}

	template < Device dev, class T >
	ConstArrayDevHandle_t<dev, T> acquire_const(std::string_view name) {
		return acquire_const<dev, T>(get_attribute_handle(name));
	}

	template < Device dev >
	void synchronize() {
		if(!m_dirty.has_changes())
			return;
		if(m_dirty.has_competing_changes())
			throw std::runtime_error("Competing changes in attribute pool");
		if(!m_dirty.needs_sync(dev))
			return;
		if(m_attribElemCount == 0)
			return;

		switch (dev) {
			case Device::CPU: {
				mAssert(m_cudaPool != nullptr);
				// Copy over dirty attributes
				std::size_t currOffset = 0u;
				for (auto& attrib : m_attributes) {
					if(attrib.dirty.needs_sync(dev)) {
						char* cpuData = attrib.accessor(m_mesh);
						copy(cpuData, &m_cudaPool[currOffset], attrib.elemSize * m_attribElemCount);
						currOffset += attrib.elemSize * m_attribElemCount;
						attrib.dirty.mark_synced(dev);
					}
				}
			}	break;
			case Device::CUDA: {
				bool copyAll = !m_cudaPool;
				if(!m_cudaPool)
					m_cudaPool = Allocator<Device::CUDA>::alloc_array<char>(m_poolSize);
				// Copy over all attributes
				std::size_t currOffset = 0u;
				for (auto& attrib : m_attributes) {
					if(copyAll || attrib.dirty.needs_sync(dev)) {
						const char* cpuData = attrib.accessor(m_mesh);
						copy(&m_cudaPool[currOffset], cpuData, attrib.elemSize * m_attribElemCount);
						currOffset += attrib.elemSize * m_attribElemCount;
					}
					attrib.dirty.mark_synced(dev);
				}
			}	break;
		}
		m_dirty.mark_synced(dev);
	}

	template < Device dev >
	void synchronize(AttributeHandle hdl) {
		mAssert(hdl.index < m_attributes.size());
		if(dev == Device::CUDA && !m_cudaPool)
			this->synchronize<dev>();

		if(!m_attributes[hdl.index].dirty.needs_sync(dev))
			return;

		const std::size_t offset = m_attributes[hdl.index].poolOffset;
		switch(dev) {
			case Device::CPU: {
				copy(m_attributes[hdl.index].accessor(m_mesh) + offset, m_cudaPool + offset, m_attributes[hdl.index].elemSize * m_attribElemCount);
			}	break;
			case Device::CUDA: {
				copy(m_cudaPool + offset, m_attributes[hdl.index].accessor(m_mesh) + offset, m_attributes[hdl.index].elemSize * m_attribElemCount);
			}	break;
		}
	}

	template < Device dev >
	void synchronize(std::string_view name) {
		return synchronize(get_attribute_handle(name));
	}

	template < Device dev >
	void unload() {
		// We cannot unload (CPU) OpenMesh data (without removing the property?)
		switch (dev) {
			case Device::CUDA:
				if(m_cudaPool)
					m_cudaPool = Allocator<Device::CUDA>::free(m_cudaPool, m_poolSize);
				break;
		}
	}

	void mark_changed(Device dev, AttributeHandle hdl) {
		mAssert(hdl.index < m_attributes.size());
		m_dirty.mark_changed(dev);
		m_attributes[hdl.index].dirty.mark_changed(dev);
	}
	void mark_changed(Device dev, std::string_view name) {
		mark_changed(dev, get_attribute_handle(name));
	}
	void mark_changed(Device dev) {
		m_dirty.mark_changed(dev);
		for (auto& attr : m_attributes) {
			attr.dirty.mark_changed(dev);
		}
	}

	// Loads the attribute from a byte stream
	std::size_t restore(AttributeHandle hdl, util::IByteReader& attrStream,
						std::size_t start, std::size_t count) {
		mAssert(hdl.index < m_attributes.size());
		this->synchronize<Device::CPU>(hdl);
		if(start + count > m_attribElemCount)
			this->resize(start + count);

		AttribInfo& attribute = m_attributes[hdl.index];
		const std::size_t elemSize = attribute.elemSize;
		char* mem = attribute.accessor(m_mesh) + elemSize * start;
		std::size_t read = attrStream.read(mem, elemSize * count);
		if(read > 0)
			this->mark_changed(Device::CPU, hdl);
		return read / attribute.elemSize;
	}

	// Loads the attribute from a byte stream
	std::size_t restore(std::string_view name, util::IByteReader& attrStream,
						std::size_t start, std::size_t count) {
		return this->restore(get_attribute_handle(name), attrStream, start, count);
	}

	std::size_t store(AttributeHandle hdl, util::IByteWriter& attrStream,
						std::size_t start, std::size_t count) {
		mAssert(hdl.index < m_attributes.size());
		this->synchronize<Device::CPU>(hdl);
		std::size_t actualCount = count;
		if(start + count > m_attribElemCount)
			actualCount = m_attribElemCount - start;

		AttribInfo& attribute = m_attributes[hdl.index];
		const std::size_t elemSize = attribute.elemSize;
		const char* mem = attribute.accessor(m_mesh) + elemSize * start;
		return attrStream.write(mem, elemSize * actualCount) / attribute.elemSize;
	}

	std::size_t store(std::string_view name, util::IByteWriter& attrStream,
					  std::size_t start, std::size_t count) {
		return this->store(get_attribute_handle(name), attrStream, start, count);
	}
	
private:
	// Bookkeeping for attributes
	struct AttribInfo {
		std::size_t elemSize = 0u;
		std::size_t poolOffset = 0u;
		std::function<char*(geometry::PolygonMeshType&)> accessor;
		util::DirtyFlags<Device> dirty{};
	};

	// Resolves a name to an attribute
	AttributeHandle get_attribute_handle(std::string_view name) {
		auto mapIter = m_nameMap.find(name);
		if(mapIter == m_nameMap.end())
			throw std::runtime_error("Could not find attribute '" + std::string(name) + "'");
		return AttributeHandle{ mapIter->second };
	}

	geometry::PolygonMeshType &m_mesh;	// References the OpenMesh mesh
	std::map<std::string, std::size_t, std::less<>> m_nameMap;
	std::size_t m_attribElemCount = 0u;
	std::size_t m_attribElemCapacity = 0u;
	std::size_t m_poolSize = 0u;
	ArrayDevHandle_t<Device::CUDA, char> m_cudaPool = nullptr;
	// TODO: OpenGL pool?

	util::DirtyFlags<Device> m_dirty;
	std::vector<AttribInfo> m_attributes;
};
/**
 * Attribute pool for multi-device attributes.
 * Holds all of its memory on every device.
 * Has management semantics, albeit with a slightly different interface in that it works on
 * individual attributes as well.
 */
class AttributePool {
public:
	// Represents a handle to an attribute (basically a glorified index)
	struct AttributeHandle {
		std::size_t index = std::numeric_limits<std::size_t>::max();
	};

	AttributePool() = default;
	AttributePool(const AttributePool&) = delete;
	AttributePool& operator=(const AttributePool&) = delete;
	AttributePool& operator=(AttributePool&&) = delete;
	AttributePool(AttributePool&& pool) :
		m_nameMap(std::move(pool.m_nameMap)),
		m_attribElemCount(pool.m_attribElemCount),
		m_attribElemCapacity(pool.m_attribElemCapacity),
		m_poolSize(pool.m_poolSize),
		m_pools(pool.m_pools),
		m_dirty(pool.m_dirty),
		m_attributes(std::move(pool.m_attributes))
	{
		m_pools.for_each([](auto& elem) {
			using ChangedBuffer = std::decay_t<decltype(elem)>;
			elem.handle = ArrayDevHandle_t<ChangedBuffer::DEVICE, char>{};
		});
	}
	~AttributePool() {
		m_pools.for_each([len = m_poolSize](auto& elem) {
			using ChangedBuffer = std::decay_t<decltype(elem)>;
			if(elem.handle)
				elem.handle = Allocator<ChangedBuffer::DEVICE>::free(elem.handle, len);
		});
	}

	// Adds a new attribute
	template < class T >
	AttributeHandle add_attribute(std::string name) {
		if(m_nameMap.find(name) != m_nameMap.end())
			throw std::runtime_error("Attribute '" + name + "' already exists");
		// Create the OpenMesh-Property
		// Create the accessor...
		AttribInfo info{
			sizeof(T),
			m_poolSize,
			util::DirtyFlags<Device>{}
		};
		m_poolSize += sizeof(T) * m_attribElemCapacity;
		// ...and map the name to the index
		m_nameMap.emplace(std::move(name), m_attributes.size());
		m_attributes.push_back(std::move(info));
		return AttributeHandle{ m_attributes.size() - 1u };
	}

	// Causes force-unload on actual reserve
	// Capacity is in terms of elements, not bytes
	void reserve(std::size_t capacity) {
		if(capacity <= m_attribElemCapacity)
			return;
		this->unload<Device::CUDA>();

		std::size_t currOffset = 0u;
		// Adjust pool offsets for the attributes
		for (auto& attrib : m_attributes) {
			attrib.poolOffset = currOffset;
			currOffset += attrib.elemSize * capacity;
		}

		m_poolSize = currOffset;

		m_attribElemCapacity = capacity;
		auto& pool = m_pools.template get<PoolHandle<Device::CPU>>().handle;
		if(pool && m_poolSize > 0)
			pool = Allocator<Device::CPU>::realloc(pool, m_poolSize, currOffset);
	}

	// Resizes the attribute, leaves the memory uninitialized
	// Force-unloads non-CPU pools if reserve necessary
	void resize(std::size_t size) {
		this->reserve(size);
		m_attribElemCount = size;
	}

	// Shrinks the memory to fit the element count on all devices
	// Does not unload any device memory
	void shrink_to_fit() {
		if(m_attribElemCount == m_attribElemCapacity)
			return;

		if(m_attribElemCount != 0) {
			std::size_t bytes = m_attribElemCount * m_poolSize / m_attribElemCapacity;
			m_pools.for_each([bytes, prev = m_poolSize](auto& pool) {
				using ChangedBuffer = std::decay_t<decltype(pool)>;
				if(pool.handle)
					pool.handle = Allocator<ChangedBuffer::DEVICE>::realloc(pool.handle, prev, bytes);
			});
			m_poolSize = bytes;
		}
		else {
			m_pools.for_each([prev = m_poolSize](auto& pool) {
				using ChangedBuffer = std::decay_t<decltype(pool)>;
				if(pool.handle)
					pool.handle = Allocator<ChangedBuffer::DEVICE>::free(pool.handle, prev);
				});
			m_poolSize = 0u;
		}
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

	template < Device dev, class T >
	ArrayDevHandle_t<dev, T> acquire(AttributeHandle hdl) {
		mAssert(hdl.index < m_attributes.size());
		// Mark both the specific attribute flags and the flags that indicate a change is present
		m_attributes[hdl.index].dirty.mark_changed(dev);
		m_dirty.mark_changed(dev);
		return reinterpret_cast<T*>(&m_pools.template get<PoolHandle<dev>>().handle[m_attributes[hdl.index].poolOffset]);
	}

	template < Device dev, class T >
	ConstArrayDevHandle_t<dev, T> acquire_const(AttributeHandle hdl) {
		mAssert(hdl.index < m_attributes.size());
		this->synchronize<dev>(hdl);
		return reinterpret_cast<const T*>(&m_pools.template get<PoolHandle<dev>>().handle[m_attributes[hdl.index].poolOffset]);
	}

	template < Device dev, class T >
	ArrayDevHandle_t<dev, T> acquire(std::string_view name) {
		return acquire<dev, T>(get_attribute_handle(name));
	}

	template < Device dev, class T >
	ConstArrayDevHandle_t<dev, T> acquire_const(std::string_view name) {
		return acquire_const<dev, T>(get_attribute_handle(name));
	}

	template < Device dev >
	void synchronize() {
		if(!m_dirty.has_changes())
			return;
		if(m_dirty.has_competing_changes())
			throw std::runtime_error("Competing changes in attribute pool");
		if(!m_dirty.needs_sync(dev))
			return;	
		if (m_attribElemCount == 0)
			return;

		ArrayDevHandle_t<dev, char>& syncPool = m_pools.template get<PoolHandle<dev>>().handle;
		char** changedPool = nullptr;

		switch (dev) {
			case Device::CPU:
				// We know that we're dirty from CUDA (since no OpenGL yet)
				changedPool = &m_pools.template get<PoolHandle<Device::CUDA>>().handle;
				break;
			case Device::CUDA:
				// We know that we're dirty from CPU (since no OpenGL yet)
				changedPool = &m_pools.template get<PoolHandle<Device::CPU>>().handle;
				break;
		}

		if(changedPool) {
			if(!syncPool) {
				// If there was no pool allocated we need to copy everything anyway
				*changedPool = Allocator<dev>::alloc_array<char>(m_poolSize);
				copy(syncPool, *changedPool, m_poolSize);
			} else {
				// Selective update is enough
				std::size_t currOffset = 0u;
				for(auto& attrib : m_attributes) {
					if(attrib.dirty.needs_sync(dev)) {
						copy(syncPool + currOffset, *changedPool + currOffset, attrib.elemSize * m_attribElemCount);
						currOffset += attrib.elemSize * m_attribElemCount;
						attrib.dirty.mark_synced(dev);
					}
				}
			}
				
		}
		m_dirty.mark_synced(dev);
	}

	template < Device dev >
	void synchronize(AttributeHandle hdl) {
		mAssert(hdl.index < m_attributes.size());
		ArrayDevHandle_t<dev, char>& syncPool = m_pools.template get<PoolHandle<dev>>().handle;
		if(!syncPool)
			this->synchronize<dev>();

		if(!m_attributes[hdl.index].dirty.needs_sync(dev))
			return;

		char** changedPool = nullptr;
		switch(dev) {
			case Device::CPU:
				// We know that we're dirty from CUDA (since no OpenGL yet)
				changedPool = &m_pools.template get<PoolHandle<Device::CUDA>>().handle;
				break;
			case Device::CUDA:
				// We know that we're dirty from CPU (since no OpenGL yet)
				changedPool = &m_pools.template get<PoolHandle<Device::CPU>>().handle;
				break;
		}

		const std::size_t offset = m_attributes[hdl.index].poolOffset;
		copy(syncPool + offset, *changedPool + offset, m_attributes[hdl.index].elemSize * m_attribElemCount);
	}

	template < Device dev >
	void synchronize(std::string_view name) {
		return synchronize(get_attribute_handle(name));
	}

	template < Device dev >
	void unload() {
		// TODO: detect if we unload last pool
		auto& pool = m_pools.template get<PoolHandle<dev>>().handle;
		if(pool)
			pool = Allocator<dev>::free(pool, m_poolSize);
	}

	void mark_changed(Device dev, AttributeHandle hdl) {
		mAssert(hdl.index < m_attributes.size());
		m_dirty.mark_changed(dev);
		m_attributes[hdl.index].dirty.mark_changed(dev);
	}
	void mark_changed(Device dev, std::string_view name) {
		mark_changed(dev, get_attribute_handle(name));
	}
	void mark_changed(Device dev) {
		m_dirty.mark_changed(dev);
		for (auto& attr : m_attributes) {
			attr.dirty.mark_changed(dev);
		}
	}

	// Loads the attribute from a byte stream
	std::size_t restore(AttributeHandle hdl, util::IByteReader& attrStream,
						std::size_t start, std::size_t count) {
		mAssert(hdl.index < m_attributes.size());
		this->synchronize<Device::CPU>(hdl);
		if(start + count > m_attribElemCount)
			this->resize(start + count);

		AttribInfo& attribute = m_attributes[hdl.index];
		const std::size_t elemSize = attribute.elemSize;
		char* mem = m_pools.template get<PoolHandle<Device::CPU>>().handle + elemSize * start;
		std::size_t read = attrStream.read(mem, elemSize * count);
		if(read > 0)
			this->mark_changed(Device::CPU, hdl);
		return read / attribute.elemSize;
	}

	// Loads the attribute from a byte stream
	std::size_t restore(std::string_view name, util::IByteReader& attrStream,
						std::size_t start, std::size_t count) {
		return this->restore(get_attribute_handle(name), attrStream, start, count);
	}

	std::size_t store(AttributeHandle hdl, util::IByteWriter& attrStream,
					  std::size_t start, std::size_t count) {
		mAssert(hdl.index < m_attributes.size());
		this->synchronize<Device::CPU>(hdl);
		std::size_t actualCount = count;
		if(start + count > m_attribElemCount)
			actualCount = m_attribElemCount - start;

		AttribInfo& attribute = m_attributes[hdl.index];
		const std::size_t elemSize = attribute.elemSize;
		const char* mem = m_pools.template get<PoolHandle<Device::CPU>>().handle + elemSize * start;
		return attrStream.write(mem, elemSize * count) / attribute.elemSize;
	}

	std::size_t store(std::string_view name, util::IByteWriter& attrStream,
					  std::size_t start, std::size_t count) {
		return this->store(get_attribute_handle(name), attrStream, start, count);
	}

private:
	// Bookkeeping for attributes
	struct AttribInfo {
		std::size_t elemSize = 0u;
		std::size_t poolOffset = 0u;
		util::DirtyFlags<Device> dirty;
	};

	template < Device dev >
	struct PoolHandle {
		static constexpr Device DEVICE = dev;
		ArrayDevHandle_t<dev, char> handle = ArrayDevHandle_t<dev, char>{};
	};

	// Resolves a name to an attribute
	AttributeHandle get_attribute_handle(std::string_view name) {
		auto mapIter = m_nameMap.find(name);
		if(mapIter == m_nameMap.end())
			throw std::runtime_error("Could not find attribute '" + std::string(name) + "'");
		return AttributeHandle{ mapIter->second };
	}

	std::map<std::string, std::size_t, std::less<>> m_nameMap;
	std::size_t m_attribElemCount = 0u;
	std::size_t m_attribElemCapacity = 0u;
	std::size_t m_poolSize = 0u;
	util::TaggedTuple<PoolHandle<Device::CPU>, PoolHandle<Device::CUDA>> m_pools = {};
	// TODO: OpenGL pool?

	util::DirtyFlags<Device> m_dirty;
	std::vector<AttribInfo> m_attributes;
};

}} // namespace mufflon::scene