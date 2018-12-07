#pragma once

#include "core/export/api.h"
#include "util/assert.hpp"
#include "util/byte_io.hpp"
#include "core/memory/allocator.hpp"
#include "core/memory/residency.hpp"
#include "core/scene/geometry/polygon_mesh.hpp"
#include <OpenMesh/Core/Utils/BaseProperty.hh>
#include <OpenMesh/Core/Utils/Property.hh>
#include <functional>
#include <istream>
#include <ostream>

namespace mufflon { namespace scene {

namespace pool_detail {

// Helper struct indicating an attribute's offset and length within the memory block (in bytes)
struct MemoryArea {
	std::size_t offset = std::numeric_limits<std::size_t>::max();
	std::size_t length = 0u;
	std::size_t elemSize = 0u;
};

} // namespace pool_detail

// Attribute pool implementation for devices which use a malloc/realloc/free pattern
template < class Alloc >
class MallocAttributePool {
public:
	template < class A >
	friend class MallocAttributePool;

	using Allocator = Alloc;

	MallocAttributePool() = default;
	MallocAttributePool(const MallocAttributePool&) = delete;
	MallocAttributePool(MallocAttributePool&& pool) :
		m_attribs(std::move(pool.m_attribs)),
		m_size(pool.m_size),
		m_attribLength(pool.m_attribLength),
		m_present(pool.m_present),
		m_memoryBlock(pool.m_memoryBlock) {
		pool.m_memoryBlock = nullptr;
	}
	MallocAttributePool& operator=(const MallocAttributePool&) = delete;
	MallocAttributePool& operator=(MallocAttributePool&& pool) {
		m_attribs = pool.m_attribs;
		m_size = pool.m_size;
		m_attribLength = pool.m_attribLength;
		m_present = pool.m_present;
		std::swap(m_memoryBlock, pool.m_memoryBlock);
	}
	virtual ~MallocAttributePool() {
		if(m_memoryBlock != nullptr)
			Allocator::free(m_memoryBlock, m_size);
	}

	// Helper class identifying an attribute in the pool
	template < class T >
	struct AttributeHandle {
		using Type = T;
		friend class MallocAttributePool<Allocator>;

		AttributeHandle() :
			m_index(std::numeric_limits<std::size_t>::max()) {}

	private:
		AttributeHandle(std::size_t idx) :
			m_index(idx) {}

		constexpr std::size_t index() const noexcept {
			return m_index;
		}

		std::size_t m_index;
	};

	// Adds a new attribute to the pool
	template < class T >
	AttributeHandle<T> add() {
		// Increase size of memory pool
		std::size_t newBytes = sizeof(T) * m_attribLength;
		pool_detail::MemoryArea area{ m_size, newBytes, sizeof(T) };
		this->realloc(m_size + newBytes);

		// Save attribute offset and length at a free spot
		auto iter = m_attribs.begin();
		for(; iter != m_attribs.end(); ++iter) {
			if(iter->offset == std::numeric_limits<std::size_t>::max()) {
				(*iter) = area;
				break;
			}
		}
		if(iter == m_attribs.end()) {
			// No empty spot -> append
			m_attribs.push_back(area);
		}

		return AttributeHandle<T>{ m_attribs.size() - 1u };
	}

	// Removes the given attribute from the pool
	template < class T >
	bool remove(const AttributeHandle<T>& hdl) {
		if(hdl.index() >= m_attribs.size())
			return false;
		if(m_attribs[hdl.index()].offset == std::numeric_limits<std::size_t>::max())
			return false;

		// Shrink the memory pool
		std::size_t removedBytes = m_attribs[hdl.index()].length;
		mAssert(removedBytes <= m_size);

		// Either entirely remove or mark as hole
		if(hdl.index() == m_attribs.size() - 1u) {
			// Is at the back -> we can reallocate
			m_attribs.pop_back();
			this->realloc(m_size - removedBytes);
		} else {
			if(m_present) {
				// We might need to shift the following attributes
				if(removedBytes == m_size) {
					// Nothing left (or nothing there to begin with) -> unload
					this->unload();
				} else {
					// Loop once to find largest attribute
					std::size_t largest = 0u;
					for(std::size_t i = hdl.index() + 1u; i < m_attribs.size(); ++i) {
						if(m_attribs[i].offset != std::numeric_limits<std::size_t>::max())
							largest = std::max(largest, m_attribs[i].length);
					}
					// Since we're not last there must be one more element that's larger
					mAssert(largest > 0u);

					// Allocate a swap buffer for the realloc
					char* swapBlock = Allocator::template alloc_array<char>(largest);
					// Start shifting the later attributes
					std::size_t currOffset = m_attribs[hdl.index()].offset;
					for(std::size_t i = hdl.index() + 1u; i < m_attribs.size(); ++i) {
						// Swap the attribute into the temporary buffer, then copy it to its new location
						if(m_attribs[i].offset != std::numeric_limits<std::size_t>::max()) {
							Allocator::copy(swapBlock, &m_memoryBlock[m_attribs[i].offset], m_attribs[i].length);
							Allocator::copy(&m_memoryBlock[currOffset], swapBlock, m_attribs[i].length);
							m_attribs[i].offset = currOffset;
							currOffset += m_attribs[i].length;
						}
					}

					// Clean up
					Allocator::free(swapBlock, largest);
					m_size -= removedBytes;
				}
			} else {
				// "Pretend" to realloc even though we'll just save the size
				this->realloc(m_size - removedBytes);
			}
			m_attribs[hdl.index()] = { std::numeric_limits<std::size_t>::max(), 0u };
		}

		return true;
	}

	// Aquires the given attribute for direct access
	template < class T >
	T* aquire(const AttributeHandle<T>& hdl) {
		if(hdl.index() >= m_attribs.size())
			return nullptr;
		if(m_attribs[hdl.index()].offset == std::numeric_limits<std::size_t>::max())
			return nullptr;

		return reinterpret_cast<T*>(&m_memoryBlock[m_attribs[hdl.index()].offset]);
	}

	// Aquires the given attribute for read-only access
	template < class T >
	const T* aquireConst(const AttributeHandle<T>& hdl) const {
		if(hdl.index() >= m_attribs.size())
			return nullptr;
		if(m_attribs[hdl.index()].offset == std::numeric_limits<std::size_t>::max())
			return nullptr;

		return reinterpret_cast<const T*>(&m_memoryBlock[m_attribs[hdl.index()].offset]);
	}

	// Resizes all attributes to the given length
	void resize(std::size_t length) {
		if(length != m_attribLength) {
			// Check if we had non-zero size before - if not, we need to compute it
			std::size_t newSize = 0u;
			if(m_attribLength == 0) {
				if(length != 0u) {
					for(const auto& area : m_attribs) {
						if(area.offset != std::numeric_limits<std::size_t>::max())
							newSize += area.elemSize;
					}
					newSize *= length;
				}
			} else {
				newSize = length * (m_size / m_attribLength);
			}

			if(newSize == 0u) {
				// Special case - remove all elements
				for(auto& area : m_attribs) {
					if(area.offset != std::numeric_limits<std::size_t>::max()) {
						area.offset = 0u;
						area.length = 0u;
					}
				}
				if(m_memoryBlock != nullptr) {
					delete[] m_memoryBlock;
					m_memoryBlock = nullptr;
				}
			} else {
				char* newBlock = nullptr;
				bool realloced = false;
				if(m_present) {
					// Can only reallocate if we're growing in size
					if(length > m_attribLength && m_memoryBlock != nullptr) {
						newBlock = Allocator::realloc(m_memoryBlock, m_size, newSize);
						realloced = true;
					} else {
						newBlock = Allocator::template alloc_array<char>(newSize);
					}
				}

				// Loop all attributes to shift them , but in reverse to ease copying
				std::size_t currEnd = newSize;
				for(auto iter = m_attribs.rbegin(); iter != m_attribs.rend(); ++iter) {
					if(iter->offset != std::numeric_limits<std::size_t>::max()) {
						const std::size_t elemSize = iter->elemSize;
						mAssert(elemSize != 0u);
						currEnd -= elemSize * length;
						// Copy the still-valid attribute values
						if(m_present && m_size != 0) {
							mAssert(newBlock != nullptr);
							mAssert(m_memoryBlock != nullptr);
							Allocator::copy(&newBlock[currEnd], &m_memoryBlock[iter->offset], iter->length);
						}
						iter->length = elemSize * length;
						iter->offset = currEnd;
					}
				}

				if(!realloced && m_memoryBlock != nullptr)
					Allocator::free(m_memoryBlock, m_size);
				m_memoryBlock = newBlock;
				m_attribLength = length;
				m_size = newSize;
			}
		}
	}

	// Read the attribute between start and start+count from the stream
	template < class T >
	std::size_t restore(const AttributeHandle<T>& hdl, util::IByteReader& stream,
						std::size_t start, std::size_t count) {
		mAssert(hdl.index() < m_attribs.size());
		if(start >= m_attribLength)
			return 0u;
		if(m_attribs[hdl.index()].offset == std::numeric_limits<std::size_t>::max())
			return 0u;
		std::size_t bytes = std::min(sizeof(T) * count,
									 m_attribs[hdl.index()].length - sizeof(T)*start);
		return stream.read(&m_memoryBlock[m_attribs[hdl.index()].offset + sizeof(T)*start],
						   bytes) / sizeof(T);
	}

	// Write the attribute to the stream
	template < class T >
	std::size_t store(const AttributeHandle<T>& hdl, util::IByteWriter& stream,
					  std::size_t start, std::size_t count) const {
		mAssert(hdl.index() < m_attribs.size());
		if(start >= m_attribLength)
			return 0u;
		if(m_attribs[hdl.index()].offset == std::numeric_limits<std::size_t>::max())
			return 0u;
		std::size_t bytes = std::min(sizeof(T) * count,
									 m_attribs[hdl.index()].length - sizeof(T)*start);
		return stream.write(&m_memoryBlock[m_attribs[hdl.index()].offset + sizeof(T)*start],
							bytes) / sizeof(T);
	}

	// Unloads the attribute pool from the device
	void unload() {
		if(m_memoryBlock != nullptr)
			m_memoryBlock = Allocator::free(m_memoryBlock, m_size);
		m_present = false;
	}

	// Returns the length of the attributes
	std::size_t get_size() const noexcept {
		return m_attribLength;
	}

	// Returns the current size of the pool in bytes
	std::size_t get_byte_count() const noexcept {
		return m_size;
	}

	// Returns whether the pool is currently allocated
	bool is_present() const noexcept {
		return m_present;
	}

	// Makes the memory buffer present (but does not sync!)
	void make_present() {
		// Since attributes are always changed for all pools, we only need to allocate the necessary space
		if(!m_present) {
			mAssert(m_memoryBlock == nullptr);
			m_memoryBlock = Allocator::template alloc_array<char>(m_size);
			m_present = true;
		}
	}

protected:
	// Gain access to the raw memory block for syncing
	char* get_pool_data() {
		return m_memoryBlock;
	}

	const char* get_pool_data() const {
		return m_memoryBlock;
	}

	// Mark the pool as being present on the device again
	void mark_present() {
		m_present = true;
	}

private:
	// Resizes the internal memory chunk with respect to presence and such
	void realloc(std::size_t newSize) {
		// Only resize if we're present on the device
		if(m_present) {
			if(newSize == 0u) {
				// Remove, but don't mark as unloaded since that is an explicit intent!
				if(m_memoryBlock != nullptr)
					m_memoryBlock = Allocator::free(m_memoryBlock, m_size);
			} else if(newSize != m_size) {
				if(m_memoryBlock == nullptr)
					m_memoryBlock = Allocator::template alloc_array<char>(newSize);
				else
					m_memoryBlock = Allocator::realloc(m_memoryBlock, m_size, newSize);
			}
		}
		m_size = newSize;
	}

	std::vector<pool_detail::MemoryArea> m_attribs;
	std::size_t m_size = 0u;
	std::size_t m_attribLength = 0u;
	bool m_present = false;
	char* m_memoryBlock = nullptr;
};

/**
 * Abstracts a memory pool for attribute storage.
 * Needs to be specialized for every device due to the different memory systems.
 * Should guarantee contiguous storage whenever possible.
 * dev: Device on which the pool is located
 */
template < Device dev >
class AttributePool;

// Forward declaration of 
template < bool isFace >
class OmAttributePool;

// Specialization for CPU without OpenMesh
template <>
class AttributePool<Device::CPU> : public MallocAttributePool<Allocator<Device::CPU>> {
public:
	static constexpr Device DEVICE = Device::CPU;
	template < Device dev >
	friend class AttributePool;
	template < class A >
	friend class MallocAttributePool;

	AttributePool() = default;
	AttributePool(const AttributePool&) = delete;
	AttributePool(AttributePool&&) = default;
	AttributePool& operator=(const AttributePool&) = delete;
	AttributePool& operator=(AttributePool&&) = default;
	~AttributePool() = default;

	template < Device dev >
	void synchronize(AttributePool<dev>& pool) {
		(void)pool;
		// Needs to be specialized on a per-device basis
		throw std::runtime_error("This synchronization is not implemented yet");
	}
};

// Specialized attribute pool for CUDA
template <>
class AttributePool<Device::CUDA> : public MallocAttributePool<Allocator<Device::CUDA>> {
public:
	static constexpr Device DEVICE = Device::CUDA;
	template < Device dev >
	friend class AttributePool;
	template < bool face >
	friend class OmAttributePool;
	template < class A >
	friend class MallocAttributePool;

	AttributePool() = default;
	AttributePool(const AttributePool&) = delete;
	AttributePool(AttributePool&&) = default;
	AttributePool& operator=(const AttributePool&) = delete;
	AttributePool& operator=(AttributePool&&) = default;
	~AttributePool() = default;

	template < Device dev >
	void synchronize(AttributePool<dev>& pool) {
		(void)pool;
		// Needs to be specialized on a per-device basis
		throw std::runtime_error("This synchronization is not implemented yet");
	}

	template < bool isFace >
	void synchronize(OmAttributePool<isFace>& pool);
};

// Specialization for CPU with OpenMesh
template < bool isFace >
class OmAttributePool {
public:
	static constexpr Device DEVICE = Device::CPU;
	static constexpr bool IS_FACE = isFace;
	template < class T >
	using PropType = std::conditional_t<IS_FACE, OpenMesh::FPropHandleT<T>, OpenMesh::VPropHandleT<T>>;
	template < Device dev >
	friend class AttributePool;
	template < class A >
	friend class MallocAttributePool;

	OmAttributePool(geometry::PolygonMeshType& mesh) :
		m_mesh(mesh)
	{}
	OmAttributePool(const OmAttributePool&) = delete;
	OmAttributePool(OmAttributePool&&) = default;
	OmAttributePool& operator=(const OmAttributePool&) = delete;
	OmAttributePool& operator=(OmAttributePool&&) = delete;
	~OmAttributePool() = default;

	// Helper class identifying an attribute in the pool
	template < class T >
	struct AttributeHandle {
		using Type = T;
		friend class OmAttributePool<IS_FACE>;

		AttributeHandle() :
			m_index(std::numeric_limits<std::size_t>::max()) {}

	private:
		AttributeHandle(std::size_t idx) :
			m_index(idx) {}

		constexpr std::size_t index() const noexcept {
			return m_index;
		}

		std::size_t m_index;
	};

	// Adds a new vertex attribute to the pool
	template < class T >
	AttributeHandle<T> add(PropType<T> hdl) {
		auto& prop = m_mesh.property(hdl);
		// Ensure that the property is at the same size
		prop.resize(m_attribLength);

		// Look for previously removed attribute spots
		for(std::size_t i = 0u; i < m_attributes.size(); ++i) {
			if(!m_attributes[i].is_valid()) {
				m_attributes[i] = OpenMesh::BaseHandle{ hdl.idx() };
				return AttributeHandle<T>{ i };
			}
		}
		m_attributes.push_back(OpenMesh::BaseHandle{ hdl.idx() });
		// Create an accessor that can use the type info
		m_accessors.push_back([](OpenMesh::BaseHandle attr, geometry::PolygonMeshType& mesh) {
			auto& prop = mesh.property(PropType<T> { attr.idx() });
			return std::make_pair<char*, std::size_t>(reinterpret_cast<char*>(prop.data_vector().data()),
				prop.size_of());
		});

		// Increase the combined element size
		m_accumElemSize += sizeof(T);

		// No hole found
		return AttributeHandle<T>{ m_attributes.size() - 1u };
	}

	// Removes the given attribute from the pool
	template < class T >
	bool remove(const AttributeHandle<T>& hdl) {
		if(hdl.index() >= m_attributes.size())
			return false;
		if(!m_attributes[hdl.index()].is_valid())
			return false;

		// Either entirely remove or mark as hole
		if(hdl.index() == m_attributes.size() - 1u) {
			m_attributes.pop_back();
			m_accessors.pop_back();
		} else {
			m_attributes[hdl.index()].invalidate();
			// Don't bother removing the accessor
		}

		return true;
	}

	// Aquires the given attribute for direct access
	template < class T >
	T* aquire(const AttributeHandle<T>& hdl) {
		if(hdl.index() >= m_attributes.size())
			return nullptr;
		if(!m_attributes[hdl.index()].is_valid())
			return nullptr;

		return m_mesh.property(PropType<T>{ m_attributes[hdl.index()].idx() }).data_vector().data();
	}

	// Aquires the given attribute for read-only access
	template < class T >
	const T* aquireConst(const AttributeHandle<T>& hdl) const {
		if(hdl.index() >= m_attributes.size())
			return nullptr;
		if(!m_attributes[hdl.index()].is_valid())
			return nullptr;

		return m_mesh.property(PropType<T> { m_attributes[hdl.index()].idx() }).data_vector().data();
	}

	// Resizes all attributes to the given length
	void resize(std::size_t length) {
		if constexpr(IS_FACE)
			m_mesh.resize(m_mesh.n_vertices(), m_mesh.n_edges(), length);
		else
			m_mesh.resize(length, m_mesh.n_edges(), m_mesh.n_faces());
		m_attribLength = length;
	}

	// Read the attribute between start and start+count from the stream
	template < class T >
	std::size_t restore(const AttributeHandle<T>& hdl, util::IByteReader& stream,
						std::size_t start, std::size_t count) {
		mAssert(hdl.index() < m_attributes.size());
		if(start >= m_attribLength)
			return 0u;
		if(!m_attributes[hdl.index()].is_valid())
			return 0u;

		T* data = m_mesh.property(PropType<T>{ m_attributes[hdl.index()].idx() }).data_vector().data();
		std::size_t bytes = sizeof(T) * (std::min(count, m_attribLength - start));
		return stream.read(reinterpret_cast<char*>(data), bytes) / sizeof(T);
	}

	// Write the attribute to the stream
	template < class T >
	std::size_t store(const AttributeHandle<T>& hdl, util::IByteWriter& stream,
					  std::size_t start, std::size_t count) const {
		mAssert(hdl.index() < m_attributes.size());
		if(start >= m_attribLength)
			return 0u;
		if(!m_attributes[hdl.index()].hdl.is_valid())
			return 0u;

		const T* data = m_mesh.property(PropType<T> { m_attributes[hdl.index()].idx() }).data_vector().data();
		std::size_t bytes = sizeof(T) * (std::min(count, m_attribLength - start));
		return stream.write(reinterpret_cast<char*>(data), bytes) / sizeof(T);
	}

	// Unloads the attribute pool from the device
	void unload() {
		// TODO: is this even possible?
	}

	template < Device dev, class = std::enable_if_t<dev != Device::CPU> >
	void synchronize(AttributePool<dev>& pool) {
		std::size_t currOffset = 0u;
		pool.make_present();

		// Loop to copy the attributes
		for(std::size_t i = 0u; i < m_attributes.size(); ++i) {
			auto attrib = m_attributes[i];
			if(attrib.is_valid()) {
				// Copy the current attribute into the buffer
				auto[propPtr, currLength] = m_accessors[i](attrib, m_mesh);
				if constexpr(dev == Device::CUDA)
					Allocator<Device::CPU>::template copy<char, Device::CUDA>(&pool.get_pool_data()[currOffset],
																			  propPtr, currLength);
				else
					throw std::runtime_error("Missing OpenGL copy");
				currOffset += currLength;
			}
		}
	}

	// Returns the length of the attributes
	std::size_t get_size() const noexcept {
		return m_attribLength;
	}

	// Returns the current size of the pool in bytes
	std::size_t get_byte_count() const noexcept {
		return m_accumElemSize * m_attribLength;
	}

	// Returns whether the pool is currently allocated
	bool is_present() const noexcept {
		// TODO
		return true;
	}

	void make_present() {
		// TODO
	}

private:
	geometry::PolygonMeshType& m_mesh;
	std::size_t m_attribLength = 0u;
	std::size_t m_accumElemSize = 0u;
	std::vector<OpenMesh::BaseHandle> m_attributes; // Attribute handle, later to be retrieved
	std::vector<std::function<std::pair<char*, std::size_t>(OpenMesh::BaseHandle,
															geometry::PolygonMeshType&)>> m_accessors; // Accessors to attribute
};

// Function overloads for "unified" call syntay
template < Device changedDev, Device syncDev >
void synchronize(AttributePool<changedDev>& changed, AttributePool<syncDev>& sync) {
	changed.template synchronize<>(sync);
}
template < Device changedDev, bool isFace >
void synchronize(AttributePool<changedDev>& changed, OmAttributePool<isFace>& sync) {
	changed.template synchronize<>(sync);
}
template < bool isFace, Device syncedDev >
void synchronize(OmAttributePool<isFace>& changed, AttributePool<syncedDev>& sync) {
	changed.template synchronize<>(sync);
}
template < bool isFace, bool syncFace >
void synchronize(OmAttributePool<isFace>& changed, OmAttributePool<syncFace>& sync) {
	throw std::runtime_error("Attempted synchronization between same devices");
}

template <>
void AttributePool<Device::CPU>::synchronize<Device::CUDA>(AttributePool<Device::CUDA>& pool);
template <>
void AttributePool<Device::CUDA>::synchronize<Device::CPU>(AttributePool<Device::CPU>& pool);

// Synchronization specialization from CUDA to CPU (non-owning)
template < bool isFace >
void AttributePool<Device::CUDA>::synchronize(OmAttributePool<isFace>& pool) {
	pool.make_present();
	std::size_t currOffset = 0u;

	// Loop to copy the attributes
	for(std::size_t i = 0u; i < pool.m_attributes.size(); ++i) {
		auto attrib = pool.m_attributes[i];
		if(attrib.is_valid()) {
			// Copy from the contiguous buffer into the attributes
			auto[propPtr, currLength] = pool.m_accessors[i](attrib, pool.m_mesh);
			Allocator::copy<char, Device::CPU>(propPtr, &this->get_pool_data()[currOffset], currLength);
			currOffset += currLength;
		}
	}
}

}} // namespace mufflon::scene