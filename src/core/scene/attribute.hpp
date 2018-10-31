#pragma once

#include "allocator.hpp"
#include "residency.hpp"
#include "util/array_wrapper.hpp"
#include <OpenMesh/Core/Utils/BaseProperty.hh>
#include <OpenMesh/Core/Utils/Property.hh>
#include <functional>
#include <istream>
#include <ostream>

namespace mufflon::scene {

namespace pool_detail {

// Helper struct indicating an attribute's offset and length within the memory block (in bytes)
struct MemoryArea {
	std::size_t offset = std::numeric_limits<std::size_t>::max();
	std::size_t length = 0u;
	std::size_t elem_size = 0u;
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
	virtual ~MallocAttributePool() {
		if(m_memoryBlock != nullptr)
			Allocator::free(m_memoryBlock, m_size);
	}

	// Helper class identifying an attribute in the pool
	template < class T >
	struct AttributeHandle {
		using Type = T;
		friend class MallocAttributePool;

		AttributeHandle() :
			m_index(std::numeric_limits<std::size_t>::max()) {}

	private:
		AttributeHandle(std::size_t idx) :
			m_index(idx)
		{}

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
		if(hdl.index == m_attribs.size() - 1u) {
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
					char* swapBlock = Allocator::template alloc<char>(largest);
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
	util::ArrayWrapper<T> aquire(const AttributeHandle<T>& hdl) {
		if(hdl.index() >= m_attribs.size())
			return nullptr;
		if(m_attribs[hdl.index()].offset == std::numeric_limits<std::size_t>::max())
			return nullptr;

		return util::ArrayWrapper<T>{ reinterpret_cast<T*>(m_memoryBlock), m_attribs[hdl.index()].length / sizeof(T) };
	}

	// Aquires the given attribute for read-only access
	template < class T >
	util::ConstArrayWrapper<T> aquireConst(const AttributeHandle<T>& hdl) const {
		if(hdl.index() >= m_attribs.size())
			return nullptr;
		if(m_attribs[hdl.index()].offset == std::numeric_limits<std::size_t>::max())
			return nullptr;

		return util::ConstArrayWrapper<T>{ reinterpret_cast<T*>(m_memoryBlock), m_attribs[hdl.index()].length / sizeof(T) };
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
							newSize += area.elem_size;
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
				this->unload();
			} else {
				char* newBlock = nullptr;
				if(m_present) {
					// Can only reallocate if we're growing in size
					if(length > m_attribLength) {
						newBlock = Allocator::realloc(m_memoryBlock, m_size, newSize);
					} else {
						newBlock = Allocator::alloc(newSize);
					}
				}

				// Loop all attributes to shift them , but in reverse to ease copying
				std::size_t currEnd = newSize;
				for(auto iter = m_attribs.rbegin(); iter != m_attribs.rend(); ++iter) {
					if(iter->offset != std::numeric_limits<std::size_t>::max()) {
						const std::size_t elemSize = iter->length / m_attribLength;
						mAssert(elemSize != 0u);
						iter->length = elemSize * length;
						currEnd -= iter->length;
						// Copy the still-valid attribute values
						if(m_present) {
							mAssert(newBlock != nullptr);
							mAssert(m_memoryBlock != nullptr);
							Allocator::copy(&newBlock[currEnd], &m_memoryBlock[iter->offset], iter->length);
						}
						iter->offset = currEnd;
					}
				}

				if(newBlock != m_memoryBlock)
					Allocator::free(m_memoryBlock, m_size);
				m_memoryBlock = newBlock;
				m_attribLength = length;
				m_size = newSize;
			}
		}
	}

	// Read the attribute between start and start+count from the stream
	template < class T >
	std::size_t restore(const AttributeHandle<T>& hdl, std::istream& stream, std::size_t start, std::size_t count) {
		mAssert(hdl.index() < m_attribs.size());
		if(start >= m_attribLength)
			return 0u;
		if(m_attribs[hdl.index()].offset == std::numeric_limits<std::size_t>::max())
			return 0u;
		std::size_t bytes = std::min(sizeof(T) * count, m_attribs[hdl.index()].length) - sizeof(T)*start;
		stream.read(&m_memoryBlock[m_attribs[hdl.index()].offset], bytes);
		return static_cast<std::size_t>(stream.gcount()) / sizeof(T);
	}

	// Write the attribute to the stream
	template < class T >
	std::size_t store(const AttributeHandle<T>& hdl, std::ostream& stream, std::size_t start, std::size_t count) const {
		mAssert(hdl.index() < m_attribs.size());
		if(start >= m_attribLength)
			return 0u;
		if(m_attribs[hdl.index()].offset == std::numeric_limits<std::size_t>::max())
			return 0u;
		std::size_t bytes = std::min(sizeof(T) * count, m_attribs[hdl.index()].length) - sizeof(T)*start;
		stream.write(&m_memoryBlock[m_attribs[hdl.index()].offset], bytes);
		return bytes / sizeof(T);
	}

	// Unloads the attribute pool from the device
	void unload() {
		if(m_memoryBlock != nullptr) {
			Allocator::free(m_memoryBlock, m_size);
			m_memoryBlock = nullptr;
		}
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

protected:
	// Makes the memory buffer present (but does not sync!)
	void make_present() {
		// Since attributes are always changed for all pools, we only need to allocate the necessary space
		if(!m_present) {
			mAssert(m_memoryBlock == nullptr);
			m_memoryBlock = Allocator::template alloc<char>(m_size);
			m_present = true;
		}
	}

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
				// Remove from device
				this->unload();
			} else if(newSize != m_size) {
				if(m_memoryBlock == nullptr)
					m_memoryBlock = Allocator::template alloc<char>(newSize);
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
 * owning: false if the pool only stores references and no actual memory itself.
 */
template < Device dev, bool owning = true >
class AttributePool;

// Specialization for CPU without OpenMesh
template <>
class AttributePool<Device::CPU, true> : public MallocAttributePool<Allocator<Device::CPU>> {
public:
	static constexpr Device DEVICE = Device::CPU;
	static constexpr bool OWNING = true;
	template < Device dev, bool owning >
	friend class AttributePool;
	template < class A >
	friend class MallocAttributePool;

	template < Device dev, bool owning = true >
	void synchronize(AttributePool<dev, owning>& pool) {
		if constexpr(dev != DEVICE || owning != OWNING) {
			// Needs to be specialized on a per-device basis
			static_assert(false, "Missing specialization for between-device synchronization");
		}
	}
};

// Specialized attribute pool for CUDA
template <>
class AttributePool<Device::CUDA, true> : public MallocAttributePool<Allocator<Device::CUDA>> {
public:
	static constexpr Device DEVICE = Device::CUDA;
	static constexpr bool OWNING = true;
	template < Device dev, bool owning >
	friend class AttributePool;
	template < class A >
	friend class MallocAttributePool;

	template < Device dev, bool owning = true >
	void synchronize(AttributePool<dev, owning>& pool) {
		if constexpr(dev != DEVICE || owning != OWNING) {
			// Needs to be specialized on a per-device basis
			static_assert(false, "Missing specialization for between-device synchronization");
		}
	}
};

// Specialization for CPU with OpenMesh
template <>
class AttributePool<Device::CPU, false> {
public:
	static constexpr Device DEVICE = Device::CPU;
	static constexpr bool OWNING = false;
	template < Device dev, bool owning >
	friend class AttributePool;
	template < class A >
	friend class MallocAttributePool;

	// Helper class identifying an attribute in the pool
	template < class T >
	struct AttributeHandle {
		using Type = T;
		friend class AttributePool<Device::CPU, false>;

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
	AttributeHandle<T> add(OpenMesh::BaseProperty& prop) {
		// Ensure that the property is at the same size
		prop.resize(m_attribLength);

		// Look for previously removed attribute spots
		for(std::size_t i = 0u; i < m_attributes.size(); ++i) {
			if(m_attributes[i] == nullptr) {
				m_attributes[i] = &prop;
				return AttributeHandle<T>{ i };
			}
		}
		m_attributes.push_back(&prop);

		// Create an accessor that can use the type info
		m_accessors.push_back([](OpenMesh::BaseProperty& prop) {
			return reinterpret_cast<char*>(dynamic_cast<OpenMesh::PropertyT<T>&>(prop).data_vector().data());
		});

		// No hole found
		return AttributeHandle<T>{ m_attributes.size() - 1u };
	}

	// Removes the given attribute from the pool
	template < class T >
	bool remove(const AttributeHandle<T>& hdl) {
		if(hdl.index() >= m_attributes.size())
			return false;
		if(m_attributes[hdl.index()].offset == std::numeric_limits<std::size_t>::max())
			return false;

		// Either entirely remove or mark as hole
		if(hdl.index == m_attributes.size() - 1u) {
			m_attributes.pop_back();
			m_accessors.pop_back();
		} else {
			m_attributes[hdl.index()] = nullptr;
			// Don't bother removing the accessor
		}

		return true;
	}

	// Aquires the given attribute for direct access
	template < class T >
	util::ArrayWrapper<T> aquire(const AttributeHandle<T>& hdl) {
		if(hdl.index() >= m_attributes.size())
			return nullptr;
		if(m_attributes[hdl.index()] == nullptr)
			return nullptr;

		auto& prop = dynamic_cast<typename OpenMesh::PropertyT<T>&>(*m_attributes[hdl.index()]);
		return util::ArrayWrapper<T>{ prop.data_vector().data(), prop.n_elements() };
	}

	// Aquires the given attribute for read-only access
	template < class T >
	util::ConstArrayWrapper<T> aquireConst(const AttributeHandle<T>& hdl) const {
		if(hdl.index() >= m_attributes.size())
			return nullptr;
		if(m_attributes[hdl.index()] == nullptr)
			return nullptr;

		const OpenMesh::PropertyT<T>& prop = dynamic_cast<const OpenMesh::PropertyT<T>&>(*m_attributes[hdl.index()]);
		return util::ConstArrayWrapper<T>{ prop.data_vector().data(), prop.n_elements() };
	}

	// Resizes all attributes to the given length
	void resize(std::size_t length) {
		for(auto& attrib : m_attributes)
			attrib->resize(length);
		m_attribLength = length;
	}

	// Read the attribute between start and start+count from the stream
	template < class T >
	std::size_t restore(const AttributeHandle<T>& hdl, std::istream& stream, std::size_t start, std::size_t count) {
		mAssert(hdl.index() < m_attributes.size());
		if(start >= m_attribLength)
			return 0u;
		if(m_attributes[hdl.index()]== nullptr)
			return 0u;
		auto& prop = dynamic_cast<OpenMesh::PropertyT<T>&>(*m_attributes[hdl.index()]);
		std::size_t bytes = sizeof(T) * (std::min(count, m_attribLength) - start);
		stream.read(reinterpret_cast<char*>(prop.data_vector().data()), bytes);
		return static_cast<std::size_t>(stream.gcount()) / sizeof(T);
	}

	// Write the attribute to the stream
	template < class T >
	std::size_t store(const AttributeHandle<T>& hdl, std::ostream& stream, std::size_t start, std::size_t count) const {
		mAssert(hdl.index() < m_attributes.size());
		if(start >= m_attribLength)
			return 0u;
		if(m_attributes[hdl.index()] == nullptr)
			return 0u;
		const auto& prop = dynamic_cast<OpenMesh::PropertyT<T>&>(*m_attributes[hdl.index()]);
		std::size_t bytes = sizeof(T) * (std::min(count, m_attribLength) - start);
		stream.write(reinterpret_cast<const char*>(prop.data_vector().data()), bytes);
		return bytes / sizeof(T);
	}

	// Unloads the attribute pool from the device
	void unload() {
		// TODO: is this even possible?
		throw std::runtime_error("Unloading OpenMesh data is not supported yet");
	}

	template < Device dev, bool owning = true >
	void synchronize(AttributePool<dev, owning>& pool) {
		if constexpr(dev != DEVICE || owning != OWNING) {
			// Needs to be specialized on a per-device basis
			static_assert(false, "Missing specialization for between-device synchronization");
		}
	}

	// Returns the length of the attributes
	std::size_t get_size() const noexcept {
		return m_attribLength;
	}

	// Returns the current size of the pool in bytes
	std::size_t get_byte_count() const noexcept {
		std::size_t bytes = 0u;
		for(auto attribute : m_attributes) {
			if(attribute != nullptr)
				bytes += attribute->size_of();
		}
		return bytes;
	}

	// Returns whether the pool is currently allocated
	bool is_present() const noexcept {
		// TODO
		return true;
	}

private:
	std::size_t m_attribLength;
	std::vector<OpenMesh::BaseProperty*> m_attributes; // Non-owning pointer to OpenMesh attribute
	std::vector<std::function<char*(OpenMesh::BaseProperty&)>> m_accessors; // Accessors to attribute
};


template <>
void AttributePool<Device::CPU, true>::synchronize<Device::CUDA, true>(AttributePool<Device::CUDA, true>& pool);
template <>
void AttributePool<Device::CPU, false>::synchronize<Device::CUDA, true>(AttributePool<Device::CUDA, true>& pool);
template <>
void AttributePool<Device::CUDA, true>::synchronize<Device::CPU, true>(AttributePool<Device::CPU, true>& pool);
template <>
void AttributePool<Device::CUDA, true>::synchronize<Device::CPU, false>(AttributePool<Device::CPU, false>& pool);

} // namespace mufflon::scene