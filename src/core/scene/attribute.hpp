#pragma once

#include "allocator.hpp"
#include "residency.hpp"
#include "util/array_wrapper.hpp"
#include <OpenMesh/Core/Utils/BaseProperty.hh>
#include <OpenMesh/Core/Utils/Property.hh>
#include <istream>
#include <ostream>

namespace mufflon::scene {

// Attribute pool implementation for devices which use a malloc/realloc/free pattern
template < class Allocator >
class MallocAttributePool {
public:
	virtual ~MallocAttributePool() {
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
		MemoryArea area{ m_size, newBytes };
		m_memoryBlock = Allocator::realloc(m_memoryBlock, m_size, m_size + newBytes);
		m_size += newBytes;

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
		if(hdl.index >= m_attribs.size())
			return false;
		if(m_attribs[hdl.index].offset == std::numeric_limits<std::size_t>::max())
			return false;

		// Shrink the memory pool
		std::size_t removedBytes = m_attribs[hdl.index].length;
		mAssert(removedBytes <= m_size);
		m_memoryBlock = Allocator::realloc(m_memoryBlock, m_size, m_size - removedBytes);
		m_size -= removedBytes;

		// Either entirely remove or mark as hole
		if(hdl.index == m_attribs.size() - 1u)
			m_attribs.pop_back();
		else
			m_attribs[hdl.index] = { std::numeric_limits<std::size_t>::max(), 0u };

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
			std::size_t newSize = length * (m_size / m_attribLength);
			char* newBlock = nullptr;
			if(m_present) {
				// Can only reallocate if we're growing in size
				if(length > m_attribLength) {
					m_memoryBlock = Allocator::realloc(m_memoryBlock, m_size, newSize);
					newBlock = m_memoryBlock;
				} else {
					newBlock = Allocator::alloc(newSize);
				}
				m_size = newSize;
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

			m_attribLength = length;
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

private:
	// Helper struct indicating an attribute's offset and length within the memory block (in bytes)
	struct MemoryArea {
		std::size_t offset = std::numeric_limits<std::size_t>::max();
		std::size_t length = 0u;
	};

	std::vector<MemoryArea> m_attribs;
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
		// No hole found
		return AttributeHandle<T>{ m_attributes.size() - 1u };
	}

	// Removes the given attribute from the pool
	template < class T >
	bool remove(const AttributeHandle<T>& hdl) {
		if(hdl.index >= m_attributes.size())
			return false;
		if(m_attributes[hdl.index].offset == std::numeric_limits<std::size_t>::max())
			return false;

		// Either entirely remove or mark as hole
		if(hdl.index == m_attributes.size() - 1u)
			m_attributes.pop_back();
		else
			m_attributes[hdl.index] = nullptr;

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
	// Non-owning pointer to OpenMesh attribute
	std::size_t m_attribLength;
	std::vector<OpenMesh::BaseProperty*> m_attributes;
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