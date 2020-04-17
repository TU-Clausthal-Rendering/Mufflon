#include "attribute.hpp"
#include "util/byte_io.hpp"
#include "util/int_types.hpp"
#include "core/memory/allocator.hpp"

//#include <ei/vector.hpp>

namespace mufflon { namespace scene {

bool starts_with(std::string_view str, std::string_view begin) {
	return str.rfind(begin.data(), 0u) != std::string_view::npos;
}

template < bool face >
bool is_ignored_property(const std::string& name) {
	// Ignore all standard properties as well as status
	// TODO: there are more of these
	if constexpr(face) {
		if(name == "<fprop>" || starts_with(name, "f:st")) // short for f:status
			return true;
	} else {
		if(name == "<vprop>" || starts_with(name, "v:st")) // short for v:status
			return true;
	}
	return false;
}

AttributePool::AttributePool(const AttributePool& pool) :
	m_attribElemCount(pool.m_attribElemCount),
	m_attribElemCapacity(pool.m_attribElemCapacity),
	m_poolSize(pool.m_poolSize),
	m_attributes(pool.m_attributes)
{
	pool.m_pools.for_each([&](auto& elem) {
		using ChangedBuffer = std::decay_t<decltype(elem)>;
		auto& pool = m_pools.template get<PoolHandle<ChangedBuffer::DEVICE>>();
		if(m_poolSize == 0 || elem.handle == nullptr) {
			pool.handle.reset();
		} else {
			pool.handle = make_udevptr_array<ChangedBuffer::DEVICE, char, false>(m_poolSize);
			mufflon::copy(pool.handle.get(), elem.handle.get(), m_poolSize);
		}
	});
}

AttributePool::AttributePool(AttributePool&& pool) noexcept :
	m_attribElemCount{ pool.m_attribElemCount },
	m_attribElemCapacity{ pool.m_attribElemCapacity },
	m_poolSize{ pool.m_poolSize },
	m_pools{ std::move(pool.m_pools) },
	m_attributes{ std::move(pool.m_attributes) }
{}

AttributePool& AttributePool::operator=(AttributePool&& pool) noexcept {
	m_attribElemCapacity = pool.m_attribElemCapacity;
	m_attribElemCount = pool.m_attribElemCount;
	m_poolSize = pool.m_poolSize;
	m_pools = std::move(pool.m_pools);
	m_attributes = std::move(pool.m_attributes);
	return *this;
}

AttributePool::~AttributePool() {
	m_pools.for_each([len = m_poolSize](auto& elem) {
		using ChangedBuffer = std::decay_t<decltype(elem)>;
		if(elem.handle != nullptr)
			elem.handle.reset();
	});
	for(auto& attrib : m_attributes)
		util::UniqueStringPool::instance().remove(attrib.name);
}
AttributeHandle AttributePool::add_attribute(const AttributeIdentifier& ident) {
	this->unload<Device::CUDA>();
	this->unload<Device::OPENGL>();
	// Create the accessor...
	AttribInfo info{
		get_attribute_size(ident.type),
		m_poolSize,
		util::UniqueStringPool::instance().insert(ident.name)
	};
	m_poolSize += info.elemSize * m_attribElemCapacity;
	// ...and map the name to the index
	const auto index = insert_attribute_at_first_empty(std::move(info));
	return AttributeHandle{ ident, static_cast<u32>(index) };
}

std::optional<AttributeHandle> AttributePool::find_attribute(const AttributeIdentifier& ident) const {
	for(u32 i = 0u; i < static_cast<u32>(m_attributes.size()); ++i) {
		if(!m_attributes[i].erased && m_attributes[i].name == ident.name) {
			return AttributeHandle{ ident, i };
		}
	}
	return std::nullopt;
}

void AttributePool::remove(AttributeHandle handle) {
	// TODO: check out-of-bounds!

	this->unload<Device::CUDA>();
	this->unload<Device::OPENGL>();
	// Adjust pool sizes for following attributes
	const std::size_t segmentSize = (handle.index == m_attributes.size() - 1u)
		? m_poolSize - m_attributes[handle.index].poolOffset
		: m_attributes[handle.index + 1].poolOffset - m_attributes[handle.index].poolOffset;
	for(std::size_t i = handle.index + 1; i < m_attributes.size(); ++i)
		m_attributes[i].poolOffset -= segmentSize;

	m_poolSize -= segmentSize;

	// Remove the attribute from bookkeeping
	if(handle.index == m_attributes.size() - 1u)
		m_attributes.pop_back();
	else
		m_attributes[handle.index].erased = true;
}

// Causes force-unload on actual reserve
// Capacity is in terms of elements, not bytes
void AttributePool::reserve(std::size_t capacity) {
	if(capacity <= m_attribElemCapacity)
		return;
	this->unload<Device::CUDA>();
	this->unload<Device::OPENGL>();

	std::size_t currOffset = 0u;
	// Adjust pool offsets for the attributes
	for(auto& attrib : m_attributes) {
		if(!attrib.erased) {
			attrib.poolOffset = currOffset;
			currOffset += attrib.elemSize * capacity;
		}
	}

	m_attribElemCapacity = capacity;
	auto& pool = m_pools.template get<PoolHandle<Device::CPU>>().handle;
	if(pool && m_poolSize > 0u) {
		const auto realloced = Allocator<Device::CPU>::template realloc<char>(pool.release(), m_poolSize, currOffset);
		pool = unique_device_ptr<Device::CPU, char[]>(realloced, { currOffset });
	} else {
		pool = make_udevptr_array<Device::CPU, char, false>(currOffset);
	}
	m_poolSize = currOffset;
}

// Resizes the attribute, leaves the memory uninitialized
// Force-unloads non-CPU pools if reserve necessary
void AttributePool::resize(std::size_t size) {
	this->reserve(size);
	m_attribElemCount = size;
}

// Shrinks the memory to fit the element count on all devices
// Does not unload any device memory
void AttributePool::shrink_to_fit() {
	if(m_attribElemCount == m_attribElemCapacity)
		return;

	if(m_attribElemCount != 0) {
		std::size_t bytes = m_attribElemCount * m_poolSize / m_attribElemCapacity;
		m_pools.for_each([bytes, prev = m_poolSize](auto& pool) {
			using ChangedBuffer = std::decay_t<decltype(pool)>;
			if(pool.handle != nullptr) {
				const auto realloced = Allocator<ChangedBuffer::DEVICE>::template realloc<char>(pool.handle.release(), prev, bytes);
				pool.handle = unique_device_ptr<ChangedBuffer::DEVICE, char[]>(realloced, { bytes });
			}
		});
		m_poolSize = bytes;
	} else {
		m_pools.for_each([prev = m_poolSize](auto& pool) {
			using ChangedBuffer = std::decay_t<decltype(pool)>;
			pool.handle.reset();
		});
		m_poolSize = 0u;
	}

	m_attribElemCapacity = m_attribElemCount;
	// We also have to adjust the offsets
	std::size_t currOffset = 0u;
	// Adjust pool offsets for the attributes
	for(auto& attrib : m_attributes) {
		if(!attrib.erased) {
			attrib.poolOffset = currOffset;
			currOffset += attrib.elemSize * m_attribElemCapacity;
		}
	}
}

template < Device dev >
void AttributePool::copy(std::size_t from, std::size_t to) {
	mAssert(from < m_attribElemCount);
	mAssert(to < m_attribElemCount);
	this->template synchronize<dev>();
	if constexpr(dev != Device::CPU)
		this->unload<Device::CPU>();
	if constexpr(dev != Device::CUDA)
		this->unload<Device::CUDA>();
	if constexpr(dev != Device::OPENGL)
		this->unload<Device::OPENGL>();

	ArrayDevHandle_t<dev, char> pool = m_pools.template get<PoolHandle<dev>>().handle.get();
	for(const auto& attr : m_attributes) {
		if(attr.erased)
			continue;

		mufflon::copy(pool + (attr.poolOffset + to * attr.elemSize),
					  pool + (attr.poolOffset + from * attr.elemSize),
					  attr.elemSize);
	}
}

template < Device dev >
void AttributePool::copy(AttributePool& fromPool, const std::size_t from, const std::size_t to) {
	mAssert(fromPool.m_attributes.size() == m_attributes.size());
	mAssert(from < fromPool.m_attribElemCount);
	mAssert(to < m_attribElemCount);
	this->template synchronize<dev>();
	fromPool.template synchronize<dev>();
	if constexpr(dev != Device::CPU)
		this->unload<Device::CPU>();
	if constexpr(dev != Device::CUDA)
		this->unload<Device::CUDA>();
	if constexpr(dev != Device::OPENGL)
		this->unload<Device::OPENGL>();

	ArrayDevHandle_t<dev, char> pool = m_pools.template get<PoolHandle<dev>>().handle.get();
	ArrayDevHandle_t<dev, char> otherPool = fromPool.m_pools.template get<PoolHandle<dev>>().handle.get();
	for(std::size_t i = 0u; i < m_attributes.size(); ++i) {
		const auto& attr = m_attributes[i];
		const auto& fromAttr = fromPool.m_attributes[i];
		mAssert(attr.elemSize == fromAttr.elemSize);
		mAssert(attr.erased == fromAttr.erased);
		if(attr.erased)
			continue;

		mufflon::copy(pool + (attr.poolOffset + to * attr.elemSize),
					  otherPool + (fromAttr.poolOffset + from * attr.elemSize),
					  attr.elemSize);
	}
}

void AttributePool::mark_changed(Device dev) {
	m_pools.for_each([dev, prev = m_poolSize](auto& pool) {
		using ChangedBuffer = std::decay_t<decltype(pool)>;
		if(ChangedBuffer::DEVICE != dev) {
			pool.handle.reset();
		}
	});
}

template < Device dev >
void AttributePool::synchronize() {
	if(m_poolSize == 0u)
		return;
	if(m_attribElemCount == 0)
		return;

	auto& syncPool = m_pools.template get<PoolHandle<dev>>().handle;
	if(syncPool != nullptr) return;

	// Always allocate memory (copies can and will only be done if there is a dirty memory)
	syncPool = make_udevptr_array<dev, char, false>(m_poolSize);

	// TODO: OpenGL
	switch(dev) {
		case Device::CPU:
			if(m_pools.template get<PoolHandle<Device::CUDA>>().handle != nullptr)
				mufflon::copy(syncPool.get(), m_pools.template get<PoolHandle<Device::CUDA>>().handle.get(), m_poolSize);
			else if(m_pools.template get<PoolHandle<Device::OPENGL>>().handle != nullptr)
				mufflon::copy(syncPool.get(), m_pools.template get<PoolHandle<Device::OPENGL>>().handle.get(), m_poolSize);
			break;
		case Device::CUDA:
			if(m_pools.template get<PoolHandle<Device::CPU>>().handle != nullptr) {
				mufflon::copy(syncPool.get(), m_pools.template get<PoolHandle<Device::CPU>>().handle.get(), m_poolSize);
			} else if(m_pools.template get<PoolHandle<Device::OPENGL>>().handle != nullptr) {
				logError("Cannot synchronize from OpenGL to CUDA yet!");
				syncPool.reset();
			}
			break;
		case Device::OPENGL:
			if(m_pools.template get<PoolHandle<Device::CPU>>().handle != nullptr) {
				mufflon::copy(syncPool.get(), m_pools.template get<PoolHandle<Device::CPU>>().handle.get(), m_poolSize);
			} else if(m_pools.template get<PoolHandle<Device::CUDA>>().handle != nullptr) {
				logError("Cannot synchronize from CUDA to OpenGL yet!");
				syncPool.reset();
			}
			break;
	}
}

template < Device dev >
void AttributePool::unload() {
	// TODO: detect if we unload last pool
	m_pools.template get<PoolHandle<dev>>().handle.reset();
}

// Loads the attribute from a byte stream
std::size_t AttributePool::restore(AttributeHandle hdl, util::IByteReader& attrStream,
					std::size_t start, std::size_t count) {
	mAssert(hdl.index < m_attributes.size());
	mAssert(!m_attributes[hdl.index].erased);
	this->synchronize<Device::CPU>();
	if(start + count > m_attribElemCount)
		this->resize(start + count);

	AttribInfo& attribute = m_attributes[hdl.index];
	const std::size_t elemSize = attribute.elemSize;
	char* mem = m_pools.template get<PoolHandle<Device::CPU>>().handle.get() + attribute.poolOffset + elemSize * start;
	std::size_t read = attrStream.read(mem, elemSize * count);
	if(read > 0)
		this->mark_changed(Device::CPU);
	return read / attribute.elemSize;
}

std::size_t AttributePool::store(AttributeHandle hdl, util::IByteWriter& attrStream,
								 std::size_t start, std::size_t count) {
	mAssert(hdl.index < m_attributes.size());
	mAssert(!m_attributes[hdl.index].erased);
	this->synchronize<Device::CPU>();
	std::size_t actualCount = count;
	if(start + count > m_attribElemCount)
		actualCount = m_attribElemCount - start;

	AttribInfo& attribute = m_attributes[hdl.index];
	const std::size_t elemSize = attribute.elemSize;
	const char* mem = m_pools.template get<PoolHandle<Device::CPU>>().handle.get() + attribute.poolOffset + elemSize * start;
	return attrStream.write(mem, elemSize * actualCount) / attribute.elemSize;
}

std::size_t AttributePool::insert_attribute_at_first_empty(AttribInfo&& info) {
	for(std::size_t i = 0u; i < m_attributes.size(); ++i) {
		if(m_attributes[i].erased) {
			m_attributes[i] = info;
			return i;
		}
	}
	m_attributes.push_back(info);
	return m_attributes.size() - 1u;
}

// Explicit instantiations
template void AttributePool::synchronize<Device::CPU>();
template void AttributePool::synchronize<Device::CUDA>();
template void AttributePool::synchronize<Device::OPENGL>();
template void AttributePool::unload<Device::CPU>();
template void AttributePool::unload<Device::CUDA>();
template void AttributePool::unload<Device::OPENGL>();
template void AttributePool::copy<Device::CPU>(std::size_t, std::size_t);
template void AttributePool::copy<Device::CUDA>(std::size_t, std::size_t);
template void AttributePool::copy<Device::OPENGL>(std::size_t, std::size_t);
template void AttributePool::copy<Device::CPU>(AttributePool&, std::size_t, std::size_t);
template void AttributePool::copy<Device::CUDA>(AttributePool&, std::size_t, std::size_t);
template void AttributePool::copy<Device::OPENGL>(AttributePool&, std::size_t, std::size_t);

}} // namespace mufflon::scene