#include "attribute.hpp"
#include "core/memory/residency.hpp"
#include "core/memory/synchronize.hpp"
#include <ei/vector.hpp>

namespace mufflon { namespace scene {

template < bool face >
OpenMeshAttributePool<face>::OpenMeshAttributePool(geometry::PolygonMeshType &mesh) :
	m_mesh(mesh)
{}

template < bool face >
OpenMeshAttributePool<face>::OpenMeshAttributePool(OpenMeshAttributePool&& pool) :
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

template < bool face >
OpenMeshAttributePool<face>::~OpenMeshAttributePool() {
	if(m_cudaPool)
		m_cudaPool = Allocator<Device::CUDA>::free(m_cudaPool, m_poolSize);
}

// Reserving memory force-unloads other devices
// Capacity is in terms of elements, not bytes
template < bool face >
void OpenMeshAttributePool<face>::reserve(std::size_t capacity) {
	if(capacity <= m_attribElemCapacity)
		return;

	this->unload<Device::CUDA>();
	if constexpr(IS_FACE)
		m_mesh.reserve(m_mesh.n_vertices(), m_mesh.n_edges(), capacity);
	else
		m_mesh.reserve(capacity, m_mesh.n_edges(), m_mesh.n_faces());

	std::size_t currOffset = 0u;
	// Adjust pool offsets for the attributes
	for(auto& attrib : m_attributes) {
		attrib.poolOffset = currOffset;
		currOffset += attrib.elemSize * capacity;
	}

	m_poolSize = currOffset;
	m_attribElemCapacity = capacity;
}

// Resizes the attribute, leaves the memory uninitialized
// Force-unloads non-CPU pools if reserve necessary
template < bool face >
void OpenMeshAttributePool<face>::resize(std::size_t size) {
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
template < bool face >
void OpenMeshAttributePool<face>::shrink_to_fit() {
	m_mesh.garbage_collection();
	if(m_attribElemCount == m_attribElemCapacity)
		return;
	if constexpr(IS_FACE)
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

template < bool face >
template < Device dev >
void OpenMeshAttributePool<face>::synchronize() {
	if(!m_dirty.has_changes())
		return;
	if(m_dirty.has_competing_changes())
		throw std::runtime_error("Competing changes in attribute pool");
	if(!m_dirty.needs_sync(dev))
		return;
	if(m_attribElemCount == 0)
		return;

	switch(dev) {
		case Device::CPU:
		{
			mAssert(m_cudaPool != nullptr);
			// Copy over dirty attributes
			std::size_t currOffset = 0u;
			for(auto& attrib : m_attributes) {
				if(attrib.dirty.needs_sync(dev)) {
					char* cpuData = attrib.accessor(m_mesh);
					copy(cpuData, &m_cudaPool[currOffset], attrib.elemSize * m_attribElemCount);
					currOffset += attrib.elemSize * m_attribElemCount;
					attrib.dirty.mark_synced(dev);
				}
			}
		}	break;
		case Device::CUDA:
		{
			bool copyAll = !m_cudaPool;
			if(!m_cudaPool)
				m_cudaPool = Allocator<Device::CUDA>::alloc_array<char>(m_poolSize);
			// Copy over all attributes
			std::size_t currOffset = 0u;
			for(auto& attrib : m_attributes) {
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

template < bool face >
template < Device dev >
void OpenMeshAttributePool<face>::synchronize(AttributeHandle hdl) {
	mAssert(hdl.index < m_attributes.size());
	if(dev == Device::CUDA && !m_cudaPool)
		this->synchronize<dev>();

	if(!m_attributes[hdl.index].dirty.needs_sync(dev))
		return;

	const std::size_t offset = m_attributes[hdl.index].poolOffset;
	switch(dev) {
		case Device::CPU:
		{
			copy(m_attributes[hdl.index].accessor(m_mesh) + offset, m_cudaPool + offset, m_attributes[hdl.index].elemSize * m_attribElemCount);
		}	break;
		case Device::CUDA:
		{
			copy(m_cudaPool + offset, m_attributes[hdl.index].accessor(m_mesh) + offset, m_attributes[hdl.index].elemSize * m_attribElemCount);
		}	break;
	}
}

template < bool face >
template < Device dev >
void OpenMeshAttributePool<face>::unload() {
	// We cannot unload (CPU) OpenMesh data (without removing the property?)
	switch(dev) {
		case Device::CUDA:
			if(m_cudaPool)
				m_cudaPool = Allocator<Device::CUDA>::free(m_cudaPool, m_poolSize);
			break;
	}
}

template < bool face >
void OpenMeshAttributePool<face>::mark_changed(Device dev, AttributeHandle hdl) {
	mAssert(hdl.index < m_attributes.size());
	m_dirty.mark_changed(dev);
	m_attributes[hdl.index].dirty.mark_changed(dev);
}

template < bool face >
void OpenMeshAttributePool<face>::mark_changed(Device dev) {
	m_dirty.mark_changed(dev);
	for(auto& attr : m_attributes) {
		attr.dirty.mark_changed(dev);
	}
}

template < bool face >
std::size_t OpenMeshAttributePool<face>::restore(AttributeHandle hdl, util::IByteReader& attrStream,
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

template < bool face >
std::size_t OpenMeshAttributePool<face>::store(AttributeHandle hdl, util::IByteWriter& attrStream,
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

template < bool face >
AttributeHandle OpenMeshAttributePool<face>::get_attribute_handle(std::string_view name) {
	auto mapIter = m_nameMap.find(name);
	if(mapIter == m_nameMap.end())
		throw std::runtime_error("Could not find attribute '" + std::string(name) + "'");
	return AttributeHandle{ mapIter->second };
}

AttributePool::AttributePool(AttributePool&& pool) :
	m_nameMap(std::move(pool.m_nameMap)),
	m_attribElemCount(pool.m_attribElemCount),
	m_attribElemCapacity(pool.m_attribElemCapacity),
	m_poolSize(pool.m_poolSize),
	m_pools(pool.m_pools),
	m_dirty(pool.m_dirty),
	m_attributes(std::move(pool.m_attributes)) {
	m_pools.for_each([](auto& elem) {
		using ChangedBuffer = std::decay_t<decltype(elem)>;
		elem.handle = ArrayDevHandle_t<ChangedBuffer::DEVICE, char>{};
	});
}

AttributePool::~AttributePool() {
	m_pools.for_each([len = m_poolSize](auto& elem) {
		using ChangedBuffer = std::decay_t<decltype(elem)>;
		if(elem.handle)
			elem.handle = Allocator<ChangedBuffer::DEVICE>::free(elem.handle, len);
	});
}

// Causes force-unload on actual reserve
// Capacity is in terms of elements, not bytes
void AttributePool::reserve(std::size_t capacity) {
	if(capacity <= m_attribElemCapacity)
		return;
	this->unload<Device::CUDA>();

	std::size_t currOffset = 0u;
	// Adjust pool offsets for the attributes
	for(auto& attrib : m_attributes) {
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
			if(pool.handle)
				pool.handle = Allocator<ChangedBuffer::DEVICE>::realloc(pool.handle, prev, bytes);
		});
		m_poolSize = bytes;
	} else {
		m_pools.for_each([prev = m_poolSize](auto& pool) {
			using ChangedBuffer = std::decay_t<decltype(pool)>;
			if(pool.handle)
				pool.handle = Allocator<ChangedBuffer::DEVICE>::free(pool.handle, prev);
		});
		m_poolSize = 0u;
	}
}

void AttributePool::mark_changed(Device dev, AttributeHandle hdl) {
	mAssert(hdl.index < m_attributes.size());
	m_dirty.mark_changed(dev);
	m_attributes[hdl.index].dirty.mark_changed(dev);
}

void AttributePool::mark_changed(Device dev) {
	m_dirty.mark_changed(dev);
	for(auto& attr : m_attributes) {
		attr.dirty.mark_changed(dev);
	}
}

template < Device dev >
void AttributePool::synchronize() {
	if(m_poolSize == 0u)
		return;
	// Always allocate memory (copies can and will only be done if there is a dirty memory)
	ArrayDevHandle_t<dev, char>& syncPool = m_pools.template get<PoolHandle<dev>>().handle;
	bool hadNoMemory = !syncPool;
	if(hadNoMemory)
		syncPool = Allocator<dev>::alloc_array<char>(m_poolSize);

	if(!m_dirty.has_changes())
		return;
	if(m_dirty.has_competing_changes())
		throw std::runtime_error("Competing changes in attribute pool");
	if(!m_dirty.needs_sync(dev))
		return;
	if(m_attribElemCount == 0)
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

	if(changedPool) {
		if(hadNoMemory) {
			// If there was no pool allocated we need to copy everything anyway
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
void AttributePool::synchronize(AttributeHandle hdl) {
	mAssert(hdl.index < m_attributes.size());
	ArrayDevHandle_t<dev, char>& syncPool = m_pools.template get<PoolHandle<dev>>().handle;
	if(!syncPool) { // If memory is missing all attributes need to be synced
		this->synchronize<dev>();
	}

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
void AttributePool::unload() {
	// TODO: detect if we unload last pool
	auto& pool = m_pools.template get<PoolHandle<dev>>().handle;
	if(pool)
		pool = Allocator<dev>::free(pool, m_poolSize);
}

// Loads the attribute from a byte stream
std::size_t AttributePool::restore(AttributeHandle hdl, util::IByteReader& attrStream,
					std::size_t start, std::size_t count) {
	mAssert(hdl.index < m_attributes.size());
	this->synchronize<Device::CPU>(hdl);
	if(start + count > m_attribElemCount)
		this->resize(start + count);

	AttribInfo& attribute = m_attributes[hdl.index];
	const std::size_t elemSize = attribute.elemSize;
	char* mem = m_pools.template get<PoolHandle<Device::CPU>>().handle + attribute.poolOffset + elemSize * start;
	std::size_t read = attrStream.read(mem, elemSize * count);
	if(read > 0)
		this->mark_changed(Device::CPU, hdl);
	return read / attribute.elemSize;
}

std::size_t AttributePool::store(AttributeHandle hdl, util::IByteWriter& attrStream,
				  std::size_t start, std::size_t count) {
	mAssert(hdl.index < m_attributes.size());
	this->synchronize<Device::CPU>(hdl);
	std::size_t actualCount = count;
	if(start + count > m_attribElemCount)
		actualCount = m_attribElemCount - start;

	AttribInfo& attribute = m_attributes[hdl.index];
	const std::size_t elemSize = attribute.elemSize;
	const char* mem = m_pools.template get<PoolHandle<Device::CPU>>().handle + attribute.poolOffset + elemSize * start;
	return attrStream.write(mem, elemSize * count) / attribute.elemSize;
}
	// Resolves a name to an attribute
AttributeHandle AttributePool::get_attribute_handle(std::string_view name) {
	auto mapIter = m_nameMap.find(name);
	if(mapIter == m_nameMap.end())
		throw std::runtime_error("Could not find attribute '" + std::string(name) + "'");
	return AttributeHandle{ mapIter->second };
}

// Explicit instantiations
template void AttributePool::synchronize<Device::CPU>();
template void AttributePool::synchronize<Device::CUDA>();
//template void AttributePool::synchronize<Device::OPENGL>();
template void AttributePool::synchronize<Device::CPU>(AttributeHandle hdl);
template void AttributePool::synchronize<Device::CUDA>(AttributeHandle hdl);
//template void AttributePool::synchronize<Device::OPENGL>(AttributeHandle hdl);
template void AttributePool::unload<Device::CPU>();
template void AttributePool::unload<Device::CUDA>();
//template void AttributePool::unload<Device::OPENGL>();
template class OpenMeshAttributePool<true>;
template class OpenMeshAttributePool<false>;
template void OpenMeshAttributePool<true>::synchronize<Device::CPU>();
template void OpenMeshAttributePool<true>::synchronize<Device::CUDA>();
//template void OpenMeshAttributePool<true>::synchronize<Device::OPENGL>();
template void OpenMeshAttributePool<true>::synchronize<Device::CPU>(AttributeHandle hdl);
template void OpenMeshAttributePool<true>::synchronize<Device::CUDA>(AttributeHandle hdl);
//template void OpenMeshAttributePool<true>::synchronize<Device::OPENGL>(AttributeHandle hdl);
template void OpenMeshAttributePool<true>::unload<Device::CPU>();
template void OpenMeshAttributePool<true>::unload<Device::CUDA>();
//template void OpenMeshAttributePool<true>::unload<Device::OPENGL>();
template void OpenMeshAttributePool<false>::synchronize<Device::CPU>();
template void OpenMeshAttributePool<false>::synchronize<Device::CUDA>();
//template void OpenMeshAttributePool<false>::synchronize<Device::OPENGL>();
template void OpenMeshAttributePool<false>::synchronize<Device::CPU>(AttributeHandle hdl);
template void OpenMeshAttributePool<false>::synchronize<Device::CUDA>(AttributeHandle hdl);
//template void OpenMeshAttributePool<false>::synchronize<Device::OPENGL>(AttributeHandle hdl);
template void OpenMeshAttributePool<false>::unload<Device::CPU>();
template void OpenMeshAttributePool<false>::unload<Device::CUDA>();
//template void OpenMeshAttributePool<false>::unload<Device::OPENGL>();

}} // namespace mufflon::scene