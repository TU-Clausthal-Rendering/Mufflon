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
	m_openglPool(pool.m_openglPool),
	m_attributes(std::move(pool.m_attributes)) {
	pool.m_cudaPool = ArrayDevHandle_t<Device::CUDA, char>{};
	pool.m_openglPool = ArrayDevHandle_t<Device::OPENGL, char>{};
}

template < bool face >
OpenMeshAttributePool<face>::~OpenMeshAttributePool() {
	if(m_cudaPool != nullptr)
		m_cudaPool = Allocator<Device::CUDA>::free(m_cudaPool, m_poolSize);
	if(m_openglPool != nullptr)
		m_openglPool = Allocator<Device::OPENGL>::free(m_openglPool, m_poolSize);
}

template < bool face >
void OpenMeshAttributePool<face>::copy(const OpenMeshAttributePool<face>& pool) {
	m_nameMap = pool.m_nameMap;
	m_attribElemCount = pool.m_attribElemCount;
	m_attribElemCapacity = pool.m_attribElemCapacity;
	m_poolSize = pool.m_poolSize;
	m_attributes = pool.m_attributes;
	m_openMeshSynced = pool.m_openMeshSynced;

	if(m_poolSize == 0 || pool.m_cudaPool == ArrayDevHandle_t<Device::CUDA, char>{}) {
		m_cudaPool = ArrayDevHandle_t<Device::CUDA, char>{};
	} else {
		m_cudaPool = Allocator<Device::CUDA>::template alloc_array<char, false>(m_poolSize);
		::mufflon::copy(m_cudaPool, pool.m_cudaPool, m_poolSize);
	}

	if(m_poolSize == 0 || pool.m_openglPool == ArrayDevHandle_t<Device::OPENGL, char>{}) {
		m_openglPool = ArrayDevHandle_t<Device::OPENGL, char>{};
	} else {
		m_openglPool = Allocator<Device::OPENGL>::template alloc_array<char, false>(m_poolSize);
		::mufflon::copy(m_openglPool, pool.m_openglPool, m_poolSize);
	}
}

// Reserving memory force-unloads other devices
// Capacity is in terms of elements, not bytes
template < bool face >
void OpenMeshAttributePool<face>::reserve(std::size_t capacity) {
	if(capacity <= m_attribElemCapacity)
		return;
	this->unload<Device::CUDA>();
	this->unload<Device::OPENGL>();
	if constexpr(IS_FACE)
		m_mesh.reserve(m_mesh.n_vertices(), m_mesh.n_edges(), capacity);
	else
		m_mesh.reserve(capacity, m_mesh.n_edges(), m_mesh.n_faces());

	std::size_t currOffset = 0u;
	// Adjust pool offsets for the attributes
	for(auto& attrib : m_attributes) {
		if(attrib.accessor) {
			attrib.poolOffset = currOffset;
			currOffset += attrib.elemSize * capacity;
		}
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
		if(m_cudaPool != nullptr)
			m_cudaPool = Allocator<Device::CUDA>::realloc(m_cudaPool, m_poolSize, bytes);
		if(m_openglPool != nullptr)
			m_openglPool = Allocator<Device::OPENGL>::realloc(m_openglPool, m_poolSize, bytes);
		m_poolSize = bytes;
	} else {
		if(m_cudaPool != nullptr)
			m_cudaPool = Allocator<Device::CUDA>::free(m_cudaPool, m_poolSize);
		if(m_openglPool != nullptr)
			m_openglPool = Allocator<Device::OPENGL>::free(m_openglPool, m_poolSize);
		m_poolSize = 0u;
	}

	m_attribElemCapacity = m_attribElemCount;
	// We also have to adjust the offsets
	std::size_t currOffset = 0u;
	// Adjust pool offsets for the attributes
	for(auto& attrib : m_attributes) {
		if(attrib.accessor) {
			attrib.poolOffset = currOffset;
			currOffset += attrib.elemSize * m_attribElemCapacity;
		}
	}
}

template < bool face >
template < Device dev >
void OpenMeshAttributePool<face>::synchronize() {
	if(m_poolSize == 0u)
		return;
	switch(dev) {
		case Device::CPU: {
			if(m_openMeshSynced)
				return;
			if(m_cudaPool != nullptr) {
				// Copy over dirty attributes
				for(auto& attrib : m_attributes) {
					if(attrib.accessor) {
						char* cpuData = attrib.accessor(m_mesh);
						::mufflon::copy(cpuData, m_cudaPool + attrib.poolOffset, attrib.elemSize * m_attribElemCount);
					}
				}
				m_openMeshSynced = true;
			} else if(m_openglPool != nullptr) {
				// Copy over dirty attributes
				for(auto& attrib : m_attributes) {
					if(attrib.accessor) {
						char* cpuData = attrib.accessor(m_mesh);
						::mufflon::copy(cpuData, m_openglPool + attrib.poolOffset, attrib.elemSize * m_attribElemCount);
					}
				}
				m_openMeshSynced = true;
			}
		}	break;
		case Device::CUDA: {
			if(m_cudaPool != nullptr)
				return;
			if(m_openMeshSynced) {
				m_cudaPool = Allocator<Device::CUDA>::alloc_array<char, false>(m_poolSize);
				// Copy over all attributes
				for(auto& attrib : m_attributes) {
					if(attrib.accessor) {
						const char* cpuData = attrib.accessor(m_mesh);
						::mufflon::copy(m_cudaPool + attrib.poolOffset, cpuData, attrib.elemSize * m_attribElemCount);
					}
				}
			} else if(m_openglPool != nullptr) {
				//m_cudaPool = Allocator<Device::CUDA>::alloc_array<char, false>(m_poolSize);
				logError("Cannot synchronize from OpenGL to CUDA yet!");
			}
		}	break;
		case Device::OPENGL:
			if(m_openglPool != nullptr)
				return;
			if(m_openMeshSynced) {
				m_openglPool = Allocator<Device::OPENGL>::alloc_array<char, false>(m_poolSize);
				// Copy over all attributes
				for(auto& attrib : m_attributes) {
					if(attrib.accessor) {
						const char* cpuData = attrib.accessor(m_mesh);
						::mufflon::copy(m_openglPool + attrib.poolOffset, cpuData, attrib.elemSize * m_attribElemCount);
					}
				}
			} else if(m_cudaPool != nullptr) {
				//m_openglPool = Allocator<Device::OPENGL>::alloc_array<char, false>(m_poolSize);
				logError("Cannot synchronize from CUDA to OpenGL yet!");
			}
			break;
	}
}

template < bool face >
template < Device dev >
void OpenMeshAttributePool<face>::unload() {
	// We cannot unload (CPU) OpenMesh data (without removing the property?)
	if constexpr(dev == Device::CPU) {
		m_openMeshSynced = false;
	} else if constexpr(dev == Device::CUDA) {
		if(m_cudaPool != nullptr)
			m_cudaPool = Allocator<Device::CUDA>::free(m_cudaPool, m_poolSize);
	} else if constexpr(dev == Device::OPENGL) {
		if(m_openglPool != nullptr)
			m_openglPool = Allocator<Device::OPENGL>::free(m_openglPool, m_poolSize);
	}
}

template < bool face >
void OpenMeshAttributePool<face>::mark_changed(Device dev) {
	if(dev != Device::CPU)
		unload<Device::CPU>();
	else
		m_openMeshSynced = true;
	if(dev != Device::CUDA)
		unload<Device::CUDA>();
	if(dev != Device::OPENGL)
		unload<Device::OPENGL>();
}

template < bool face >
std::size_t OpenMeshAttributePool<face>::restore(AttributeHandle hdl, util::IByteReader& attrStream,
												 std::size_t start, std::size_t count) {
	mAssert(hdl.index < m_attributes.size());
	mAssert(m_attributes[hdl.index].accessor);
	this->synchronize<Device::CPU>();
	if(start + count > m_attribElemCount)
		this->resize(start + count);

	AttribInfo& attribute = m_attributes[hdl.index];
	const std::size_t elemSize = attribute.elemSize;
	char* mem = attribute.accessor(m_mesh) + elemSize * start;
	std::size_t read = attrStream.read(mem, elemSize * count);
	if(read > 0)
		this->mark_changed(Device::CPU);
	return read / attribute.elemSize;
}

template < bool face >
std::size_t OpenMeshAttributePool<face>::store(AttributeHandle hdl, util::IByteWriter& attrStream,
											   std::size_t start, std::size_t count) {
	mAssert(hdl.index < m_attributes.size());
	mAssert(m_attributes[hdl.index].accessor);
	this->synchronize<Device::CPU>();
	std::size_t actualCount = count;
	if(start + count > m_attribElemCount)
		actualCount = m_attribElemCount - start;

	AttribInfo& attribute = m_attributes[hdl.index];
	const std::size_t elemSize = attribute.elemSize;
	const char* mem = attribute.accessor(m_mesh) + elemSize * start;
	return attrStream.write(mem, elemSize * actualCount) / attribute.elemSize;
}

template < bool face >
AttributeHandle OpenMeshAttributePool<face>::get_attribute_handle(StringView name) {
	auto mapIter = m_nameMap.find(name);
	if(mapIter == m_nameMap.end())
		throw std::runtime_error("Could not find attribute '" + std::string(name) + "'");
	return AttributeHandle{ mapIter->second };
}

AttributePool::AttributePool(const AttributePool& pool) :
	m_nameMap(pool.m_nameMap),
	m_attribElemCount(pool.m_attribElemCount),
	m_attribElemCapacity(pool.m_attribElemCapacity),
	m_poolSize(pool.m_poolSize),
	m_attributes(pool.m_attributes)
{
	pool.m_pools.for_each([&](auto& elem) {
		using ChangedBuffer = std::decay_t<decltype(elem)>;
		auto& pool = m_pools.template get<PoolHandle<ChangedBuffer::DEVICE>>();
		if(m_poolSize == 0 || elem.handle == ArrayDevHandle_t<ChangedBuffer::DEVICE, char>{}) {
			pool.handle = ArrayDevHandle_t<ChangedBuffer::DEVICE, char>{};
		} else {
			pool.handle = Allocator<ChangedBuffer::DEVICE>::template alloc_array<char, false>(m_poolSize);
			copy(pool.handle, elem.handle, m_poolSize);
		}
	});
}

AttributePool::AttributePool(AttributePool&& pool) :
	m_nameMap(std::move(pool.m_nameMap)),
	m_attribElemCount(pool.m_attribElemCount),
	m_attribElemCapacity(pool.m_attribElemCapacity),
	m_poolSize(pool.m_poolSize),
	m_pools(pool.m_pools),
	m_attributes(std::move(pool.m_attributes))
{
	m_pools.for_each([](auto& elem) {
		using ChangedBuffer = std::decay_t<decltype(elem)>;
		elem.handle = ArrayDevHandle_t<ChangedBuffer::DEVICE, char>{};
	});
}

AttributePool::~AttributePool() {
	m_pools.for_each([len = m_poolSize](auto& elem) {
		using ChangedBuffer = std::decay_t<decltype(elem)>;
		if(elem.handle != nullptr)
			elem.handle = Allocator<ChangedBuffer::DEVICE>::template free<char>(elem.handle, len);
	});
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
	if(pool && m_poolSize > 0u)
		pool = Allocator<Device::CPU>::realloc(pool, m_poolSize, currOffset);
	else
		pool = Allocator<Device::CPU>::alloc_array<char, false>(currOffset);
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
			if(pool.handle != nullptr)
				pool.handle = Allocator<ChangedBuffer::DEVICE>::template realloc<char>(pool.handle, prev, bytes);
		});
		m_poolSize = bytes;
	} else {
		m_pools.for_each([prev = m_poolSize](auto& pool) {
			using ChangedBuffer = std::decay_t<decltype(pool)>;
			if(pool.handle != nullptr)
				pool.handle = Allocator<ChangedBuffer::DEVICE>::template free<char>(pool.handle, prev);
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

void AttributePool::mark_changed(Device dev) {
	m_pools.for_each([dev, prev = m_poolSize](auto& pool) {
		using ChangedBuffer = std::decay_t<decltype(pool)>;
		if(ChangedBuffer::DEVICE != dev) {
			if(pool.handle != nullptr)
				pool.handle = Allocator<ChangedBuffer::DEVICE>::template free<char>(pool.handle, prev);
		}
	});
}

template < Device dev >
void AttributePool::synchronize() {
	if(m_poolSize == 0u)
		return;
	if(m_attribElemCount == 0)
		return;

	ArrayDevHandle_t<dev, char>& syncPool = m_pools.template get<PoolHandle<dev>>().handle;
	if(syncPool != nullptr) return;

	// Always allocate memory (copies can and will only be done if there is a dirty memory)
	syncPool = Allocator<dev>::template alloc_array<char, false>(m_poolSize);

	// TODO: OpenGL
	switch(dev) {
		case Device::CPU:
			if(m_pools.template get<PoolHandle<Device::CUDA>>().handle != nullptr)
				copy(syncPool, m_pools.template get<PoolHandle<Device::CUDA>>().handle, m_poolSize);
			else if(m_pools.template get<PoolHandle<Device::OPENGL>>().handle != nullptr)
				copy(syncPool, m_pools.template get<PoolHandle<Device::OPENGL>>().handle, m_poolSize);
			break;
		case Device::CUDA:
			if(m_pools.template get<PoolHandle<Device::CPU>>().handle != nullptr) {
				copy(syncPool, m_pools.template get<PoolHandle<Device::CPU>>().handle, m_poolSize);
			} else if(m_pools.template get<PoolHandle<Device::OPENGL>>().handle != nullptr) {
				logError("Cannot synchronize from OpenGL to CUDA yet!");
				syncPool = Allocator<dev>::free(syncPool, m_poolSize);
			}
			break;
		case Device::OPENGL:
			if(m_pools.template get<PoolHandle<Device::CPU>>().handle != nullptr) {
				copy(syncPool, m_pools.template get<PoolHandle<Device::CPU>>().handle, m_poolSize);
			} else if(m_pools.template get<PoolHandle<Device::CUDA>>().handle != nullptr) {
				logError("Cannot synchronize from CUDA to OpenGL yet!");
				syncPool = Allocator<dev>::free(syncPool, m_poolSize);
			}
			break;
	}
}

template < Device dev >
void AttributePool::unload() {
	// TODO: detect if we unload last pool
	auto& pool = m_pools.template get<PoolHandle<dev>>().handle;
	if(pool != nullptr)
		pool = Allocator<dev>::free(pool, m_poolSize);
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
	char* mem = m_pools.template get<PoolHandle<Device::CPU>>().handle + attribute.poolOffset + elemSize * start;
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
	const char* mem = m_pools.template get<PoolHandle<Device::CPU>>().handle + attribute.poolOffset + elemSize * start;
	return attrStream.write(mem, elemSize * actualCount) / attribute.elemSize;
}
// Resolves a name to an attribute
AttributeHandle AttributePool::get_attribute_handle(StringView name) {
	auto mapIter = m_nameMap.find(name);
	if(mapIter == m_nameMap.end())
		throw std::runtime_error("Could not find attribute '" + std::string(name) + "'");
	return AttributeHandle{ mapIter->second };
}

// Explicit instantiations
template void AttributePool::synchronize<Device::CPU>();
template void AttributePool::synchronize<Device::CUDA>();
template void AttributePool::synchronize<Device::OPENGL>();
template void AttributePool::unload<Device::CPU>();
template void AttributePool::unload<Device::CUDA>();
template void AttributePool::unload<Device::OPENGL>();
template class OpenMeshAttributePool<true>;
template class OpenMeshAttributePool<false>;
template void OpenMeshAttributePool<true>::synchronize<Device::CPU>();
template void OpenMeshAttributePool<true>::synchronize<Device::CUDA>();
template void OpenMeshAttributePool<true>::synchronize<Device::OPENGL>();
template void OpenMeshAttributePool<true>::unload<Device::CPU>();
template void OpenMeshAttributePool<true>::unload<Device::CUDA>();
template void OpenMeshAttributePool<true>::unload<Device::OPENGL>();
template void OpenMeshAttributePool<false>::synchronize<Device::CPU>();
template void OpenMeshAttributePool<false>::synchronize<Device::CUDA>();
template void OpenMeshAttributePool<false>::synchronize<Device::OPENGL>();
template void OpenMeshAttributePool<false>::unload<Device::CPU>();
template void OpenMeshAttributePool<false>::unload<Device::CUDA>();
template void OpenMeshAttributePool<false>::unload<Device::OPENGL>();

}} // namespace mufflon::scene