#pragma once

#include "attribute_sizes.hpp"
#include "attribute_handles.hpp"
#include "core/memory/allocator.hpp"
#include "core/memory/residency.hpp"
#include "core/scene/geometry/polygon_mesh.hpp"
#include "util/byte_io.hpp"
#include "util/string_pool.hpp"
#include "util/tagged_tuple.hpp"
#include <cstddef>
#include <functional>
#include <climits>
#include <optional>
#include <stdexcept>
#include <string>
#include "util/string_view.hpp"
#include <type_traits>
#include <unordered_map>
#include <vector>
#include "core/memory/dyntype_memory.hpp"

namespace mufflon::scene {

struct VertexAttributeHandle final : public AttributeHandle {
	VertexAttributeHandle(const AttributeIdentifier& ident, const std::uint32_t index) : AttributeHandle(ident, index) {}
};
struct FaceAttributeHandle final : public AttributeHandle {
	FaceAttributeHandle(const AttributeIdentifier& ident, const std::uint32_t index) : AttributeHandle(ident, index) {}
};
struct SphereAttributeHandle final : public AttributeHandle {
	SphereAttributeHandle(const AttributeIdentifier& ident, const std::uint32_t index) : AttributeHandle(ident, index) {}
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
public:
	static constexpr bool IS_FACE = IsFace;
	template < class T >
	using PropertyHandleType = std::conditional_t<IS_FACE, OpenMesh::FPropHandleT<T>, OpenMesh::VPropHandleT<T>>;
	using AttrHandle = std::conditional_t<IS_FACE, FaceAttributeHandle, VertexAttributeHandle>;

private:
	template < Device dev >
	struct AcquireHelper {};

	template <> 
	struct AcquireHelper< Device::CPU > {
		template <class T>
		static ArrayDevHandle_t<Device::CPU, T> acquire(OpenMeshAttributePool<IsFace>& parent, const AttrHandle& handle) {
			return as<ArrayDevHandle_t<Device::CPU, T>, ArrayDevHandle_t<Device::CPU, char>>(
				parent.get_attribute_cpu_data(handle));
		}
	};

	template <>
	struct AcquireHelper< Device::CUDA > {
		template <class T>
		static ArrayDevHandle_t<Device::CUDA, T> acquire(OpenMeshAttributePool<IsFace>& parent, const AttrHandle& handle) {
			return as<ArrayDevHandle_t<Device::CUDA, T>, ArrayDevHandle_t<Device::CUDA, char>>(
				parent.m_cudaPool + parent.get_attribute_pool_offset(handle));
		}
	};

	template <>
	struct AcquireHelper< Device::OPENGL > {
		template <class T>
		static ArrayDevHandle_t<Device::OPENGL, T> acquire(OpenMeshAttributePool<IsFace>& parent, const AttrHandle& handle) {
			return as<ArrayDevHandle_t<Device::OPENGL, T>, ArrayDevHandle_t<Device::OPENGL, char>>(
				parent.m_openglPool + parent.get_attribute_pool_offset(handle));
		}
	};

	friend AcquireHelper < Device::CPU >;
	friend AcquireHelper < Device::CUDA >;
	friend AcquireHelper < Device::OPENGL >;

public:
	OpenMeshAttributePool(geometry::PolygonMeshType &mesh);
	OpenMeshAttributePool(const OpenMeshAttributePool&) = delete;
	OpenMeshAttributePool& operator=(const OpenMeshAttributePool&) = delete;
	OpenMeshAttributePool& operator=(OpenMeshAttributePool&&) = delete;
	OpenMeshAttributePool(OpenMeshAttributePool&& pool);
	~OpenMeshAttributePool();

	// Copies over name maps etc.
	void copy(const OpenMeshAttributePool<IsFace>& pool);

	// Registers OpenMesh's default 'point' attribute
	template < bool Face = IS_FACE >
	std::enable_if_t<!Face, AttrHandle> register_point_attribute() {
		if(!m_mesh.points_pph().is_valid())
			throw std::runtime_error("Got invalid points property (perhaps it is not selected as default?)");
		return AttrHandle{
			AttributeIdentifier{ AttributeType::FLOAT3, "v:points" },
			static_cast<std::uint32_t>(m_mesh.points_pph().idx())
		};
	}
	// Registers OpenMesh's default 'normals' attribute
	template < bool Face = IS_FACE >
	std::enable_if_t<!Face, AttrHandle> register_normal_attribute() {
		if(!m_mesh.vertex_normals_pph().is_valid())
			throw std::runtime_error("Got invalid normals property (perhaps it is not selected as default?)");
		return AttrHandle{
			AttributeIdentifier{ AttributeType::FLOAT3, "v:normals" },
			static_cast<std::uint32_t>(m_mesh.vertex_normals_pph().idx())
		};
	}
	// Registers OpenMesh's default 'uv' attribute
	template < bool Face = IS_FACE >
	std::enable_if_t<!Face, AttrHandle> register_uv_attribute() {
		if(!m_mesh.vertex_texcoords2D_pph().is_valid())
			throw std::runtime_error("Got invalid uv property (perhaps it is not selected as default?)");
		return AttrHandle{
			AttributeIdentifier{ AttributeType::FLOAT2, "v:texcoords2D" },
			static_cast<std::uint32_t>(m_mesh.vertex_texcoords2D_pph().idx())
		};
	}

	// Add a new attribute. Overwrites any existing attribute with the same name
	AttrHandle add_attribute(const AttributeIdentifier& ident);
	std::optional<AttrHandle> find_attribute(const AttributeIdentifier& ident) const;
	void remove_attribute(const AttrHandle& handle);

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
	ArrayDevHandle_t<dev, T> acquire(const AttrHandle& handle) {
		this->synchronize<dev>();
		return AcquireHelper<dev>::template acquire<T>(*this, handle);
	}
	template < Device dev, class T >
	ConstArrayDevHandle_t<dev, T> acquire_const(const AttrHandle& handle) {
		this->synchronize<dev>();
		return AcquireHelper<dev>::template acquire<T>(*this, handle);
	}

	template < Device dev >
	void synchronize();

	template < Device dev >
	void unload();

	void mark_changed(Device dev);

	// Loads the attribute from a byte stream, starting at elem start
	// Resizes the attributes if necessary
	std::size_t restore(const AttrHandle& handle, util::IByteReader& attrStream,
						std::size_t start, std::size_t count);

	std::size_t get_attribute_elem_count() const noexcept {
		return m_attribElemCount;
	}

	std::size_t get_attribute_elem_capacity() const noexcept {
		return m_attribElemCapacity;
	}

private:
	// Provide attribute iterator semantics
	auto begin() {
		if constexpr(IS_FACE)
			return m_mesh.fprops_begin();
		else
			return m_mesh.vprops_begin();
	}
	auto end() {
		if constexpr(IS_FACE)
			return m_mesh.fprops_end();
		else
			return m_mesh.vprops_end();
	}

	// If you have a handle, this is legit
	char* get_attribute_cpu_data(const AttrHandle& handle);
	char* get_attribute_cpu_data(OpenMesh::BaseProperty& prop);
	std::size_t get_attribute_pool_offset(const AttrHandle& handle);
	std::size_t get_attribute_element_size(const AttrHandle& handle);
	std::size_t get_attribute_element_size(const OpenMesh::BaseProperty& prop);

	geometry::PolygonMeshType &m_mesh;	// References the OpenMesh mesh
	std::size_t m_attribElemCount = 0u;
	std::size_t m_attribElemCapacity = 0u;
	std::size_t m_poolSize = 0u;
	ArrayDevHandle_t<Device::CUDA, char> m_cudaPool = nullptr;
	ArrayDevHandle_t<Device::OPENGL, char> m_openglPool;
	// TODO: OpenGL pool?

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
	SphereAttributeHandle add_attribute(const AttributeIdentifier& ident);
	std::optional<SphereAttributeHandle> find_attribute(const AttributeIdentifier& ident) const;
	void remove(SphereAttributeHandle handle);

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
	ArrayDevHandle_t<dev, T> acquire(SphereAttributeHandle hdl) {
		mAssert(hdl.index < m_attributes.size());
		mAssert(!m_attributes[hdl.index].erased);
		this->synchronize<dev>();
		return as<ArrayDevHandle_t<dev, T>, ArrayDevHandle_t<dev, char>>(
			m_pools.template get<PoolHandle<dev>>().handle + m_attributes[hdl.index].poolOffset);
	}

	template < Device dev, class T >
	ConstArrayDevHandle_t<dev, T> acquire_const(SphereAttributeHandle hdl) {
		mAssert(hdl.index < m_attributes.size());
		mAssert(!m_attributes[hdl.index].erased);
		this->synchronize<dev>();
		return as<ArrayDevHandle_t<dev, T>, ArrayDevHandle_t<dev, char>>(
			m_pools.template get<PoolHandle<dev>>().handle + m_attributes[hdl.index].poolOffset);
	}

	template < Device dev >
	void synchronize();

	template < Device dev >
	void unload();

	void mark_changed(Device dev);

	// Loads the attribute from a byte stream, starting at elem start
	// Resizes the attributes if necessary
	// Returns the number of read instances.
	std::size_t restore(SphereAttributeHandle hdl, util::IByteReader& attrStream,
						std::size_t start, std::size_t count);

	// Store the attribute to a byte stream, starting at elem start
	std::size_t store(SphereAttributeHandle hdl, util::IByteWriter& attrStream,
					  std::size_t start, std::size_t count);

	std::size_t get_attribute_elem_count() const noexcept {
		return m_attribElemCount;
	}

	std::size_t get_attribute_elem_capacity() const noexcept {
		return m_attribElemCapacity;
	}

	// Resolves a name to an attribute
	SphereAttributeHandle get_attribute_handle(StringView name);
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
		ArrayDevHandle_t<dev, char> handle = ArrayDevHandle_t<dev, char>{};
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
