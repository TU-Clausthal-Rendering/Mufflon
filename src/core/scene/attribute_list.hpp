#pragma once

#include "core/memory/accessor.hpp"
#include "attribute.hpp"
#include "core/memory/synchronize.hpp"
#include "export/api.hpp"
#include "util/assert.hpp"
#include "util/tagged_tuple.hpp"
#include "util/byte_io.hpp"
#include <istream>
#include <ostream>
#include <cstdlib>
#include <unordered_map>
#include <optional>
#include <vector>

namespace mufflon::scene {

/**
 * Manages attributes and attribute pool access.
 * Access may happen via handles only, ie. one needs to aquire an attribute first.
 * Attribute pools may be synchronized to or unloaded from devices.
 * useOpenMesh: special case for OpenMesh, which stores its attributes on
 *     CPU-side itself.
 */
template < bool useOpenMesh = false, Device defaultDev = Device::CPU >
class AttributeList {
public:
	static constexpr bool USES_OPENMESH = useOpenMesh;
	static constexpr Device DEFAULT_DEVICE = defaultDev;
	using AttributePools = util::TaggedTuple<AttributePool<Device::CPU, !USES_OPENMESH>,
		AttributePool<Device::CUDA, true>>;

	AttributeList() {
		// Evaluate if we use OpenMesh and if the default device uses OpenMesh
		constexpr bool openMeshForDefDevice = std::conditional_t<DEFAULT_DEVICE == Device::CPU,
			std::bool_constant<useOpenMesh>, std::bool_constant<!useOpenMesh>>::value;
		// The default device gets to be present in the beginning
		m_attributePools.template get<AttributePool<DEFAULT_DEVICE, !openMeshForDefDevice>>().make_present();
	}

	AttributeList(const AttributeList&) = delete;
	AttributeList(AttributeList&&) = default;
	AttributeList& operator=(const AttributeList&) = delete;
	AttributeList& operator=(AttributeList&&) = default;
	~AttributeList() = default;

	template < class T >
	class AttributeHandle {
	public:
		using Type = T;

		AttributeHandle(std::size_t idx) :
			m_index(idx)
		{}

		constexpr std::size_t index() const noexcept {
			return m_index;
		}

	private:
		std::size_t m_index;
	};

	// Base type for inheritance
	class IBaseAttribute {
	public:
		virtual ~IBaseAttribute() {}
	};

	/**
	 * Attribute class granting access to the actual memory.
	 */
	template < class T >
	class Attribute : public IBaseAttribute {
	public:
		using Type = T;
		static constexpr bool USES_OPENMESH = useOpenMesh;

		template < bool openMesh = USES_OPENMESH, typename = std::enable_if_t<!openMesh> >
		Attribute(AttributePools& pools, util::DirtyFlags<Device>& flags) :
			m_pools(pools),
			m_flags(flags) {
			this->init_impl<0u>();
		}

		template < bool openMesh = USES_OPENMESH, typename = std::enable_if_t<openMesh> >
		Attribute(AttributePools& pools, util::DirtyFlags<Device>& flags, OpenMesh::PropertyT<T>& prop) :
			m_pools(pools),
			m_handles(),
			m_flags(flags) {
			this->init_impl<0u>(prop);
		}

		~Attribute() {
			this->destroy_impl<0u>();
		}

		// Aquires a read-write accessor to the attribute
		template < Device dev = DEFAULT_DEVICE >
		auto aquire() {
			this->synchronize<dev>();
			auto& pool = m_pools.template get<AttributePool<dev, stores_itself<dev>()>>();
			auto& handle = m_handles.template get<typename AttributePool<dev, stores_itself<dev>()>::template AttributeHandle<T>>();
			return Accessor<ArrayDevHandle<dev, Type>>{ static_cast<ArrayDevHandle_t<dev, Type>>(pool.aquire(handle)), m_flags };
		}

		// Aquires a read-only accessor to the attribute
		template < Device dev = DEFAULT_DEVICE >
		auto aquireConst() {
			this->synchronize<dev>();
			const auto& pool = m_pools.template get<AttributePool<dev, stores_itself<dev>()>>();
			const auto& handle = m_handles.template get<typename AttributePool<dev, stores_itself<dev>()>::template AttributeHandle<T>>();
			return ConstAccessor<ArrayDevHandle<dev, Type>>{ pool.aquireConst(handle) };
		}

		// Synchronizes the attribute pool to the given device
		template < Device dev = DEFAULT_DEVICE >
		void synchronize() {
			mufflon::synchronize<dev>(m_pools, m_flags, m_pools.template get<AttributePool<dev, stores_itself<dev>()>>());
		}

		std::size_t get_size() const noexcept {
			return m_pools.template get<0>().get_size();
		}

		constexpr std::size_t get_elem_size() const noexcept {
			return sizeof(T);
		}

		std::size_t get_byte_count() const noexcept {
			return this->get_size() * this->get_elem_size();
		}

		// Restore the attribute from a stream (CPU only)
		std::size_t restore(util::IByteReader& stream, std::size_t start, std::size_t count) {
			auto& handle = m_handles.template get<typename AttributePool<Device::CPU, !USES_OPENMESH>::template AttributeHandle<Type>>();
			std::size_t read = m_pools.template get<AttributePool<Device::CPU, !USES_OPENMESH>>().restore(handle, stream, start, count);
			if(read > 0u)
				m_flags.mark_changed(Device::CPU);
			return read;
		}

		// Store the attribute into a stream (CPU only)
		std::size_t store(util::IByteWriter& stream, std::size_t start, std::size_t count) const {
			const auto& handle = m_handles.template get<typename AttributePool<Device::CPU, !USES_OPENMESH>::template AttributeHandle<Type>>();
			return m_pools.template get<AttributePool<Device::CPU, !USES_OPENMESH>>().store(handle, stream, start, count);
		}

	private:
		// Initializes the attribute for all devices
		template < std::size_t I = 0u >
		void init_impl() {
			if constexpr(I < AttributePools::size) {
				m_handles.template get<I>() = m_pools.template get<I>().template add<Type>();
				this->init_impl<I + 1u>();
			}
		}

		// Initializes the attribute for all devices
		template < std::size_t I = 0u >
		void init_impl(OpenMesh::PropertyT<T>& prop) {
			if constexpr(I < AttributePools::size) {
				auto& pool = m_pools.template get<I>();
				constexpr Device DEVICE = std::decay_t<decltype(pool)>::DEVICE;
				if constexpr(DEVICE == Device::CPU) {
					m_handles.template get<I>() = pool.template add<Type>(prop);
				} else {
					m_handles.template get<I>() = pool.template add<Type>();
				}
				this->init_impl<I + 1u>();
			}
		}

		// Removes the attribute from all devices
		template < std::size_t I = 0u >
		void destroy_impl() {
			if constexpr(I < AttributePools::size) {
				m_pools.template get<I>().template remove<Type>(m_handles.template get<I>());
			}
		}

		AttributePools& m_pools;
		util::TaggedTuple<typename AttributePool<Device::CPU, !USES_OPENMESH>::template AttributeHandle<T>,
			typename AttributePool<Device::CUDA, true>::template AttributeHandle<T>> m_handles;
		util::DirtyFlags<Device>& m_flags;
	};

	// Adds a new attribute to the list, initially marked as absent
	template < class T, bool openMesh = USES_OPENMESH, typename = std::enable_if_t<!openMesh> >
	AttributeHandle<T> add(std::string name) {
		auto attr = this->find<T>(name);
		if(attr.has_value())
			return attr.value();

		// Create a new attribute
		m_attributes.push_back(std::make_unique<Attribute<T>>(m_attributePools, m_flags));
		return { m_attributes.size() - 1u };
	}

	// Adds a new attribute to the list, initially marked as absent
	template < class T, bool openMesh = USES_OPENMESH, typename = std::enable_if_t<openMesh> >
	AttributeHandle<T> add(std::string name, OpenMesh::PropertyT<T>& prop) {
		auto attr = this->find<T>(name);
		if(attr.has_value())
			return attr.value();

		// Create a new attribute
		m_attributes.push_back(std::make_unique<Attribute<T>>(m_attributePools, m_flags, prop));
		return { m_attributes.size() - 1u };
	}

	// Finds an attribute by its name and returns a handle to it
	template < class T >
	std::optional<AttributeHandle<T>> find(const std::string& name) const {
		auto iter = m_mapping.find(name);
		if(iter != m_mapping.cend()) {
			// Make sure the mapping is still valid
			if(iter->second >= m_attributes.size() || m_attributes[iter->second] == nullptr)
				return std::nullopt;
			return AttributeHandle<T>{iter->second};
		}
		return std::nullopt;
	}

	template < class T >
	void remove(const AttributeHandle<T>& handle) {
		if(handle.index() >= m_attributes.size() || m_attributes[handle.index()] == nullptr)
			return;
		m_attributes[handle.index()].reset();
		// Since we don't have a bimap, iterate all names
		for(auto iter = m_mapping.begin(); iter != m_mapping.end(); ++iter) {
			if(iter->second == handle.index()) {
				m_mapping.erase(iter);
				break;
			}
		}
	}

	// Returns the length of all attributes
	std::size_t get_size() const noexcept {
		return m_attributePools.template get<0u>().get_size();
	}

	template < Device dev >
	std::size_t get_bytes() const noexcept {
		return m_attributePools.template get<dev>().get_bytes();
	}

	// Resizes all attributes on all devices (only reallocs if present)
	void resize(std::size_t n) {
		m_attributePools.for_each([n](auto& elem) {
			elem.resize(n);
			return false;
		});
	}

	// Synchronizes all attributes on the given device from the last changed device
	template < Device dev = DEFAULT_DEVICE >
	void synchronize() {
		mufflon::synchronize<dev>(m_attributePools, m_flags, 
								  m_attributePools.template get<AttributePool<dev, stores_itself<dev>()>>());
	}

	// Unloads all attributes from the given device
	template < Device dev = DEFAULT_DEVICE >
	void unload() {
		// TODO: make sure that we have at least one loaded device
		m_attributePools.template get<AttributePool<dev>>().unload();
		m_flags.unload(dev);
	}

	template < Device dev = DEFAULT_DEVICE >
	void mark_changed() {
		m_flags.mark_changed(dev);
	}

	// Aquires an attribute from its handle
	template < class T >
	Attribute<T>& aquire(const AttributeHandle<T>& hdl) {
		mAssert(hdl.index() < m_attributes.size());
		return dynamic_cast<Attribute<T>&>(*m_attributes[hdl.index()]);
	}

	// Aquires an attribute from its handle
	template < class T >
	const Attribute<T>& aquire(const AttributeHandle<T>& hdl) const {
		mAssert(hdl.index() < m_attributes.size());
		return dynamic_cast<const Attribute<T>&>(*m_attributes[hdl.index()]);
	}

private:
	// Helper function to shorten template formulation
	template < Device dev >
	static constexpr bool stores_itself() {
		if constexpr(dev == Device::CPU)
			return !USES_OPENMESH;
		else
			return true;
	}

	AttributePools m_attributePools;
	std::vector<std::unique_ptr<IBaseAttribute>> m_attributes;
	util::DirtyFlags<Device> m_flags;
	std::unordered_map<std::string, std::size_t> m_mapping;
};

} // namespace mufflon::scene