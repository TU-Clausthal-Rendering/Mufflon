#pragma once

#include "accessor.hpp"
#include "attribute.hpp"
#include "util/assert.hpp"
#include "util/tagged_tuple.hpp"
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

	template < class T >
	class AttributeHandle {
		using Type = T;
		friend class AttributeList<USES_OPENMESH, DEFAULT_DEVICE>;

	private:
		AttributeHandle(std::size_t idx) :
			m_index(idx)
		{}

		constexpr std::size_t index() const noexcept {
			return m_index;
		}

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
	template < class T, bool usesOpenMesh = USES_OPENMESH >
	class Attribute : public IBaseAttribute {
	public:
		using Type = T;
		static constexpr bool USES_OPENMESH = usesOpenMesh;

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
			m_pools.for_each([](std::size_t i, auto& pool) {
				pool.remove<Type>();
				return false;
			});
		}

		// Aquires a read-write accessor to the attribute
		template < Device dev = DEFAULT_DEVICE >
		auto aquire() {
			using DeviceHdl = DeviceArrayHandle<dev, Type>;
			this->synchronize<dev>();
			auto& pool = m_pools.get<AttributePool<dev, stores_itself<dev>()>>();
			auto& handle = m_handles.get<typename AttributePool<dev, stores_itself<dev>()>::template AttributeHandle<T>>();
			auto attr = pool.aquire(handle);

			return Accessor<DeviceHdl>{ static_cast<typename DeviceHdl::HandleType>(attr), m_flags };
		}

		// Aquires a read-only accessor to the attribute
		template < Device dev = DEFAULT_DEVICE >
		auto aquireConst() const {
			using DeviceHdl = DeviceArrayHandle<dev, Type>;
			this->synchronize<dev>();
			const auto& pool = m_pools.get<AttributePool<dev, stores_itself<dev>()>>();
			const auto& handle = m_handles.get<typename AttributePool<dev, stores_itself<dev>()>::template AttributeHandle<T>>();
			const auto attr = pool.aquire(handle);

			return ConstAccessor<DeviceHdl>{ static_cast<const typename DeviceHdl::HandleType>(attr), m_flags };
		}

		// Synchronizes the attribute pool to the given device
		template < Device dev = DEFAULT_DEVICE >
		void synchronize() {
			if(m_flags.needs_sync(dev)) {
				if(m_flags.has_competing_changes())
					throw std::runtime_error("Competing changes for attribute detected!");
				// Synchronize
				this->synchronize_impl<0u, dev>();
			}
		}

		// Restore the attribute from a stream (CPU only)
		std::size_t restore(std::istream& stream, std::size_t start, std::size_t count) {
			auto& handle = m_handles.get<typename AttributePool<Device::CPU, !USES_OPENMESH>::template AttributeHandle<Type>>();
			return m_pools.get<AttributePool<Device::CPU, !USES_OPENMESH>>().restore(handle, stream, start, count);
		}

		// Store the attribute into a stream (CPU only)
		std::size_t store(std::ostream& stream, std::size_t start, std::size_t count) const {
			const auto& handle = m_handles.get<typename AttributePool<Device::CPU, !USES_OPENMESH>::template AttributeHandle<Type>>();
			return m_pools.get<AttributePool<Device::CPU, !USES_OPENMESH>>().store(handle, stream, start, count);
		}

	private:
		// Initializes the attribute for all devices
		template < std::size_t I >
		void init_impl() {
			if constexpr(I < AttributePools::size) {
				m_handles.get<I>() = m_pools.get<I>().add<T>();
				this->init_impl<I + 1u>();
			}
		}

		// Initializes the attribute for all devices
		template < std::size_t I >
		void init_impl(OpenMesh::PropertyT<T>& prop) {
			if constexpr(I < AttributePools::size) {
				auto& pool = m_pools.get<I>();
				constexpr Device DEVICE = std::decay_t<decltype(pool)>::DEVICE;
				if constexpr(DEVICE == Device::CPU) {
					m_handles.get<I>() = pool.add<T>(prop);
				} else {
					m_handles.get<I>() = pool.add<T>();
				}
				this->init_impl<I + 1u>();
			}
		}

		// Checks all devices for changes and syncs upon finding one
		template < std::size_t I, Device dev >
		void synchronize_impl() {
			if constexpr(I < AttributePools::size) {
				// Workaround for VS2017 bug: otherwise you may use the 'Type' template of the
				// tagged tuple
				auto& changed = m_pools.get<I>();
				constexpr Device CHANGED_DEVICE = std::decay_t<decltype(changed)>::DEVICE;
				if(m_flags.has_changes(CHANGED_DEVICE)) {
					changed.synchronize<dev>(m_pools.get<AttributePool<dev, stores_itself<dev>()>>());
				} else {
					this->synchronize_impl<I + 1u, dev>();
				}
			}
		}

		// Helper function to shorten template formulation
		template < Device dev >
		static constexpr bool stores_itself() {
			if constexpr(dev == Device::CPU)
				return !USES_OPENMESH;
			else
				return true;
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
		m_attributes.push_back(std::make_unique<Attribute<T, USES_OPENMESH>>(m_attributePools, m_flags));
		return { m_attributes.size() - 1u };
	}

	// Adds a new attribute to the list, initially marked as absent
	template < class T, bool openMesh = USES_OPENMESH, typename = std::enable_if_t<openMesh> >
	AttributeHandle<T> add(std::string name, OpenMesh::PropertyT<T>& prop) {
		auto attr = this->find<T>(name);
		if(attr.has_value())
			return attr.value();

		// Create a new attribute
		m_attributes.push_back(std::make_unique<Attribute<T, USES_OPENMESH>>(m_attributePools, m_flags, prop));
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
		return m_attributePools.get<0u>().get_size();
	}

	template < Device dev >
	std::size_t get_bytes() const noexcept {
		return m_attributePools.get<dev>().get_bytes();
	}

	// Resizes all attributes on all devices (only reallocs if present)
	void resize(std::size_t n) {
		m_attributePools.for_each([n](std::size_t i, auto& elem) {
			elem.resize(n);
			return false;
		});
	}

	// Synchronizes all attributes on the given device from the last changed device
	template < Device dev = DEFAULT_DEVICE >
	void synchronize() {
		// TODO
		if(m_flags.needs_sync(dev)) {
			if(m_flags.has_competing_changes())
				throw std::runtime_error("Competing changes for attribute detected!");
			this->synchronize_impl<0u, dev>();
		}
	}

	// Unloads all attributes from the given device
	template < Device dev = DEFAULT_DEVICE >
	void unload() {
		m_attributePools.get<AttributePool<dev>>().unload();
	}

	template < Device dev = DEFAULT_DEVICE >
	void mark_changed() {
		m_flags.mark_changed(dev);
	}

	// Aquires an attribute from its handle
	template < class T >
	Attribute<T, USES_OPENMESH>& aquire(const AttributeHandle<T>& hdl) {
		mAssert(hdl.index() < m_attributes.size());
		return dynamic_cast<Attribute<T, USES_OPENMESH>&>(*m_attributes[hdl.index()]);
	}

	// Aquires an attribute from its handle
	template < class T >
	const Attribute<T, USES_OPENMESH>& aquire(const AttributeHandle<T>& hdl) const {
		mAssert(hdl.index() < m_attributes.size());
		return dynamic_cast<const Attribute<T, USES_OPENMESH>&>(*m_attributes[hdl.index()]);
	}

private:
	// Recursing implementation for synchronization
	template < std::size_t I, Device dev >
	void synchronize_impl() {
		if constexpr(I < AttributePools::size) {
			// Workaround for VS2017 bug: otherwise you may use the 'Type' template of the
			// tagged tuple
			auto& changed = m_attributePools.get<I>();
			constexpr Device CHANGED_DEVICE = std::decay_t<decltype(changed)>::DEVICE;
			if(m_flags.has_changes(CHANGED_DEVICE)) {
				changed.synchronize<dev>(m_attributePools.get<AttributePool<dev>>());
			} else {
				synchronize_impl<I + 1u, dev>();
			}
		}
	}

	AttributePools m_attributePools;
	std::vector<std::unique_ptr<IBaseAttribute>> m_attributes;
	util::DirtyFlags<Device> m_flags;
	std::unordered_map<std::string, std::size_t> m_mapping;
};

} // namespace mufflon::scene