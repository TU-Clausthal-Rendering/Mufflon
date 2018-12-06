#pragma once

#include "core/memory/accessor.hpp"
#include "attribute.hpp"
#include "core/memory/synchronize.hpp"
#include "util/assert.hpp"
#include "util/tagged_tuple.hpp"
#include "util/log.hpp"
#include <cstdlib>
#include <unordered_map>
#include <optional>
#include <vector>

namespace mufflon { namespace util {
class IByteReader;
}}

namespace mufflon { namespace scene {

// Helper structs containing the pool and handle types for the attribute lists
template < bool isFace >
struct OmAttributeListTypes {
	using AttributePools = util::TaggedTuple<OmAttributePool<isFace>, AttributePool<Device::CUDA>>;
	template < class T >
	using AttributeHandles = util::TaggedTuple<typename OmAttributePool<isFace>::template AttributeHandle<T>,
		typename AttributePool<Device::CUDA>::template AttributeHandle<T>>;
};
struct AttributeListTypes {
	using AttributePools = util::TaggedTuple<AttributePool<Device::CPU>, AttributePool<Device::CUDA>>;
	template < class T >
	using AttributeHandles = util::TaggedTuple<typename AttributePool<Device::CPU>::template AttributeHandle<T>,
		typename AttributePool<Device::CUDA>::template AttributeHandle<T>>;
};

/**
 * Base class for managed attributes and attribute pool access.
 * Access may happen via handles only, ie. one needs to aquire an attribute first.
 * Attribute pools may be synchronized to or unloaded from devices.
 */
template < Device defaultDev, class List, class Types >
class AttributeListBase {
public:
	static constexpr Device DEFAULT_DEVICE = defaultDev;
	using ListType = List;
	using AttributePools = typename Types::AttributePools;
	template < class T >
	using AttributeHandles = typename Types::template AttributeHandles<T>;

	// Base type for inheritance
	class IBaseAttribute {
	public:
		friend class AttributeListBase;
		virtual ~IBaseAttribute() {}

	protected:
		virtual void adjust_pool_pointers(AttributePools& pools) noexcept = 0;
	};

	template < class... Args >
	AttributeListBase(Args&& ...args) : m_attributePools(std::forward<Args>(args)...)
	{}

	AttributeListBase(const AttributeListBase&) = delete;
	//AttributeList(AttributeList&&) = default;
	AttributeListBase(AttributeListBase&& list) :
		m_attributePools(std::move(list.m_attributePools)),
		m_attributes(std::move(list.m_attributes)),
		m_flags(std::move(list.m_flags)),
		m_mapping(std::move(list.m_mapping)) {
		// Update the pool pointers in the attributes
		for(auto& attrib : m_attributes)
			attrib->adjust_pool_pointers(m_attributePools);
	}
	AttributeListBase& operator=(const AttributeListBase&) = delete;
	AttributeListBase& operator=(AttributeListBase&&) = default;
	~AttributeListBase() {
		// Unload all pools to avoid realloc calls from attribute destructors
		m_attributePools.for_each([](auto& pool) {
			pool.unload();
		});
		// Ensure that the order of attribute destruction is back-to-front
		// to avoid unnecessary allocations in the memory pools
		const std::size_t destroyNAttribs = m_attributes.size();
		for(std::size_t i = 0; i < destroyNAttribs; ++i)
			m_attributes.pop_back();
	}

	template < class T >
	class AttributeHandle {
	public:
		using Type = T;

		AttributeHandle(std::size_t idx) :
			m_index(idx) {}

		constexpr std::size_t index() const noexcept {
			return m_index;
		}

	private:
		std::size_t m_index;
	};

	/**
	 * Attribute class granting access to the actual memory.
	 */
	template < class T >
	class BaseAttribute : public IBaseAttribute {
	public:
		using Type = T;
		friend class AttributeListBase;

		// Hide the construction from public eye
		BaseAttribute(AttributePools& pools, util::DirtyFlags<Device>& flags) :
			m_pools(&pools),
			m_flags(flags) {
			// In the inheriting attribute you MUST initialize the handles!
		}

		BaseAttribute(const BaseAttribute&) = delete;
		BaseAttribute(BaseAttribute&&) = default;
		BaseAttribute& operator=(const BaseAttribute&) = delete;
		BaseAttribute& operator=(BaseAttribute&&) = default;

		virtual ~BaseAttribute() {
			this->destroy_impl<0u>();
		}

		// Aquires a read-write accessor to the attribute
		template < Device dev = DEFAULT_DEVICE >
		auto aquire() {
			mAssert(m_pools != nullptr);
			this->synchronize<dev>();
			auto& pool = ListType::template getPool<dev>(*m_pools);
			using PoolType = std::decay_t<decltype(pool)>;
			constexpr std::size_t POOL_INDEX = m_pools->template get_index<PoolType>();
			auto& handle = m_handles.template get<POOL_INDEX>();
			return Accessor<ArrayDevHandle<dev, Type>>{ pool.aquire(handle), m_flags };
		}

		// Aquires a read-only accessor to the attribute
		template < Device dev = DEFAULT_DEVICE >
		auto aquireConst() {
			mAssert(m_pools != nullptr);
			this->synchronize<dev>();
			const auto& pool = ListType::template getPool<dev>(*m_pools);
			using PoolType = std::decay_t<decltype(pool)>;
			constexpr std::size_t POOL_INDEX = m_pools->template get_index<PoolType>();
			const auto& handle = m_handles.template get<POOL_INDEX>();
			return ConstAccessor<ArrayDevHandle<dev, Type>>{ pool.aquireConst(handle) };
		}

		// Synchronizes the attribute pool to the given device
		template < Device dev = DEFAULT_DEVICE >
		void synchronize() {
			mAssert(m_pools != nullptr);
			mufflon::synchronize<dev>(*m_pools, m_flags, ListType::template getPool<dev>(*m_pools));
		}

		std::size_t get_size() const noexcept {
			mAssert(m_pools != nullptr);
			return m_pools->template get<0>().get_size();
		}

		constexpr std::size_t get_elem_size() const noexcept {
			return sizeof(T);
		}

		std::size_t get_byte_count() const noexcept {
			return this->get_size() * this->get_elem_size();
		}

		// Restore the attribute from a stream (CPU only)
		std::size_t restore(util::IByteReader& stream, std::size_t start, std::size_t count) {
			mAssert(m_pools != nullptr);
			auto& pool = ListType::template getPool<DEFAULT_DEVICE>(*m_pools);
			using PoolType = std::decay_t<decltype(pool)>;
			constexpr std::size_t POOL_INDEX = m_pools->template get_index<PoolType>();
			const auto& handle = m_handles.template get<POOL_INDEX>();
			std::size_t read = pool.restore<T>(handle, stream, start, count);
			if(read > 0u)
				m_flags.mark_changed(Device::CPU);
			return read;
		}

		// Store the attribute into a stream (CPU only)
		std::size_t store(util::IByteWriter& stream, std::size_t start, std::size_t count) const {
			auto& pool = ListType::template getPool<DEFAULT_DEVICE>(*m_pools);
			using PoolType = std::decay_t<decltype(pool)>;
			constexpr std::size_t POOL_INDEX = m_pools->template get_index<PoolType>();
			const auto& handle = m_handles.template get<POOL_INDEX>();
			return pool.store<T>(handle, stream, start, count);
		}

	protected:
		// We need this for move construction
		virtual void adjust_pool_pointers(AttributePools& pools) noexcept override {
			m_pools = &pools;
		}

		AttributePools* m_pools;
		AttributeHandles<T> m_handles;
		util::DirtyFlags<Device>& m_flags;

	private:
		// Removes the attribute from all devices
		template < std::size_t I = 0u >
		void destroy_impl() {
			if constexpr(I < AttributePools::size) {
				mAssert(m_pools != nullptr);
				m_pools->template get<I>().template remove<Type>(m_handles.template get<I>());
			}
		}
	};

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
				--m_numAttributes;
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
		ListType::template getPool<dev>(m_attributePools).get_bytes();
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
		mufflon::synchronize<dev>(m_attributePools, m_flags, ListType::template getPool<dev>(m_attributePools));
	}

	// Unloads all attributes from the given device
	template < Device dev = DEFAULT_DEVICE >
	void unload() {
		// TODO: make sure that we have at least one loaded device
		if(m_flags.is_last_present(dev)) {
			logError("[AttributeList::unload] Cannot unload the last present device");
		} else {
			ListType::template getPool<dev>(m_attributePools).unload();
			m_flags.unload(dev);
		}
	}

	template < Device dev = DEFAULT_DEVICE >
	void mark_changed() {
		m_flags.mark_changed(dev);
	}

	std::size_t get_num_attributes() const noexcept {
		return m_numAttributes;
	}

	// Aquires an attribute from its handle
	template < class T >
	BaseAttribute<T>& aquire(const AttributeHandle<T>& hdl) {
		mAssert(hdl.index() < m_attributes.size());
		return dynamic_cast<BaseAttribute<T>&>(*m_attributes[hdl.index()]);
	}

	// Aquires an attribute from its handle
	template < class T >
	const BaseAttribute<T>& aquire(const AttributeHandle<T>& hdl) const {
		mAssert(hdl.index() < m_attributes.size());
		return dynamic_cast<const BaseAttribute<T>&>(*m_attributes[hdl.index()]);
	}

protected:
	AttributePools m_attributePools;
	std::vector<std::unique_ptr<IBaseAttribute>> m_attributes;
	std::size_t m_numAttributes = 0u;
	util::DirtyFlags<Device> m_flags;
	std::unordered_map<std::string, std::size_t> m_mapping;
};

/**
 * Specialized attribute class for using OpenMesh-properties on CPU side.
 * isFace specifies whether the attribute is a face attribute or a vertex attribute.
 */
template < bool isFace, Device defaultDev = Device::CPU >
class OmAttributeList : public AttributeListBase<defaultDev, OmAttributeList<isFace, defaultDev>, OmAttributeListTypes<isFace>> {
public:
	static constexpr Device DEFAULT_DEVICE = defaultDev;
	static constexpr bool IS_FACE = isFace;
	using BaseType = AttributeListBase<defaultDev, OmAttributeList<isFace, defaultDev>, OmAttributeListTypes<isFace>>;
	using AttributePools = typename BaseType::AttributePools;
	template < class T >
	using AttributeHandle = typename BaseType::template AttributeHandle<T>;
	template < class T >
	using PropType = typename OmAttributePool<IS_FACE>::template PropType<T>;

	OmAttributeList(geometry::PolygonMeshType& mesh) :
		BaseType{ OmAttributePool<IS_FACE>{mesh}, AttributePool<Device::CUDA>{} }
	{}
	OmAttributeList(const OmAttributeList&) = delete;
	OmAttributeList(OmAttributeList&&) = default;
	OmAttributeList& operator=(const OmAttributeList&) = delete;
	OmAttributeList& operator=(OmAttributeList&&) = default;
	~OmAttributeList() = default;

	template < class T >
	class Attribute : public BaseType::template BaseAttribute<T> {
	public:
		using BaseAttr = typename BaseType::template BaseAttribute<T>;

		Attribute(AttributePools& pools, util::DirtyFlags<Device>& flags,
					PropType<T> hdl) :
			BaseType::template BaseAttribute<T>(pools, flags)
		{
			this->init_impl<0u>(hdl);
		}
	private:
		// Initializes the attribute for all devices
		template < std::size_t I = 0u >
		void init_impl(PropType<T> hdl) {
			if constexpr(I < AttributePools::size) {
				auto& pool = BaseAttr::m_pools->template get<I>();
				constexpr Device DEVICE = std::decay_t<decltype(pool)>::DEVICE;
				if constexpr(DEVICE == Device::CPU) {
					BaseAttr::m_handles.template get<I>() = pool.template add<T>(hdl);
				} else {
					BaseAttr::m_handles.template get<I>() = pool.template add<T>();
				}
				this->init_impl<I + 1u>(hdl);
			}
		}
	};

	// Adds a new attribute to the list, initially marked as absent
	template < class T >
	AttributeHandle<T> add(std::string name, PropType<T> hdl) {
		auto attr = this->template find<T>(name);
		if(attr.has_value())
			return attr.value();

		// Create a new attribute
		BaseType::m_attributes.push_back(std::make_unique<Attribute<T>>(BaseType::m_attributePools,
																		BaseType::m_flags, hdl));
		++BaseType::m_numAttributes;
		return { BaseType::m_attributes.size() - 1u };
	}

	template < Device dev >
	static auto& getPool(AttributePools& pool) {
		if constexpr(dev == Device::CPU)
			return pool.template get<OmAttributePool<IS_FACE>>();
		else
			return pool.template get<AttributePool<dev>>();
	}
};

/**
 * Specialization for regular attributes which manage their own mememory.
 */
template < Device defaultDev = Device::CPU >
class AttributeList : public AttributeListBase<defaultDev, AttributeList<defaultDev>, AttributeListTypes> {
public:
	static constexpr Device DEFAULT_DEVICE = defaultDev;
	using BaseType = AttributeListBase<defaultDev, AttributeList<defaultDev>, AttributeListTypes>;
	using AttributePools = typename BaseType::AttributePools;
	template < class T >
	using AttributeHandle = typename BaseType::template AttributeHandle<T>;


	AttributeList() = default;
	AttributeList(const AttributeList&) = delete;
	AttributeList(AttributeList&&) = default;
	AttributeList& operator=(const AttributeList&) = delete;
	AttributeList& operator=(AttributeList&&) = default;
	~AttributeList() = default;

	template < class T >
	class Attribute : public BaseType::template BaseAttribute<T> {
	public:
		using BaseAttr = typename BaseType::template BaseAttribute<T>;

		Attribute(AttributePools& pools, util::DirtyFlags<Device>& flags) :
			BaseType::template BaseAttribute<T>(pools, flags) {
			this->init_impl<0u>();
		}
	private:
		// Initializes the attribute for all devices
		template < std::size_t I = 0u >
		void init_impl() {
			if constexpr(I < AttributePools::size) {
				auto& pool = BaseAttr::m_pools->template get<I>();
				BaseAttr::m_handles.template get<I>() = pool.template add<T>();
				this->init_impl<I + 1u>();
			}
		}
	};

	// TODO
	// Adds a new attribute to the list, initially marked as absent
	template < class T >
	AttributeHandle<T> add(std::string name) {
		auto attr = this->template find<T>(name);
		if(attr.has_value())
			return attr.value();

		// Create a new attribute
		BaseType::m_attributes.push_back(std::make_unique<Attribute<T>>(BaseType::m_attributePools,
																		BaseType::m_flags));
		++BaseType::m_numAttributes;
		return { BaseType::m_attributes.size() - 1u };
	}

	template < Device dev >
	static auto& getPool(AttributePools& pool) {
		return pool.template get<AttributePool<dev>>();
	}
};

}} // namespace mufflon::scene