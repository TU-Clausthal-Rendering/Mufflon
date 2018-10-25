#pragma once

#include "residency.hpp"
#include "accessor.hpp"
#include "util/assert.hpp"
#include <cstddef>
#include <istream>
#include <ostream>
#include <string>
#include <string_view>
#include <vector>

namespace mufflon::scene {

/**
	* Base class for all attribtues.
	* Any attribute which should be used in an attribute array needs to inherit
	* from this.
	* Exceptioned from the "getter get 'get' prefix"-rule due to OpenMesh's
	* 'BaseProperty' interface.
	*/
class IBaseAttribute {
public:
	IBaseAttribute() = default;
	IBaseAttribute(const IBaseAttribute&) = default;
	IBaseAttribute(IBaseAttribute&&) = default;
	IBaseAttribute& operator=(const IBaseAttribute&) = default;
	IBaseAttribute& operator=(IBaseAttribute&&) = default;
	virtual ~IBaseAttribute() = default;

	// Methods that we and OpenMesh both care about
	virtual void reserve(std::size_t count) = 0;
	virtual void resize(std::size_t count) = 0;
	virtual void clear() = 0;
	virtual std::size_t n_elements() const = 0;
	virtual std::size_t element_size() const = 0;
	virtual std::size_t size_of() const = 0; // Size in bytes
	virtual std::size_t size_of(std::size_t n) const = 0; // Estimated for no of elems
	virtual const std::string& name() const = 0;

	// Methods that we both like, but disagree on the interface
	virtual std::size_t restore(std::istream&, std::size_t elems) = 0;
	virtual std::size_t store(std::ostream&) const = 0;

	// Methods that OpenMesh would like to have (for future reference)
	//virtual void push_back() = 0;
	//virtual void swap(std::size_t i0, std::size_t i1) = 0;
	//virtual void copy(std::size_t i0, std::size_t i1) = 0;
	//virtual IBaseAttribute* clone() const = 0;
	//virtual void set_persistent() = 0;
};

/**
 * SyncAttribute
 */
template < class Attr, template < Device, Device > class Sync,
			template < Device > class Hdl >
class ISyncedAttribute {
public:
	/**
	 * Helper struct needed for storing the actual value behind a handle.
	 * Otherwise we would be storing a pointer and would have to allocate
	 * them manually in the tuple.
	 */
	template < class H >
	struct DeviceValue {
		using HandleType = H;
		using Type = typename::type_info;
		using ValueType = typename HandleType::ValueType;

		ValueType value;
	};

	using AttributeType = Attr;
	static constexpr Device DEFAULT_DEVICE = AttributeType::DEFAULT_DEVICE;
	using Type = typename AttributeType::T;
	template < Device dev >
	using DeviceHandleType = Hdl<dev>;
	template < Device changed, Device sync >
	using SynchronizeOps = Sync<changed, sync>;
	template < Device dev >
	using DeviceValueType = DeviceValue<DeviceHandleType<dev>>;
	using DeviceValueTypes = util::TaggedTuple<DeviceValueType<Device::CPU>,
		DeviceValueType<Device::CUDA>>;
	using DefaultHandleType = DeviceHandleType<DEFAULT_DEVICE>;
	using DefaultValueType = DeviceValueType<DEFAULT_DEVICE>;

	// Aquire a read-only accessor
	template < Device dev = DEFAULT_DEVICE >
	ConstAccessor<Type, dev> aquireConst() {
		this->synchronize<dev>();
		return ConstAccessor<dev>(&m_value.get<DeviceValueType<dev>>().value);
	}

	// Aquire a writing (and thus dirtying) accessor
	template < Device dev = DEFAULT_DEVICE >
	Accessor<Type, dev> aquire() {
		this->synchronize<dev>();
		return Accessor<dev>(&m_value.get<DeviceValueType<dev>>().value, m_dirty);
	}

	// Explicitly synchronize the given device
	template < Device dev = DEFAULT_DEVICE >
	void synchronize() {
		if (m_dirty.has_competing_changes())
			throw std::runtime_error("Failure: competing changes for this attribute");
		if (m_dirty.has_changes()) {
			SyncHelper<0u>::synchronize(m_value, m_dirty, m_value.get<DeviceValueType<dev>>());
		}
	}

private:
	// Helper struct for iterating through all devices and finding one which has changes
	template < std::size_t I >
	struct SyncHelper {
		template < Device dev >
		static void synchronize(DeviceValueTypes &values, util::DirtyFlags<Device>& flags,
			DefaultValueType& sync) {
			if constexpr (I < HandleTypes::size) {
				auto& changed = values.get<I>();
				constexpr Device CHANGED_DEVICE = changed.DEVICE;
				if (flags.has_changes(CHANGED_DEVICE)) {
					// Perform synchronization if necessary
					SynchronizeOps<CHANGED_DEVICE, dev>::synchronize(changed, sync);
				}
				else {
					// Keep looking for device changes
					SyncHelper<I + 1u>::synchronize(values, flags, sync);
				}
			}
		}
	};

	util::DirtyFlags<Device> m_dirty;
	DeviceValueTypes m_value;
};

// Struct for array synchronization operations
template < template < Device, class > class V, class T,
		Device change, Device sync >
struct ArraySynchronizeOps {
	using Type = T;
	template < Device dev >
	using DeviceValueType = V<dev1, Type>;
	static constexpr Device CHANGED_DEVICE = change;
	static constexpr Device CHANGED_DEVICE = sync;

	static void synchronize(DeviceValueType<CHANGED_DEVICE>& changed,
		DeviceValueType<SYNCED_DEVICE>& sync) {
		std::size_t changedSize = ArrayOps<CHANGED_DEVICE>::get_size(changed);
		if (changedSize != ArrayOps<dev>::get_size(sync))
			ArrayOps<dev>::resize(sync, changedSize);
		ArrayOps<dev>::copy(changed, sync);
	}
};

/**
 * Array attribute class.
 * Contains handles for all devices.
 * Can hand out read or write access.
 * Synchronizes between devices.
 */
template < class T, Device defaultDev = Device::CPU >
class ArrayAttribute : public IBaseAttribute,
					   public ISyncedAttribute<ArrayAttribute, ArraySynchronizeOps,
												DeviceHandleType> {
public:
	static constexpr Device DEFAULT_DEVICE = Device::CPU;
	using Type = T;
	template < Device dev >
	using DeviceHandleType = DeviceArrayHandle<dev, Type>;

	virtual void reserve(std::size_t n) override {
		if (n != this->get_capacity())
			m_dirty.mark_changed(DEFAULT_DEVICE);
		return DeviceArrayOps<DEFAULT_DEVICE>::reserve(m_handles.get<DefaultHandleType>(), n);
	}

	virtual void resize(std::size_t n) override {
		if (n != this->get_capacity())
			m_dirty.mark_changed(DEFAULT_DEVICE);
		return DeviceArrayOps<DEFAULT_DEVICE>::resize(m_handles.get<DefaultHandleType>(), n);
	}

	virtual void clear() override {
		DeviceArrayOps<DEFAULT_DEVICE>::clear(m_handles.get<DefaultHandleType>());
	}

	virtual std::size_t n_elements() const override {
		return DeviceArrayOps<DEFAULT_DEVICE>::get_size(m_handles.get<DefaultHandleType>());
	}

	virtual std::size_t element_size() const override {
		return sizeof(Type);
	}

	virtual std::size_t size_of() const override {
		return this->element_size() * this->n_elements();
	}

	virtual std::size_t size_of(std::size_t n) const override {
		return this->element_size() * n;
	}

	virtual const std::string& name() const noexcept override {
		return m_name;
	}

	virtual std::size_t restore(std::istream& stream, std::size_t elems) override {
		if (elems == 0)
			return 0u;

		const std::size_t currSize = this->get_size();
		this->resize(currSize + elems);
		Accessor<Type, DEFAULT_DEVICE> accessor = this->aquire<DEFAULT_DEVICE>();
		Type* nativePtr = &(*accessor)[currSize];
		stream.read(reinterpret_cast<char*>(nativePtr), sizeof(Type) * elems);
		m_dirty.mark_changed(Device::CPU);
		return static_cast<std::size_t>(stream.gcount()) / sizeof(Type);
	}

	virtual std::size_t store(std::ostream& stream) const override {
		// TODO: read/write to other device?
		ConstAccessor<Type, DEFAULT_DEVICE> accessor = this->aquireConst<DEFAULT_DEVICE>();
		const Type* nativePtr = &(*accessor)[currSize];
		stream.write(reinterpret_cast<const char*>(nativePtr), sizeof(Type) * elems);
		return elems;
	}

private:
	std::string m_name;
};

} // namespace mufflon::scene