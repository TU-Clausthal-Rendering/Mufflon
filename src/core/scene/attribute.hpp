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
	virtual std::size_t store(std::ostream&) = 0;

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
template < class T, Device defaultDev, template < template < Device, class > class,
												class, Device, Device > class Sync,
			template < Device, class > class Hdl >
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
		using Type = typename HandleType::Type;
		using ValueType = typename HandleType::ValueType;
		static constexpr Device DEVICE = HandleType::DEVICE;

		ValueType value;
	};

	static constexpr Device DEFAULT_DEVICE = defaultDev;
	using Type = T;
	template < Device dev >
	using DeviceHandleType = Hdl<dev, Type>;
	template < Device change, Device sync >
	using SynchronizeOps = Sync<Hdl, Type, change, sync>;
	template < Device dev >
	using DeviceValueType = DeviceValue<DeviceHandleType<dev>>;
	using DeviceValueTypes = util::TaggedTuple<DeviceValueType<Device::CPU>,
		DeviceValueType<Device::CUDA>>;
	using DefaultHandleType = DeviceHandleType<DEFAULT_DEVICE>;
	using DefaultValueType = DeviceValueType<DEFAULT_DEVICE>;

	virtual ~ISyncedAttribute() = default;

	// Aquire a read-only accessor
	template < Device dev = DEFAULT_DEVICE >
	ConstAccessor<DeviceHandleType<dev>> aquireConst() {
		this->synchronize<dev>();
		return ConstAccessor<DeviceHandleType<dev>>(&m_values.get<DeviceValueType<dev>>().value);
	}

	// Aquire a writing (and thus dirtying) accessor
	template < Device dev = DEFAULT_DEVICE >
	Accessor<DeviceHandleType<dev>> aquire() {
		this->synchronize<dev>();
		return Accessor<DeviceHandleType<dev>>(&m_values.get<DeviceValueType<dev>>().value, m_dirty);
	}

	// Explicitly synchronize the given device
	template < Device dev = DEFAULT_DEVICE >
	void synchronize() {
		if (m_dirty.has_competing_changes())
			throw std::runtime_error("Failure: competing changes for this attribute");
		if (m_dirty.has_changes()) {
			sync_impl<dev, 0u>(m_values, m_dirty, m_values.get<DeviceValueType<dev>>());
		}
	}

protected:
	template < Device dev >
	void mark_changed() {
		m_dirty.mark_changed(dev);
	}

	template < Device dev >
	DeviceValueType<dev> &get_value() {
		return m_values.get<DeviceValueType<dev>>();
	}

	template < Device dev >
	const DeviceValueType<dev> &get_value() const {
		return m_values.get<DeviceValueType<dev>>();
	}

private:
	// Helper function for iterating through all devices and finding one which has changes
	template < Device dev, std::size_t I >
	static void sync_impl(DeviceValueTypes &values, util::DirtyFlags<Device>& flags,
						  DeviceValueType<dev>& sync) {
		if constexpr (I < DeviceValueTypes::size) {
			// TODO!
			/*auto& changed = values.get<I>();
			constexpr Device CHANGED_DEVICE = DeviceValueTypes::Type<I>::DEVICE;
			// TODO
			//constexpr Device CHANGED_DEVICE = Device::CUDA;
			if (flags.has_changes(CHANGED_DEVICE)) {
				// Perform synchronization if necessary
				SynchronizeOps<CHANGED_DEVICE, dev>::synchronize(changed.value, sync.value);
			}
			else {
				// Keep looking for device changes
				sync_impl<dev, I + 1u>(values, flags, sync);
			}*/
		}
	}

	util::DirtyFlags<Device> m_dirty;
	DeviceValueTypes m_values;
};

// Struct for array synchronization operations
template < template < Device, class > class H, class T,
		Device change, Device syn >
struct ArraySynchronizeOps {
	static constexpr Device CHANGED_DEVICE = change;
	static constexpr Device SYNCED_DEVICE = syn;
	using Type = T;
	template < Device dev >
	using DeviceHandleType = H<dev, Type>;
	template < Device dev >
	using DeviceValueType = typename DeviceHandleType<dev>::ValueType;
	template < Device dev >
	using DeviceOps = DeviceArrayOps<dev, Type, H>;

	static void synchronize(DeviceValueType<CHANGED_DEVICE>& changed,
		DeviceValueType<SYNCED_DEVICE>& sync) {
		// Check size of the arrays and possibly resize
		std::size_t changedSize = DeviceOps<CHANGED_DEVICE>::get_size(changed);
		if (changedSize != DeviceOps<SYNCED_DEVICE>::get_size(sync))
			DeviceOps<SYNCED_DEVICE>::resize(sync, changedSize);
		// Copy over
		DeviceOps<SYNCED_DEVICE>::copy<CHANGED_DEVICE>(changed, sync);
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
					   public ISyncedAttribute<T, defaultDev, ArraySynchronizeOps,
												DeviceArrayHandle> {
public:
	static constexpr Device DEFAULT_DEVICE = defaultDev;
	using Type = T;
	template < Device dev >
	using ArrayOps = DeviceArrayOps<dev, Type, DeviceArrayHandle>;

	ArrayAttribute(std::string name)
		: IBaseAttribute(), m_name(std::move(name))
	{}

	virtual void reserve(std::size_t n) override {
		auto& value = this->get_value<DEFAULT_DEVICE>().value;
		if (n != ArrayOps<DEFAULT_DEVICE>::get_capacity(value)) {
			this->mark_changed<DEFAULT_DEVICE>();
			ArrayOps<DEFAULT_DEVICE>::reserve(value, n);
		}
	}

	virtual void resize(std::size_t n) override {
		if (n != this->n_elements()) {
			this->mark_changed<DEFAULT_DEVICE>();
			ArrayOps<DEFAULT_DEVICE>::resize(this->get_value<DEFAULT_DEVICE>().value, n);
		}
	}

	virtual void clear() override {
		if (this->n_elements() != 0u) {
			this->mark_changed<DEFAULT_DEVICE>();
			ArrayOps<DEFAULT_DEVICE>::clear(this->get_value<DEFAULT_DEVICE>().value);
		}
	}

	virtual std::size_t n_elements() const override {
		return ArrayOps<DEFAULT_DEVICE>::get_size(this->get_value<DEFAULT_DEVICE>().value);
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

		const std::size_t currSize = this->n_elements();
		this->resize(currSize + elems);
		auto accessor = this->aquire<DEFAULT_DEVICE>();
		Type* nativePtr = (*accessor)->data() + currSize;
		stream.read(reinterpret_cast<char*>(nativePtr), sizeof(Type) * elems);
		return static_cast<std::size_t>(stream.gcount()) / sizeof(Type);
	}

	virtual std::size_t store(std::ostream& stream) override {
		// TODO: read/write to other device?
		auto accessor = this->aquireConst<DEFAULT_DEVICE>();
		const Type* nativePtr = (*accessor)->data();
		stream.write(reinterpret_cast<const char*>(nativePtr), this->size_of());
		return this->n_elements();
	}

private:
	std::string m_name;
};

} // namespace mufflon::scene