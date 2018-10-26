#pragma once

#include <OpenMesh/Core/Utils/Property.hh>
#include "residency.hpp"
#include "accessor.hpp"
#include "util/assert.hpp"
#include <cstddef>
#include <istream>
#include <ostream>
#include <string>
#include <string_view>
#include <vector>
#include <utility>

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
class SyncedAttribute {
public:
	static constexpr Device DEFAULT_DEVICE = defaultDev;
	using Type = T;
	template < Device dev >
	using DeviceHandleType = Hdl<dev, Type>;
	template < Device change, Device sync >
	using SynchronizeOps = Sync<Hdl, Type, change, sync>;

	using DeviceHandleTypes = util::TaggedTuple<DeviceHandleType<Device::CPU>,
		DeviceHandleType<Device::CUDA>>;
	using DefaultHandleType = DeviceHandleType<DEFAULT_DEVICE>;

	SyncedAttribute(DeviceHandleTypes hdls) :
		m_handles(std::move(hdls))
	{}

	virtual ~SyncedAttribute() = default;

	// Aquire a read-only accessor
	template < Device dev = DEFAULT_DEVICE >
	ConstAccessor<DeviceHandleType<dev>> aquireConst() {
		this->synchronize<dev>();
		return ConstAccessor<DeviceHandleType<dev>>(m_handles.get<DeviceHandleType<dev>>().handle);
	}

	// Aquire a writing (and thus dirtying) accessor
	template < Device dev = DEFAULT_DEVICE >
	Accessor<DeviceHandleType<dev>> aquire() {
		this->synchronize<dev>();
		return Accessor<DeviceHandleType<dev>>(m_handles.get<DeviceHandleType<dev>>().handle, m_dirty);
	}

	// Explicitly synchronize the given device
	template < Device dev = DEFAULT_DEVICE >
	void synchronize() {
		if (m_dirty.has_competing_changes())
			throw std::runtime_error("Failure: competing changes for this attribute");
		if (m_dirty.has_changes()) {
			sync_impl<dev, 0u>(m_handles, m_dirty, m_handles.get<DeviceHandleType<dev>>());
		}
	}

protected:
	template < Device dev >
	void mark_changed() {
		m_dirty.mark_changed(dev);
	}

	template < Device dev >
	DeviceHandleType<dev> get_handle() const {
		return m_handles.get<DeviceHandleType<dev>>();
	}

private:
	// Helper function for iterating through all devices and finding one which has changes
	template < Device dev, std::size_t I >
	static void sync_impl(DeviceHandleTypes& values, util::DirtyFlags<Device>& flags,
						DeviceHandleType<dev>& sync) {
		if constexpr (I < DeviceHandleTypes::size) {
			// TODO!
			auto& changed = values.get<I>();
			constexpr Device CHANGED_DEVICE = std::decay_t<decltype(changed)>::DEVICE;
			if (flags.has_changes(CHANGED_DEVICE)) {
				// Perform synchronization if necessary
				SynchronizeOps<CHANGED_DEVICE, dev>::synchronize(changed, sync);
			}
			else {
				// Keep looking for device changes
				sync_impl<dev, I + 1u>(values, flags, sync);
			}
		}
	}

	util::DirtyFlags<Device> m_dirty;
	DeviceHandleTypes m_handles;
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
	using DeviceOps = DeviceArrayOps<dev, Type, H>;

	static void synchronize(DeviceHandleType<CHANGED_DEVICE>& changed,
		DeviceHandleType<SYNCED_DEVICE>& sync) {
		// Check size of the arrays and possibly resize
		std::size_t changedSize = DeviceOps<CHANGED_DEVICE>::get_size(changed);
		if (changedSize != DeviceOps<SYNCED_DEVICE>::get_size(sync))
			DeviceOps<SYNCED_DEVICE>::resize(sync, changedSize);
		// Copy over
		DeviceOps<SYNCED_DEVICE>::copy<CHANGED_DEVICE>(changed, sync);
	}
};

/** Struct for storing device values (purely necessary because base-class
 * constructors need to be called before any member initialization is
 * possible).
 * If we're not using OpenMesh, which keeps the value somewhere internally,
 * we allocate an additional data member to use.
 */
template < class T, class DHs, bool storesCpu = true >
struct ArrayAttributeValues {
	using Type = T;
	using DeviceHandleTypes = DHs;

	// Helper struct for storing device values
	template < Device dev >
	struct DeviceValue {
		using Type = T;
		using HandleType = DeviceArrayHandle<dev, Type>;
		// Special casing for CPU (due to OpenMesh): only store reference sometimes
		using ValueType = std::conditional_t<!storesCpu && dev == Device::CPU,
			typename HandleType::ValueType*, typename HandleType::ValueType>;
		static constexpr Device DEVICE = dev;

		ValueType value;
	};

	// Helper struct for extracting the handles for values
	template < class >
	struct DeviceValueHelper;
	template < std::size_t... Is >
	struct DeviceValueHelper<std::index_sequence<Is...>> {
		// Allows us to access the individual types
		template < std::size_t N >
		using DeviceHandleType = typename DeviceHandleTypes::template Type<N>;
		
		template < std::size_t N >
		static constexpr Device DEVICE = DeviceHandleType<N>::DEVICE;

		// Gives us the equivalent value type for a handle
		template < std::size_t N >
		using DeviceValueType = DeviceValue<DeviceHandleType<N>::DEVICE>;

		using DeviceValueTypes = util::TaggedTuple<DeviceValueType<Is>...>;

		static typename DeviceHandleTypes get_handles(DeviceValueTypes& values) {
			return {get_handle<Is>(values)...};
		}

	private:
		template < std::size_t N >
		static auto get_handle(DeviceValueTypes& values) {
			// Make a distinction: if we don't store the CPU part, we only have a pointer,
			// and taking a pointer of a pointer is pointless
			if constexpr(!storesCpu && DEVICE<N> == Device::CPU)
				return DeviceArrayHandle<DEVICE<N>, DeviceValueType<N>::Type>(values.get<N>().value);
			else
				return DeviceArrayHandle<DEVICE<N>, DeviceValueType<N>::Type>(&values.get<N>().value);
		}
	};

	// For external storage so that you may pass the handles into the constructor
	using DeviceIndices = std::make_index_sequence<DeviceHandleTypes::size>;
	using DeviceValueTypes = typename DeviceValueHelper<DeviceIndices>::DeviceValueTypes;

	template < bool cpuSide = storesCpu, typename = std::enable_if_t<cpuSide> >
	ArrayAttributeValues() {}

	// Constructor (only if we're in the special case scenario of OpenMesh
	template < bool cpuSide = storesCpu, typename Ref = std::enable_if_t<!cpuSide, typename DeviceArrayHandle<Device::CPU, T>::ValueType&> >
	ArrayAttributeValues(Ref ref) {
		values.get<DeviceValue<Device::CPU>>().value = &ref;
	}

	// Actual attribute values
	DeviceValueTypes values;
};

/**
 * Custom array attribute class.
 * Contains handles for all devices.
 * Can hand out read or write access.
 * Synchronizes between devices.
 */
template < class T, bool storesCpu = true, Device defaultDev = Device::CPU >
class ArrayAttribute : public IBaseAttribute,
					   protected ArrayAttributeValues<T, typename SyncedAttribute<T,
												   defaultDev, ArraySynchronizeOps,
												   DeviceArrayHandle>::DeviceHandleTypes, storesCpu>,
					   public SyncedAttribute<T, defaultDev, ArraySynchronizeOps,
											  DeviceArrayHandle> {
public:
	static constexpr Device DEFAULT_DEVICE = defaultDev;
	using Type = T;
	template < Device dev >
	using ArrayOps = DeviceArrayOps<dev, Type, DeviceArrayHandle>;
	using SyncAttr = typename SyncedAttribute<T, defaultDev, ArraySynchronizeOps,
											  DeviceArrayHandle>;
	using DeviceHandleTypes = typename SyncAttr::DeviceHandleTypes;
	using DeviceIndices = std::make_index_sequence<DeviceHandleTypes::size>;
	using ArrayAttrVals = ArrayAttributeValues<T, DeviceHandleTypes, storesCpu>;

	// Constructor for storing CPU data ourselves
	ArrayAttribute(std::string name) :
		IBaseAttribute(),
		ArrayAttrVals(),
		SyncAttr(typename ArrayAttrVals::template DeviceValueHelper<DeviceIndices>::
				 get_handles(ArrayAttrVals::values)),
		m_name(std::move(name))
	{}

	template < class VT >
	// Constructor for storing CPU data externally
	ArrayAttribute(std::string name, VT& ref) :
		IBaseAttribute(),
		ArrayAttrVals(ref),
		SyncAttr(typename ArrayAttrVals::template DeviceValueHelper<DeviceIndices>::
				 get_handles(ArrayAttrVals::values)),
		m_name(std::move(name)) {}

	virtual void reserve(std::size_t n) override {
		auto handle = this->get_handle<DEFAULT_DEVICE>();
		if (n != ArrayOps<DEFAULT_DEVICE>::get_capacity(handle)) {
			this->mark_changed<DEFAULT_DEVICE>();
			ArrayOps<DEFAULT_DEVICE>::reserve(handle, n);
		}
	}

	virtual void resize(std::size_t n) override {
		if (n != this->n_elements()) {
			this->mark_changed<DEFAULT_DEVICE>();
			ArrayOps<DEFAULT_DEVICE>::resize(this->get_handle<DEFAULT_DEVICE>(), n);
		}
	}

	virtual void clear() override {
		if (this->n_elements() != 0u) {
			this->mark_changed<DEFAULT_DEVICE>();
			ArrayOps<DEFAULT_DEVICE>::clear(this->get_handle<DEFAULT_DEVICE>());
		}
	}

	virtual std::size_t n_elements() const override {
		return ArrayOps<DEFAULT_DEVICE>::get_size(this->get_handle<DEFAULT_DEVICE>());
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