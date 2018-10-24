#pragma once

#include "residency.hpp"
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
 */
class IBaseAttribute {
public:
	IBaseAttribute() = default;
	IBaseAttribute(const IBaseAttribute&) = default;
	IBaseAttribute(IBaseAttribute&&) = default;
	IBaseAttribute& operator=(const IBaseAttribute&) = default;
	IBaseAttribute& operator=(IBaseAttribute&&) = default;
	virtual ~IBaseAttribute() = default;
	virtual std::string_view get_name() const noexcept = 0;
	virtual std::size_t get_capacity() const = 0;
	virtual std::size_t get_size() const = 0;
	virtual std::size_t get_elem_size() const = 0;
	virtual void reserve(std::size_t count) = 0;
	virtual void resize(std::size_t count) = 0;
	virtual void clear() = 0;
	virtual std::size_t read(std::size_t elems, std::istream&) = 0;
	virtual std::size_t write(std::ostream&) const = 0;
};

/**
 * Array attribute class.
 * Contains handles for all devices.
 * Can hand out read or write access.
 * Synchronizes between devices.
 */
template < class T, Residency defaultDev = Residency::CPU >
class ArrayAttribute : public IBaseAttribute {
public:
	static constexpr Residency DEFAULT_DEVICE = defaultDev;
	using Type = T;
	// TODO: readd OpenGL
	using HandleTypes = DeviceArrayHandles<Type, Residency::CPU, Residency::CUDA>;
	using DefaultHandleType = DeviceArrayHandle<DEFAULT_DEVICE, Type>;
	template < Residency dev >
	using ArrayOps = DeviceArrayOps<dev, Type>;

	// Provides constant-only access to the attribute data
	template < Residency dev >
	class ConstAccessor {
	public:
		static constexpr Residency DEVICE = dev;
		using Type = T;
		using HandleType = typename DeviceArrayHandle<DEVICE, Type>::HandleType;

		ConstAccessor(const HandleType& handle) :
			m_handle(&handle) {}
		ConstAccessor(const ConstAccessor&) = default;
		ConstAccessor(ConstAccessor&&) = default;
		ConstAccessor& operator=(const ConstAccessor&) = default;
		ConstAccessor& operator=(ConstAccessor&&) = default;
		~ConstAccessor() = default;

		const HandleType& operator*() const {
			mAssert(m_handle != nullptr);
			return *m_handle;
		}

		const HandleType* operator->() const {
			return m_handle;
		}

	private:
		const HandleType* m_handle;
	};

	// Provides read-and-write access to the attribute data. Flags as dirty upon destruction
	template < Residency dev >
	class Accessor {
	public:
		static constexpr Residency DEVICE = dev;
		using Type = T;
		using HandleType = typename DeviceArrayHandle<DEVICE, Type>::HandleType;

		Accessor(HandleType& handle, util::DirtyFlags<Residency> &flags) :
			m_handle(&handle), m_flags(flags) {}
		Accessor(const Accessor&) = delete;
		Accessor(Accessor&&) = default;
		Accessor& operator=(const Accessor&) = delete;
		Accessor& operator=(Accessor&&) = default;
		~Accessor() {
			// TODO: would we like lazy copying?
			m_flags.mark_changed(DEVICE);
		}

		HandleType& operator*() {
			mAssert(m_handle != nullptr);
			return *m_handle;
		}

		HandleType* operator->() {
			return m_handle;
		}

	private:
		HandleType* m_handle;
		util::DirtyFlags<Residency> &m_flags;
	};

	ArrayAttribute(std::string name) :
		m_name(std::move(name))
	{}
	ArrayAttribute(const ArrayAttribute&) = default;
	ArrayAttribute(ArrayAttribute&&) = default;
	ArrayAttribute& operator=(const ArrayAttribute&) = default;
	ArrayAttribute& operator=(ArrayAttribute&&) = default;
	~ArrayAttribute() = default;

	virtual std::string_view get_name() const noexcept override {
		return m_name;
	}

	virtual std::size_t get_size() const override {
		return ArrayOps<DEFAULT_DEVICE>::get_size(m_handles.get<DefaultHandleType>());
	}

	virtual std::size_t get_capacity() const override {
		return ArrayOps<DEFAULT_DEVICE>::get_capacity(m_handles.get<DefaultHandleType>());
	}

	virtual std::size_t get_elem_size() const override {
		return sizeof(Type);
	}

	virtual void reserve(std::size_t n) override {
		if(n != this->get_capacity())
			m_dirty.mark_changed(DEFAULT_DEVICE);
		return ArrayOps<DEFAULT_DEVICE>::reserve(m_handles.get<DefaultHandleType>(), n);
	}

	virtual void resize(std::size_t n) override {
		if(n != this->get_capacity())
			m_dirty.mark_changed(DEFAULT_DEVICE);
		return ArrayOps<DEFAULT_DEVICE>::resize(m_handles.get<DefaultHandleType>(), n);
	}

	virtual void clear() override {
		ArrayOps<DEFAULT_DEVICE>::clear(m_handles.get<DefaultHandleType>());
	}

	virtual std::size_t read(std::size_t elems, std::istream& stream) override {
		const std::size_t currSize = this->get_size();
		this->resize(currSize + elems);
		Type* nativePtr = &m_handles.get<DefaultHandleType>().handle.data()[currSize];
		stream.read(reinterpret_cast<char*>(nativePtr), sizeof(Type) * elems);
		m_dirty.mark_changed(Residency::CPU);
		return static_cast<std::size_t>(stream.gcount()) / sizeof(Type);
	}

	virtual std::size_t write(std::ostream& stream) const override {
		// TODO: read/write to other device?
		const std::size_t elems = this->get_size();
		const Type* nativePtr = m_handles.get<DefaultHandleType>().handle.data();
		stream.write(reinterpret_cast<const char*>(nativePtr),
					 sizeof(Type) * elems);
		return elems;
	}

	// Aquire a read-only accessor
	template < Residency dev = DEFAULT_DEVICE >
	ConstAccessor<dev> aquireConst() {
		this->synchronize<dev>();
		return ConstAccessor<dev>(m_handles.get<DeviceArrayHandle<dev, Type>>().handle);
	}

	// Aquire a writing (and thus dirtying) accessor
	template < Residency dev = DEFAULT_DEVICE >
	Accessor<dev> aquire() {
		this->synchronize<dev>();
		return Accessor<dev>(m_handles.get<DeviceArrayHandle<dev, Type>>().handle, m_dirty);
	}

	// Explicitly synchronize the given device
	template < Residency dev = DEFAULT_DEVICE >
	void synchronize() {
		if(m_dirty.has_competing_changes())
			throw std::runtime_error("Failure: competing changes for this attribute");
		if(m_dirty.has_changes()) {
			Synchronizer<0u>::synchronize(m_handles, m_dirty, m_handles.get<DeviceArrayHandle<dev, Type>>());
		}
	}

private:
	// Helper struct for iterating through all devices and finding one which has changes
	template < std::size_t I >
	struct Synchronizer {
		template < Residency dev >
		static void synchronize(HandleTypes &handles, util::DirtyFlags<Residency>& flags,
								DeviceArrayHandle<dev, Type>& sync) {
			if constexpr(I < HandleTypes::size) {
				auto& changed = handles.get<I>();
				constexpr Residency CHANGED_DEVICE = changed.DEVICE;
				if(flags.has_changes(CHANGED_DEVICE)) {
					// Check if resize is necessary and copy over the data
					std::size_t changedSize = ArrayOps<CHANGED_DEVICE>::get_size(changed);
					if(changedSize != ArrayOps<dev>::get_size(sync))
						ArrayOps<dev>::resize(sync, changedSize);
					ArrayOps<dev>::copy(changed, sync);
				} else {
					// Keep looking for device changes
					Synchronizer<I + 1u>::synchronize(handles, flags, sync);
				}
			}
		}
	};

	std::string m_name;
	HandleTypes m_handles;
	util::DirtyFlags<Residency> m_dirty;
};


} // namespace mufflon::scene