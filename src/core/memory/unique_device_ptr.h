#pragma once
#include "residency.hpp"
#include <memory>
#include "allocator.hpp"
#include "core/opengl/gl_buffer.hpp"

namespace mufflon {
template < Device dev, typename T >
class unique_device_ptr : public std::unique_ptr<T, Deleter<dev>> {
	// inherit constructors
	using std::unique_ptr<T, Deleter<dev>>::unique_ptr;
};

template <typename T>
class unique_device_ptr<Device::OPENGL, T> {
public:
	using TType = std::remove_pointer_t<std::decay_t<T>>;

	unique_device_ptr() = default;
	unique_device_ptr(gl::BufferHandle<TType> handle, Deleter<Device::OPENGL> deleter) :
		m_handle(handle), 
		m_deleter(deleter)
	{}
	unique_device_ptr(const unique_device_ptr<Device::OPENGL, T>&) = delete;
	unique_device_ptr<Device::OPENGL, T>& operator=(const unique_device_ptr<Device::OPENGL, T>&) = delete;
	unique_device_ptr(unique_device_ptr<Device::OPENGL, T>&& o) noexcept :
		m_handle(o.m_handle),
		m_deleter(o.m_deleter)
	{
		o.m_handle = {0, 0};
	}
	unique_device_ptr<Device::OPENGL, T>& operator=(unique_device_ptr<Device::OPENGL, T>&& o) noexcept {
		std::swap(m_handle, o.m_handle);
		std::swap(m_deleter, o.m_deleter);
		return *this;
	}
	unique_device_ptr(std::nullptr_t) :
		unique_device_ptr()
	{}
	~unique_device_ptr() {
		m_deleter.operator()(m_handle);
	}
	gl::BufferHandle<TType> get() const {
		return m_handle;
	}
	gl::BufferHandle<TType> get() {
		return m_handle;
	}
	bool empty() const noexcept {
		return m_handle == 0;
	}
	bool operator==(std::nullptr_t) const noexcept {
		return empty();
	}
	bool operator!=(std::nullptr_t) const noexcept {
		return !empty();
	}
private:
	gl::BufferHandle<TType> m_handle = {0, 0};
	Deleter<Device::OPENGL> m_deleter;
};

template < Device dev, typename T, typename... Args > 
inline unique_device_ptr<dev, T> make_udevptr(Args... args) {
	return unique_device_ptr<dev, T>(
		Allocator<dev>::template alloc<T>(std::forward<Args>(args)...),
		Deleter<dev>(1)
	);
}

template < Device dev, typename T, bool Init = true, typename... Args > 
inline unique_device_ptr<dev, T[]> make_udevptr_array(std::size_t n, Args... args) {
	return unique_device_ptr<dev, T[]>(
		Allocator<dev>::template alloc_array<T, Init>(n, std::forward<Args>(args)...),
		Deleter<dev>(n)
	);
}
}
