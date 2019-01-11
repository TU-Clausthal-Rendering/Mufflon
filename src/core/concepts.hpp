#pragma once

#include "core/memory/residency.hpp"
#include <type_traits>

namespace mufflon {

#ifndef __CUDACC__
/* The manager concept is a host (cpu) class which provides access
 * and synchronization to some resource for rendering.
 * A manager should not implement any functions on the data. Rather,
 * the functional layer must work on the return values of acquire_const()
 * and acquire() which are also called Descriptors.
 *
 * Usage: put 'template DeviceManagerConcept<YourType>;'
 * after the closing }; of the class YourType.
 *
 * Enforced members:
 *		template<Device dev> acquire_const();
 *			- must call synchronize() implicitly (not enforceable)
 *		template<Device dev> unload();
 *		template<Device dev> synchronize();
 * Optional, but not enforced:
 *		template<Device dev> acquire();
 *
 * Inspired by: https://stackoverflow.com/a/37117023/1913512
 */
template<class T>
struct DeviceManagerConcept {
private:
	template<class> struct result_type_of;
	template<class R, class... Args> struct result_type_of<R (Args...)> {
		using type = R;
	};
	template<class R, class... Args> struct result_type_of<R (T::*)(Args...)> {
		using type = R;
	};
	template<class R, class... Args> struct result_type_of<R (T::*)(Args...) noexcept> {
		using type = R;
	};
	template<class T>
	using result_type_of_t = typename result_type_of<T>::type;

	// Runs fine with cpp, but not with nvcc around...
	template<class T, typename = decltype(&T::acquire_const<Device::CPU>)> static constexpr bool has_acquire_const() { return true; }
	template<class T> static constexpr bool has_acquire_const(...) { return false; }
	template<class T, typename = std::enable_if_t<std::is_trivially_copyable<
		std::remove_cv_t<std::decay_t< result_type_of_t<decltype(&T::acquire_const<Device::CPU>)> >>
	>::value>> static constexpr bool descriptor_is_copyable() { return true; }
	template<class T> static constexpr bool descriptor_is_copyable(...) { return false; }
	template<class T, typename = decltype(&T::unload<Device::CPU>)> static constexpr bool has_unload() { return true; }
	template<class T> static constexpr bool has_unload(...) { return false; }
	template<class T, typename = decltype(&T::synchronize<Device::CPU>)> static constexpr bool has_sync() { return true; }
	template<class T> static constexpr bool has_sync(...) { return false; }

public:
	//static_assert(std::is_member_function_pointer<decltype(&T::acquire_const)>::value
	//	&& std::is_trivially_copyable<decltype(std::declval<T>().acquire_const())>::value,
	//	"Manager classes must provide readable access to a copyabel multi-device descriptor. The function shall not have any parameter.");
	static_assert(has_acquire_const<T>(), "Must have a member acquire_const<Device>().");
	static_assert(descriptor_is_copyable<T>(), "Descriptors returned by aquire_const<Device>() must be trivially copyable.");
	static_assert(has_unload<T>(), "Must have a member unload<Device>().");
	static_assert(has_sync<T>(), "Must have a member synchronize<Device>().");
};
#else
// There are compilation issues with the meta-programming above (although they are pure
// C++14 and should work.
template<class T>
struct DeviceManagerConcept {};
#endif



/* Examples: 
struct A {
	int acquire_const() { return 0; }
	template<mufflon::Device dev> void unload() {
	}
};
template DeviceManagerConcept<A>;

struct B {
	//int acquire_const(float) { return 0; }
	std::mutex acquire_const() { return std::mutex{}; }
};
template DeviceManagerConcept<B>;

template<int I>
struct C {
};
template DeviceManagerConcept<C<0>>;
*/

} // namespace mufflon
