#pragma once

#include "core/memory/residency.hpp"
#include <type_traits>

namespace mufflon {

namespace concept_details {
	template<class> struct result_type_of;
	template<class R, class... Args> struct result_type_of<R (Args...)> {
		using type = R;
	};
	template<class T, class R, class... Args> struct result_type_of<R (T::*)(Args...)> {
		using type = R;
	};
#if __cplusplus >= 201703L
	template<class T, class R, class... Args> struct result_type_of<R (T::*)(Args...) noexcept> {
		using type = R;
	};
#endif // __CUDACC__
	template<class T>
	using result_type_of_t = typename result_type_of<T>::type;
}

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
	template<class T, typename G = decltype(&T::template acquire_const<Device::CPU>)> static constexpr bool has_acquire_const(int) { return true; }
	template<class T> static constexpr bool has_acquire_const(...) { return false; }
	template<class T, typename = std::enable_if_t<std::is_trivially_copyable<
		std::remove_cv_t<std::decay_t< concept_details::result_type_of_t<decltype(&T::template acquire_const<Device::CPU>)> >>
	>::value>> static constexpr bool descriptor_is_copyable(int) { return true; }
	template<class T> static constexpr bool descriptor_is_copyable(...) { return false; }
	template<class T, typename = decltype(&T::template unload<Device::CPU>)> static constexpr bool has_unload(int) { return true; }
	template<class T> static constexpr bool has_unload(...) { return false; }
	template<class T, typename = decltype(&T::template synchronize<Device::CPU>)> static constexpr bool has_sync(int) { return true; }
	template<class T> static constexpr bool has_sync(...) { return false; }

public:
	//static_assert(std::is_member_function_pointer<decltype(&T::acquire_const)>::value
	//	&& std::is_trivially_copyable<decltype(std::declval<T>().acquire_const())>::value,
	//	"Manager classes must provide readable access to a copyable multi-device descriptor. The function shall not have any parameter.");
	static_assert(has_acquire_const<T>(0), "Must have a member acquire_const<Device>().");
	static_assert(descriptor_is_copyable<T>(0), "Descriptors returned by aquire_const<Device>() must be trivially copyable.");
	static_assert(has_unload<T>(0), "Must have a member unload<Device>().");
	static_assert(has_sync<T>(0), "Must have a member synchronize<Device>().");
};



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
