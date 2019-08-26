#pragma once

#include <type_traits>

namespace mufflon {

template < bool B, bool... Bs >
struct AndPack { static constexpr bool value = B && AndPack<Bs...>::value; };
template < bool B >
struct AndPack<B> { static constexpr bool value = B; };

template < template < class, class > class Pred, class H, class... Tails >
struct IsAllDistinct {
	static constexpr bool value = AndPack<!Pred<H, Tails>::value...>::value
		&& IsAllDistinct<Pred, Tails...>::value;
};
template < template < class, class > class Pred, class H >
struct IsAllDistinct<Pred, H> : std::true_type {};

template < template < class, class > class Pred, class H, class... Tails >
struct IsAllSame {
	static constexpr bool value = AndPack<Pred<H, Tails>::value...>::value
		&& IsAllSame<Pred, Tails...>::value;
};

template < template < class, class > class Pred, class H >
struct IsAllSame<Pred, H> : std::true_type {};


template < class, template < class... > class >
struct IsInstanceOf : std::false_type {};
template < class... Ts, template < class... > class U >
struct IsInstanceOf<U<Ts...>, U> : std::true_type {};

template < class A, class B >
struct HasSameName {
	static constexpr bool strings_equal(const char* a, const char* b) {
		return *a == *b && (*a == '\0' || strings_equal(a + 1, b + 1));
	}
	static constexpr bool value = strings_equal(A::NAME, B::NAME);
};

template < class... Ts >
constexpr bool have_distinct_names() noexcept {
	return IsAllDistinct<HasSameName, Ts...>::value;
}
template < class... Ts >
constexpr bool have_distinct_types() noexcept {
	return IsAllDistinct<std::is_same, Ts...>::value;
}

} // namespace mufflon