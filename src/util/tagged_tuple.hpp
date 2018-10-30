#pragma once

#include <tuple>

namespace mufflon { namespace util {

// Utility functions to ensure that all types are distinct
namespace tagged_tuple_detail {

/// Iterates all values of the variadic template and returns whether they are all true
template < bool B, bool... Bs >
struct and_pack {
	static constexpr bool value = B && and_pack<Bs...>::value;
};

template < bool B >
struct and_pack<B> {
	static constexpr bool value = B;
};

/// Returns whether all types of the variadic template are distinct
template < class H, class... Tails >
struct is_all_distinct {
	static constexpr bool value = and_pack<!std::is_same<H, Tails>::value...>::value
		&& is_all_distinct<Tails...>::value;
};

template < class H >
struct is_all_distinct<H> {
	static constexpr bool value = true;
};

/// Returns whether all types of the variadic template are equal
template < class H, class... Tails >
struct is_all_same {
	static constexpr bool value = and_pack<std::is_same<H, Tails>::value...>::value
		&& is_all_same<Tails...>::value;
};

template < class H >
struct is_all_same<H> {
	static constexpr bool value = true;
};

} // namespace typelist_detail

/**
 * Class containing a tagged tuple implementation.
 * That means that, unlike and std::tuple, the index for access is a C++
 * type and not a number. This means that all types must be distinct,
 * which will be ensured at compile time.
 */
template < class... Args >
class TaggedTuple {
public:
	static_assert(tagged_tuple_detail::is_all_distinct<Args...>::value,
				  "The types of a tagged tuple must be distinct!");

	using TupleType = std::tuple<Args...>;

	template < std::size_t N >
	using Type = std::tuple_element_t<N, std::tuple<Args...>>;

	static constexpr std::size_t size = sizeof...(Args);

	TaggedTuple(Args&& ...args) :
		m_tuple(std::forward<Args>(args)...)
	{}
	TaggedTuple() = default;
	TaggedTuple(const TaggedTuple&) = default;
	TaggedTuple(TaggedTuple&&) = default;
	TaggedTuple& operator=(const TaggedTuple&) = default;
	TaggedTuple& operator=(TaggedTuple&&) = default;
	~TaggedTuple() = default;

	// Returns the numerical index of the type in the tagged tuple.
	template < class T >
	static constexpr std::size_t get_index() noexcept {
		return Index<T, Args...>::value;
	}

	// Access to the tuple value by index.
	template < std::size_t I >
	constexpr Type<I>& get() noexcept {
		return std::get<I>(m_tuple);
	}

	// Access to the tuple value by index.
	template < std::size_t I >
	constexpr const Type<I>& get() const noexcept {
		return std::get<I>(m_tuple);
	}

	// Access to the tuple value by type.
	template < class T >
	constexpr T& get() noexcept {
		return std::get<get_index<T>()>(m_tuple);
	}

	// Access to the tuple value by type.
	template < class T >
	constexpr const T& get() const noexcept {
		return std::get<get_index<T>()>(m_tuple);
	}

	// Checks whether
	template < class T >
	static constexpr bool has() noexcept {
		return get_index<T>() < size;
	}

	template < class Op >
	void for_each(Op&& op) {

	}

private:
	// Helper class for finding the index of a type for tuple lookup
	template < class... T >
	struct Index;

	// Terminate search
	template < class H, class... Tails >
	struct Index<H, H, Tails...> : public std::integral_constant<std::size_t, 0> {};

	// Recurse until we find the type and count the recursions
	template < class T, class H, class... Tails >
	struct Index<T, H, Tails...> : public std::integral_constant<std::size_t, 1 + Index<T, Tails...>::value> {};

	template < std::size_t I, class Op >
	void for_each_impl(Op& op) {
		if constexpr(I < size) {
			if(!op(I, this->get<I>()))
				for_each_impl<I + 1u>(op);
		}
	}

	TupleType m_tuple;
};

// Function overloads to use this type instead of a tuple
template < class T, class... Types >
constexpr T& get(TaggedTuple<Types...>& tuple) noexcept {
	return tuple.get<T>();
}
template < class T, class... Types >
constexpr const T& get(const TaggedTuple<Types...>& tuple) noexcept {
	return tuple.get<T>();
}

template < class... Types >
class std::tuple_size<TaggedTuple<Types...>> :
	public std::integral_constant<std::size_t, TaggedTuple<Types...>::size> {};

}} // namespace mufflon::util