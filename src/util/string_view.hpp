#pragma once

#include <algorithm>
#include <climits>
#include <string>

namespace mufflon {

// I know it sucks, but since CUDA only supports C++14 and we got some ODR violations we
// need this
// Most of the implementation is trivial; the rest is taken from MSVCs cppstdlib implementation
template < class CharT, class Traits = std::char_traits<CharT> >
class BasicStringView {
public:
	using traits_type = Traits;
	using value_type = CharT;
	using pointer = CharT*;
	using const_pointer = const CharT*;
	using reference = CharT&;
	using const_reference = const CharT&;
	using const_iterator = const_pointer;
	using iterator = const_iterator;
	using const_reverse_iterator = std::reverse_iterator<const_iterator>;
	using reverse_iterator = const_reverse_iterator;
	using size_type = std::size_t;
	using difference_type = std::ptrdiff_t;

	static constexpr size_type npos = size_type(-1);

	// Constructors
	constexpr BasicStringView() noexcept :
		m_data{ nullptr },
		m_size{ 0u } {}
	constexpr BasicStringView(const BasicStringView& other) noexcept = default;
	constexpr BasicStringView(const_pointer s, size_type count) noexcept :
		m_data{ s },
		m_size{ count } {}
	constexpr BasicStringView(const_pointer s) noexcept :
		m_data{ s },
		m_size{ traits_type::length(s) } {}
	template < class Allocator >
	constexpr BasicStringView(const std::basic_string<CharT, Traits, Allocator>& s) :
		m_data{ s.data() },
		m_size{ s.size() } {}
	~BasicStringView() = default;

	// Assignment
	constexpr BasicStringView& operator=(const BasicStringView&) = default;

	// Conversion
	explicit operator std::string() const { return std::string(data(), size()); }

	// Iterators
	constexpr iterator begin() const noexcept { return m_data; }
	constexpr const_iterator cbegin() const noexcept { return m_data; }
	constexpr iterator end() const noexcept { return m_data + m_size; }
	constexpr const_iterator cend() const noexcept { return m_data + m_size; }
	constexpr reverse_iterator rbegin() const noexcept { return reverse_iterator{ end() }; }
	constexpr const_reverse_iterator crbegin() const noexcept { return const_reverse_iterator{ end() }; }
	constexpr reverse_iterator rend() const noexcept { return reverse_iterator{ begin() }; }
	constexpr const_reverse_iterator crend() const noexcept { return const_reverse_iterator{ begin() }; }

	// Accessors
	constexpr const_reference operator[](size_type pos) const { return *(m_data + pos); }
	constexpr const_reference at(size_type pos) const {
		if(pos >= m_size)
			throw std::out_of_range("string_view index is out of range");
		return m_data[pos];
	}
	constexpr const_reference front() const { return *m_data; }
	constexpr const_reference back() const { return *(m_data + m_size - size_type(1u)); }
	constexpr const_pointer data() const noexcept { return m_data; }

	// Capacity
	constexpr size_type size() const noexcept { return m_size; }
	constexpr size_type length() const noexcept { return m_size; }
	constexpr size_type max_size() const noexcept {
		return std::min(static_cast<size_type>(std::numeric_limits<difference_type>::max()),
						static_cast<size_type>(-1) / sizeof(value_type));
	}
	constexpr bool empty() const noexcept { return m_size == 0u; }

	// Modifiers
	constexpr void remove_prefix(size_type n) { m_data += n; }
	constexpr void remove_suffix(size_type n) { m_size -= n; }
	constexpr void swap(BasicStringView& v) noexcept { std::swap(m_data, v.m_data); std::swap(m_size, v.m_size); }

	// Operations
	constexpr size_type copy(pointer dest, size_type count, size_type pos = 0) const {
		traits_type::copy(dest, data(), std::min(count, size() - pos));
	}
	constexpr BasicStringView substr(size_type pos = 0, size_type count = npos) const {
		return BasicStringView{ m_data + pos, std::min(count == npos ? m_size : count, size() - pos) };
	}
	constexpr int compare(BasicStringView v) const noexcept {
		int c = traits_type::compare(data(), v.data(), std::min(size(), v.size()));
		if(c==0) {
			if(size() < v.size()) return -1;
			if(size() == v.size()) return 0;
			return 1;
		}
		return c;
	}
	constexpr int compare(size_type pos1, size_type count1, BasicStringView v) const { return substr(pos1, count1).compare(v); }
	constexpr int compare(size_type pos1, size_type count1, BasicStringView v,
						  size_type pos2, size_type count2) const {
		return substr(pos1, count1).compare(v.substr(pos2, count2));
	}
	constexpr int compare(const_pointer s) const { return compare(BasicStringView{ s }); }
	constexpr int compare(size_type pos1, size_type count1, const_pointer s) const { return substr(pos1, count1).compare(BasicStringView{ s }); }
	constexpr int compare(size_type pos1, size_type count1, const_pointer s,
						  size_type count2) const {
		return substr(pos1, count1).compare(BasicStringView{ s, count2});
	}
	constexpr bool starts_with(BasicStringView x) const noexcept { return size() >= x.size() && compare(0, x.size(), x) == 0; }
	constexpr bool starts_with(value_type x) const noexcept { return starts_with(BasicStringView{ std::addressof(x), 1 }); }
	constexpr bool starts_with(const_pointer x) const noexcept { return starts_with(BasicStringView{ x }); }
	constexpr bool ends_with(BasicStringView x) const noexcept { return size() >= x.size() && compare(size() - x.size(), npos, x) == 0; }
	constexpr bool ends_with(value_type x) const noexcept { return ends_with(BasicStringView{ std::addressof(x), 1 }); }
	constexpr bool ends_with(const_pointer x) const noexcept { return ends_with(BasicStringView{ x }); }
	constexpr size_type find(BasicStringView v, size_type pos = 0) const noexcept {
		// Taken from MSVC's implementation
		// search [_Haystack, _Haystack + _Hay_size) for [_Needle, _Needle + _Needle_size), at/after _Start_at
		if(v.size() > size()|| pos > size() - v.size()) {	// xpos cannot exist, report failure
				// N4659 24.3.2.7.2 [string.find]/1 says:
				// 1. _Start_at <= xpos
				// 2. xpos + _Needle_size <= _Hay_size;
				// therefore:
				// 3. _Needle_size <= _Hay_size (by 2) (checked above)
				// 4. _Start_at + _Needle_size <= _Hay_size (substitute 1 into 2)
				// 5. _Start_at <= _Hay_size - _Needle_size (4, move _Needle_size to other side) (also checked above)
			return (static_cast<size_t>(-1));
		}

		if(size() == 0) {	// empty string always matches if xpos is possible
			return pos;
		}

		const auto possibleMatchesEnd = data() + (size() - v.size()) + 1;
		for(auto matchTry = data() + pos; ; ++matchTry) {
			matchTry = traits_type::find(matchTry, static_cast<size_type>(possibleMatchesEnd - matchTry), *v.data());
			if(!matchTry) {	// didn't find first character; report failure
				return (static_cast<size_type>(-1));
			}

			if(traits_type::compare(matchTry, v.data(), v.size()) == 0) {	// found match
				return (static_cast<size_t>(matchTry - data()));
			}
		}
	}
	constexpr size_type find(value_type ch, size_type pos = 0) const noexcept { return find(BasicStringView{ std::addressof(ch), 1 }, pos); }
	constexpr size_type find(const_pointer s, size_type pos, size_type count) const { return find(BasicStringView{ s, count }, pos); }
	constexpr size_type find(const_pointer s, size_type pos = 0) const { return find(BasicStringView{ s }, pos); }
	/*constexpr size_type rfind(BasicStringView v, size_type pos = npos) const noexcept {
		// TODO
	}
	constexpr size_type rfind(value_type ch, size_type pos = npos) const noexcept { return rfind(BasicStringView{ std::addressof(ch), 1 }, pos); }
	constexpr size_type rfind(const_pointer s, size_type pos, size_type count) const { return rfind(BasicStringView{ s, count }, pos); }
	constexpr size_type rfind(const_pointer s, size_type pos = npos) const { return rfind(BasicStringView{ s }, pos); }
	constexpr size_type find_first_of(BasicStringView v, size_type pos = 0) const noexcept {
		// TODO
	}
	constexpr size_type find_first_of(value_type ch, size_type pos = 0) const noexcept { return find_first_of(BasicStringView{ std::addressof(ch), 1 }, pos); }
	constexpr size_type find_first_of(const_pointer s, size_type pos, size_type count) const { return find_first_of(BasicStringView{ s, count }, pos); }
	constexpr size_type find_first_of(const_pointer s, size_type pos = 0) const { return find_first_of(BasicStringView{ s }, pos); }
	constexpr size_type find_last_of(BasicStringView v, size_type pos = npos) const noexcept {
		// TODO
	}
	constexpr size_type find_last_of(value_type ch, size_type pos = npos) const noexcept { return find_last_of(BasicStringView{ std::addressof(ch), 1 }, pos); }
	constexpr size_type find_last_of(const_pointer s, size_type pos, size_type count) const { return find_last_of(BasicStringView{ s, count }, pos); }
	constexpr size_type find_last_of(const_pointer s, size_type pos = npos) const { return find_last_of(BasicStringView{ s }, pos); }
	constexpr size_type find_first_not_of(BasicStringView v, size_type pos = 0) const noexcept {
		// TODO
	}
	constexpr size_type find_first_not_of(value_type ch, size_type pos = 0) const noexcept { return find_first_not_of(BasicStringView{ std::addressof(ch), 1 }, pos); }
	constexpr size_type find_first_not_of(const_pointer s, size_type pos, size_type count) const { return find_first_not_of(BasicStringView{ s, count }, pos); }
	constexpr size_type find_first_not_of(const_pointer s, size_type pos = 0) const { return find_first_not_of(BasicStringView{ s }, pos); }
	constexpr size_type find_last_not_of(BasicStringView v, size_type pos = npos) const noexcept {
		// TODO
	}
	constexpr size_type find_last_not_of(value_type ch, size_type pos = npos) const noexcept { return find_last_not_of(BasicStringView{ std::addressof(ch), 1 }, pos); }
	constexpr size_type find_last_not_of(const_pointer s, size_type pos, size_type count) const { return find_last_not_of(BasicStringView{ s, count }, pos); }
	constexpr size_type find_last_not_of(const_pointer s, size_type pos = npos) const { return find_last_not_of(BasicStringView{ s }, pos); }*/

private:
	const_pointer m_data;
	size_type m_size;
};

// Operators
template < class CharT, class Traits >
constexpr bool operator==(BasicStringView<CharT, Traits> lhs, BasicStringView<CharT, Traits> rhs) {
	return lhs.compare(rhs) == 0;
}
template < class CharT, class Traits >
constexpr bool operator!=(BasicStringView<CharT, Traits> lhs, BasicStringView<CharT, Traits> rhs) {
	return lhs.compare(rhs) != 0;
}
template < class CharT, class Traits >
constexpr bool operator<(BasicStringView<CharT, Traits> lhs, BasicStringView<CharT, Traits> rhs) {
	return lhs.compare(rhs) < 0;
}
template < class CharT, class Traits >
constexpr bool operator<=(BasicStringView<CharT, Traits> lhs, BasicStringView<CharT, Traits> rhs) {
	return lhs.compare(rhs) <= 0;
}
template < class CharT, class Traits >
constexpr bool operator>(BasicStringView<CharT, Traits> lhs, BasicStringView<CharT, Traits> rhs) {
	return lhs.compare(rhs) > 0;
}
template < class CharT, class Traits >
constexpr bool operator>=(BasicStringView<CharT, Traits> lhs, BasicStringView<CharT, Traits> rhs) {
	return lhs.compare(rhs) >= 0;
}
// Operators with string
template < class CharT, class Traits, class Allocator >
constexpr bool operator<(BasicStringView<CharT, Traits> lhs, const std::basic_string<CharT, Traits, Allocator>& rhs) {
	return lhs.compare(rhs) < 0;
}
template < class CharT, class Traits, class Allocator >
constexpr bool operator<(const std::basic_string<CharT, Traits, Allocator>& lhs, BasicStringView<CharT, Traits> rhs) {
	return rhs.compare(lhs) > 0;
}

// Input/output
template < class CharT, class Traits >
inline std::basic_ostream<CharT, Traits>& operator<<(std::basic_ostream<CharT, Traits>& os, BasicStringView<CharT, Traits> v) {
	using size_type = typename BasicStringView<CharT, Traits>::size_type;
	if(v.size() < static_cast<size_type>(os.width()) && !(os.flags() & std::ios_base::adjustfield)) {
		for(typename BasicStringView<CharT, Traits>::size_type i = 0; i < os.width() - v.size(); ++i)
			os.rdbuf()->sputc(os.fill());
	}

	os.rdbuf()->sputn(v.data(), std::max(static_cast<size_type>(os.width()), v.size()));

	if(v.size() < static_cast<size_type>(os.width()) && (os.flags() & std::ios_base::adjustfield)) {
		for(typename BasicStringView<CharT, Traits>::size_type i = 0; i < static_cast<size_type>(os.width()) - v.size(); ++i)
			os.rdbuf()->sputc(os.fill());
	}
	os.width(0);
	return os;
}

// Conversion
template < class CharT, class Traits, class Allocator = std::allocator<CharT> >
inline std::basic_string<CharT, Traits, Allocator> to_string(BasicStringView<CharT, Traits> v) {
	return std::basic_string<CharT, Traits, Allocator>{ v.data(), v.size() };
}

// Typedefs
using StringView = BasicStringView<char>;
using WStringView = BasicStringView<wchar_t>;
using U16StringView = BasicStringView<char16_t>;
using U32StringView = BasicStringView<char32_t>;

} // namespace mufflon

// Hashing
namespace std {

template <class CharT, class Traits >
struct hash<mufflon::BasicStringView<CharT, Traits>> {
	size_t operator()(const mufflon::BasicStringView<CharT, Traits> keyVal) const noexcept {
		return hash<basic_string<CharT, Traits>>()(std::basic_string<CharT, Traits>{ keyVal.data(), keyVal.size() });
	}

};

} // namespace std