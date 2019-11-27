#pragma once

#include "core/scene/geometry/polygon_mesh.hpp"
#include <ei/3dtypes.hpp>
#include <ei/vector.hpp>
#include <OpenMesh/Core/IO/SR_binary.hh>
#include <cstddef>
#include <cstdint>

// Specialize OpenMesh's size structure
namespace OpenMesh::IO {

template <>
struct binary<ei::Vec<unsigned char, 2u>> {
	using value_type = ei::Vec<unsigned char, 2u>;
	static constexpr bool is_streamable = false;
	static std::size_t size_of() { return sizeof(value_type); }
	static std::size_t size_of(const value_type&) { return sizeof(value_type); }
	static std::size_t store(std::ostream&, const value_type&, bool) { return 0u; }
	static std::size_t restore(std::istream&, value_type&, bool) { return 0u; }
};
template <>
struct binary<ei::Vec<unsigned char, 3u>> {
	using value_type = ei::Vec<unsigned char, 3u>;
	static constexpr bool is_streamable = false;
	static std::size_t size_of() { return sizeof(value_type); }
	static std::size_t size_of(const value_type&) { return sizeof(value_type); }
	static std::size_t store(std::ostream&, const value_type&, bool) { return 0u; }
	static std::size_t restore(std::istream&, value_type&, bool) { return 0u; }
};
template <>
struct binary<ei::Vec<unsigned char, 4u>> {
	using value_type = ei::Vec<unsigned char, 4u>;
	static constexpr bool is_streamable = false;
	static std::size_t size_of() { return sizeof(value_type); }
	static std::size_t size_of(const value_type&) { return sizeof(value_type); }
	static std::size_t store(std::ostream&, const value_type&, bool) { return 0u; }
	static std::size_t restore(std::istream&, value_type&, bool) { return 0u; }
};
template <>
struct binary<ei::Vec<std::int32_t, 2u>> {
	using value_type = ei::Vec<std::int32_t, 2u>;
	static constexpr bool is_streamable = false;
	static std::size_t size_of() { return sizeof(value_type); }
	static std::size_t size_of(const value_type&) { return sizeof(value_type); }
	static std::size_t store(std::ostream&, const value_type&, bool) { return 0u; }
	static std::size_t restore(std::istream&, value_type&, bool) { return 0u; }
};
template <>
struct binary<ei::Vec<std::int32_t, 3u>> {
	using value_type = ei::Vec<std::int32_t, 3u>;
	static constexpr bool is_streamable = false;
	static std::size_t size_of() { return sizeof(value_type); }
	static std::size_t size_of(const value_type&) { return sizeof(value_type); }
	static std::size_t store(std::ostream&, const value_type&, bool) { return 0u; }
	static std::size_t restore(std::istream&, value_type&, bool) { return 0u; }
};
template <>
struct binary<ei::Vec<std::int32_t, 4u>> {
	using value_type = ei::Vec<std::int32_t, 4u>;
	static constexpr bool is_streamable = false;
	static std::size_t size_of() { return sizeof(value_type); }
	static std::size_t size_of(const value_type&) { return sizeof(value_type); }
	static std::size_t store(std::ostream&, const value_type&, bool) { return 0u; }
	static std::size_t restore(std::istream&, value_type&, bool) { return 0u; }
};
template <>
struct binary<ei::Sphere> {
	using value_type = ei::Sphere;
	static constexpr bool is_streamable = false;
	static std::size_t size_of() { return sizeof(value_type); }
	static std::size_t size_of(const value_type&) { return sizeof(value_type); }
	static std::size_t store(std::ostream&, const value_type&, bool) { return 0u; }
	static std::size_t restore(std::istream&, value_type&, bool) { return 0u; }
};

} // namespace OpenMesh::IO