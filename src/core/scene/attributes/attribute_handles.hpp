#pragma once

#include "util/string_pool.hpp"
#include "util/string_view.hpp"
#include <ei/3dtypes.hpp>
#include <ei/vector.hpp>
#include <OpenMesh/Core/Geometry/VectorT.hh>
#include <OpenMesh/Core/IO/SR_binary.hh>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace mufflon::scene {

// Allowed attribute types. Expand as needed.
enum class AttributeType : std::uint16_t {
	CHAR,
	UCHAR,
	SHORT,
	USHORT,
	INT,
	UINT,
	LONG,
	ULONG,
	FLOAT,
	DOUBLE,
	UCHAR2,
	UCHAR3,
	UCHAR4,
	INT2,
	INT3,
	INT4,
	FLOAT2,
	FLOAT3,
	FLOAT4,
	SPHERE
};

/* Attribute identifier. Uniquely identifies a (desired) attribute.
 * Cannot be used directly, but can be used to try and find an attribute handle.
 */
struct AttributeIdentifier {
	AttributeIdentifier(AttributeType type,
						StringView name) :
		type{ type },
		name{ util::UniqueStringPool::instance().insert(name) }
	{}
	AttributeIdentifier(const AttributeIdentifier& other) :
		type{ other.type },
		name{ util::UniqueStringPool::instance().insert(other.name) }
	{}
	AttributeIdentifier(AttributeIdentifier&&) = delete;
	AttributeIdentifier& operator=(const AttributeIdentifier& other) {
		if(&other == this)
			return *this;
		util::UniqueStringPool::instance().remove(name);
		type = other.type;
		name = util::UniqueStringPool::instance().insert(other.name);
		return *this;
	}
	AttributeIdentifier& operator=(AttributeIdentifier&&) = delete;
	~AttributeIdentifier() {
		util::UniqueStringPool::instance().remove(name);
	}

	bool operator==(const AttributeIdentifier& rhs) const noexcept {
		return (type == rhs.type) && (name == rhs.name);
	}

	AttributeType type;
	StringView name;
};

/* General attribute handle. Specifies its index and type.
 * Should be treated as an opaque handle.
 */
struct AttributeHandle {
	AttributeHandle(const AttributeIdentifier& ident, const std::uint32_t index) :
		identifier{ ident },
		index{ index }
	{}

	AttributeIdentifier identifier;
	std::uint32_t index;
};

// Returns the attribute type enum value for a given type
template < class T >
inline constexpr AttributeType get_attribute_type() { static_assert(sizeof(T) == 0, "Unsupported attribute type!"); }
template <>
inline constexpr AttributeType get_attribute_type<std::int16_t>() { return AttributeType::SHORT; }
template <>
inline constexpr AttributeType get_attribute_type<std::uint16_t>() { return AttributeType::USHORT; }
template <>
inline constexpr AttributeType get_attribute_type<std::int32_t>() { return AttributeType::INT; }
template <>
inline constexpr AttributeType get_attribute_type<std::uint32_t>() { return AttributeType::UINT; }
template <>
inline constexpr AttributeType get_attribute_type<float>() { return AttributeType::FLOAT; }
template <>
inline constexpr AttributeType get_attribute_type<ei::Vec2>() { return AttributeType::FLOAT2; }
template <>
inline constexpr AttributeType get_attribute_type<ei::Vec3>() { return AttributeType::FLOAT3; }
template <>
inline constexpr AttributeType get_attribute_type<ei::Vec4>() { return AttributeType::FLOAT4; }
// Special cases for points, normals, and UVs
template <>
inline constexpr AttributeType get_attribute_type<OpenMesh::Vec2f>() { return AttributeType::FLOAT2; }
template <>
inline constexpr AttributeType get_attribute_type<OpenMesh::Vec3f>() { return AttributeType::FLOAT3; }
template <>
inline constexpr AttributeType get_attribute_type<ei::Sphere>() { return AttributeType::SPHERE; }

// Executes some expression depending on the attribute type
#define OM_ATTRIB_SWITCH(type, expr)									\
	switch(type) {														\
		case AttributeType::CHAR: {										\
			using BaseType = char;										\
			expr;														\
		}	break;														\
		case AttributeType::UCHAR: {									\
			using BaseType = unsigned char;								\
			expr;														\
		}	break;														\
		case AttributeType::SHORT: {									\
			using BaseType = std::int16_t;								\
			expr;														\
		}	break;														\
		case AttributeType::USHORT: {									\
			using BaseType = std::uint16_t;								\
			expr;														\
		}	break;														\
		case AttributeType::INT: {										\
			using BaseType = std::int32_t;								\
			expr;														\
		}	break;														\
		case AttributeType::UINT: {										\
			using BaseType = std::uint32_t;								\
			expr;														\
		}	break;														\
		case AttributeType::LONG: {										\
			using BaseType = std::int64_t;								\
			expr;														\
		}	break;														\
		case AttributeType::ULONG: {									\
			using BaseType = std::uint64_t;								\
			expr;														\
		}	break;														\
		case AttributeType::FLOAT: {									\
			using BaseType = float;										\
			expr;														\
		}	break;														\
		case AttributeType::DOUBLE: {									\
			using BaseType = double;									\
			expr;														\
		}	break;														\
		case AttributeType::UCHAR2: {									\
			using BaseType = ei::Vec<unsigned char, 2u>;				\
			expr;														\
		}	break;														\
		case AttributeType::UCHAR3: {									\
			using BaseType = ei::Vec<unsigned char, 3u>;				\
			expr;														\
		}	break;														\
		case AttributeType::UCHAR4: {									\
			using BaseType = ei::Vec<unsigned char, 4u>;				\
			expr;														\
		}	break;														\
		case AttributeType::INT2: {										\
			using BaseType = ei::Vec<std::int32_t, 2u>;					\
			expr;														\
		}	break;														\
		case AttributeType::INT3: {										\
			using BaseType = ei::Vec<std::int32_t, 3u>;					\
			expr;														\
		}	break;														\
		case AttributeType::INT4: {										\
			using BaseType = ei::Vec<std::int32_t, 4u>;					\
			expr;														\
		}	break;														\
		case AttributeType::FLOAT2: {									\
			using BaseType = OpenMesh::Vec2f;							\
			expr;														\
		}	break;														\
		case AttributeType::FLOAT3: {									\
			using BaseType = OpenMesh::Vec3f;							\
			expr;														\
		}	break;														\
		case AttributeType::FLOAT4: {									\
			using BaseType = OpenMesh::Vec4f;							\
			expr;														\
		}	break;														\
		case AttributeType::SPHERE: {									\
			using BaseType = ei::Sphere;								\
			expr;														\
		}	break;														\
		default: throw std::runtime_error("Unknown property type!");	\
	}

inline constexpr std::size_t get_attribute_size(const AttributeType& type) {
	OM_ATTRIB_SWITCH(type, return sizeof(BaseType));
}

} // namespace mufflon::scene