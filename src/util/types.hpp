#pragma once

#include <OpenMesh/Core/Geometry/VectorT.hh> // TODO: remove (leftover, ei::vec replaces this)
#include <ei/vector.hpp>
#include <cstdint>

namespace mufflon {

using u8 = std::uint8_t;
using u16 = std::uint16_t;
using u32 = std::uint32_t;
using u64 = std::uint64_t;
using i8 = std::int8_t;
using i16 = std::int16_t;
using i32 = std::int32_t;
using i64 = std::int64_t;

using Real = float;

using Spectrum = ei::Vec3;

// Angle types
// Radians is the default type (used in all the trigonometric function).
// Therefore, it converts implicitly from and to diffrent representations.
class Radians {
	float a;
public:
	Radians(float a)		: a(a) {}
	operator float() const	{ return a; }
};
// Degrees type for (human) interfaces. More explicit to avoid errorneous
// convertions.
class Degrees {
	float a;
public:
	explicit Degrees(float a)		: a(a) {}
	explicit Degrees(Radians a)		: a(a / ei::PI * 180.0f) {}
	operator Radians()				{ return a * ei::PI / 180.0f; }
	explicit operator float() const { return a; }
};

using Pixel = ei::IVec2;
using Voxel = ei::IVec3;

} // namespace mufflon