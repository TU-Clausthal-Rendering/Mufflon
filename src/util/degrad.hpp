#pragma once

#include <ei/vector.hpp>

namespace mufflon {

// Angle types
// Radians is the default type (used in all the trigonometric function).
// Therefore, it converts implicitly from and to diffrent representations.
class Radians {
	float a = 0.f;
public:
	Radians() = default;
	Radians(float a) : a(a) {}
	operator float() const { return a; }
};
// Degrees type for (human) interfaces. More explicit to avoid errorneous
// convertions.
class Degrees {
	float a = 0.f;
public:
	explicit Degrees() = default;
	explicit Degrees(float a) : a(a) {}
	explicit Degrees(Radians a) : a(a / ei::PI * 180.0f) {}
	operator Radians() const { return a * ei::PI / 180.0f; }
	explicit operator float() const { return a; }
};

} // namespace mufflon