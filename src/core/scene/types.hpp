#pragma once

#include "util/types.hpp"

namespace mufflon { namespace scene {

	// Scene specific types go here

	using Point = ei::Vec3;
	using Direction = ei::Vec3;
	using Normal = Direction;
	using Tangent = Direction;
	using UvCoordinate = ei::Vec2;
	using MaterialIndex = u16;

	struct TangentSpace {
		Normal shadingN;		// The shading normal
		Normal geoN;			// The geometric normal (of the real traced surface)
		Tangent shadingTX;		// First tangent: shadingTX ⊥ shadingN
		Tangent shadingTY;		// Second tangent: shadingTY ⊥ shadingN ∧ shadingTX ⊥ shadingTY
	};

}} // namespace mufflon::scene