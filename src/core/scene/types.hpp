#pragma once

#include "util/types.hpp"

namespace mufflon { namespace scene {

	// Scene specific types go here

	using Point = ei::Vec3;
	using Direction = ei::Vec3;
	using Normal = Direction;
	using TangentU = Direction;
	using TangentV = Direction;
	using UvCoordinate = ei::Vec2;
	using MaterialIndex = u32;

	struct TangentSpace {
		Normal shadingN;		// The shading normal
		Normal geoN;			// The geometric normal (of the real traced surface)
		TangentU tU;
		TangentV tV;
	};

}} // namespace mufflon::scene