#pragma once

#include "util/vectors.hpp"
#include <cstdint>

namespace mufflon::scene {

// TODO
struct MaterialIndex {
	std::uint16_t index;
};

struct Sphere {
	util::Vec3 position;
	float radius;
	MaterialIndex mat_index;
};

} // namespace mufflon::scene