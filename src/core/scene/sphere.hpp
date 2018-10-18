#pragma once

#include "util/types.hpp"
#include <cstdint>

namespace mufflon::scene {

// TODO
struct MaterialIndex {
	std::uint16_t index;
};

struct Sphere {
	Vec3f position;
	Real radius;
	MaterialIndex matIndex;
};

} // namespace mufflon::scene