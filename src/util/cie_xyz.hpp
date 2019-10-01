#pragma once

#include <ei/vector.hpp>

namespace mufflon::spectrum {

// Strong typedef's to not confuse the units
struct Kelvin {
	double value;
};

ei::Vec3 compute_black_body_color(const Kelvin temperature, const bool normalize = true);

} // namespace mufflon::spectrum