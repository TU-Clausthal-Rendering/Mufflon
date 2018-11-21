#pragma once

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

using Pixel = ei::IVec2;
using Voxel = ei::IVec3;

} // namespace mufflon