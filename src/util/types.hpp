#pragma once

#include <OpenMesh/Core/Geometry/VectorT.hh>
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

// TODO: replace with proper math library
using Vec2f = OpenMesh::Vec2f;
using Vec3f = OpenMesh::Vec3f;
using Vec4f = OpenMesh::Vec4f;

using Real = float;

} // namespace mufflon