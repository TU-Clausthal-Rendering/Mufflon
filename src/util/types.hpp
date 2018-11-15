#pragma once

#include "ei/vector.hpp"
#include <cuda_runtime.h>
#include <limits>
#include <cstdint>
#include <cstddef>

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
	float a = 0.f;
public:
	Radians() = default;
	Radians(float a)		: a(a) {}
	operator float() const	{ return a; }
};
// Degrees type for (human) interfaces. More explicit to avoid errorneous
// convertions.
class Degrees {
	float a = 0.f;
public:
	explicit Degrees() = default;
	explicit Degrees(float a)		: a(a) {}
	explicit Degrees(Radians a)		: a(a / ei::PI * 180.0f) {}
	operator Radians()				{ return a * ei::PI / 180.0f; }
	explicit operator float() const { return a; }
};

class AngularPdf;

// PDF types. Either per area or per steradians
class AreaPdf {
public:
	__host__ __device__ constexpr AreaPdf() : m_pdf{ 0 } {}
	__host__ __device__ explicit constexpr AreaPdf(Real pdf) noexcept : m_pdf(pdf) {}
	__host__ __device__ explicit constexpr operator Real() const noexcept { return m_pdf; }
	__host__ __device__ AreaPdf& operator+=(AreaPdf pdf) noexcept {
		m_pdf += pdf.m_pdf;
		return *this;
	}
	__host__ __device__ AreaPdf& operator-=(AreaPdf pdf) noexcept {
		m_pdf -= pdf.m_pdf;
		return *this;
	}
	__host__ __device__ AreaPdf& operator*=(AreaPdf pdf) noexcept {
		m_pdf *= pdf.m_pdf;
		return *this;
	}
	__host__ __device__ AreaPdf& operator/=(AreaPdf pdf) noexcept {
		m_pdf /= pdf.m_pdf;
		return *this;
	}
	__host__ __device__ constexpr AngularPdf to_angular_pdf(Real cos, Real distSqr) const noexcept;
	__host__ __device__ static constexpr AreaPdf infinite() {
		return AreaPdf{ std::numeric_limits<float>::infinity() };
	}

private:
	Real m_pdf = 0.f;
};
class AngularPdf {
public:
	__host__ __device__ constexpr AngularPdf() : m_pdf{ 0 } {}
	__host__ __device__ explicit constexpr AngularPdf(Real pdf) noexcept : m_pdf(pdf) {}
	__host__ __device__ explicit constexpr operator Real() const noexcept { return m_pdf; }
	__host__ __device__ AngularPdf& operator+=(AngularPdf pdf) noexcept {
		m_pdf += pdf.m_pdf;
		return *this;
	}
	__host__ __device__ AngularPdf& operator-=(AngularPdf pdf) noexcept {
		m_pdf -= pdf.m_pdf;
		return *this;
	}
	__host__ __device__ AngularPdf& operator*=(AngularPdf pdf) noexcept {
		m_pdf *= pdf.m_pdf;
		return *this;
	}
	__host__ __device__ AngularPdf& operator/=(AngularPdf pdf) noexcept {
		m_pdf /= pdf.m_pdf;
		return *this;
	}
	__host__ __device__ constexpr AreaPdf to_area_pdf(Real cos, Real distSqr) const noexcept {
		return AreaPdf{ m_pdf * cos / distSqr };
	}
	__host__ __device__ static constexpr AngularPdf infinite() {
		return AngularPdf{std::numeric_limits<float>::infinity()};
	}

private:
	Real m_pdf = 0.f;
};

__host__ __device__ constexpr AngularPdf AreaPdf::to_angular_pdf(Real cos, Real distSqr) const noexcept {
	return AngularPdf{ m_pdf * distSqr / cos };
}

using Pixel = ei::IVec2;
using Voxel = ei::IVec3;

} // namespace mufflon
