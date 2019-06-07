#pragma once

#include "degrad.hpp"
#include "int_types.hpp"
#include "ei/vector.hpp"
#include <cuda_runtime.h>
#include <limits>
#include <cstdint>
#include <cstddef>

namespace mufflon {

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
	__host__ __device__ AreaPdf& operator*=(float probability) noexcept {
		m_pdf *= probability;
		return *this;
	}
	__host__ __device__ AreaPdf& operator/=(float probability) noexcept {
		m_pdf /= probability;
		return *this;
	}
	__host__ __device__ AreaPdf operator+(AreaPdf pdf) const noexcept {
		return static_cast<AreaPdf>(m_pdf + pdf.m_pdf);
	}
	__host__ __device__ AreaPdf operator-(AreaPdf pdf) const noexcept {
		return static_cast<AreaPdf>(m_pdf - pdf.m_pdf);
	}
	// Returnvalue is AreaSqPdf (not an AreaPdf)
	__host__ __device__ float operator*(AreaPdf pdf) const noexcept {
		return m_pdf * pdf.m_pdf;
	}
	// Return values is unitless
	__host__ __device__ float operator/(AreaPdf pdf) const noexcept {
		return m_pdf / pdf.m_pdf;
	}
	__host__ __device__ AngularPdf to_angular_pdf(Real cos, Real distSqr) const noexcept;
	__host__ __device__ static constexpr AreaPdf infinite() {
		return AreaPdf{ 4294967296.f }; // In theory infinite, but must behave well in float32 (power of two does not change the mantissa, shortens out anyway)
	}

	__host__ __device__ bool is_infinite() const { return m_pdf >= float(infinite()); }
	__host__ __device__ bool is_zero() const { return m_pdf <= 0.0f; }
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
	__host__ __device__ AngularPdf& operator*=(float probability) noexcept {
		m_pdf *= probability;
		return *this;
	}
	__host__ __device__ AngularPdf& operator/=(float probability) noexcept {
		m_pdf /= probability;
		return *this;
	}
	__host__ __device__ AngularPdf operator+(AngularPdf pdf) const noexcept {
		return static_cast<AngularPdf>(m_pdf + pdf.m_pdf);
	}
	__host__ __device__ AngularPdf operator-(AngularPdf pdf) const noexcept {
		return static_cast<AngularPdf>(m_pdf - pdf.m_pdf);
	}
	__host__ __device__ float operator*(AngularPdf pdf) const noexcept {
		return m_pdf * pdf.m_pdf;
	}
	__host__ __device__ AngularPdf operator*(float probability) const noexcept {
		return AngularPdf{ m_pdf * probability };
	}
	__host__ __device__ float operator/(AngularPdf pdf) const noexcept {
		return m_pdf / pdf.m_pdf;
	}
	__host__ __device__ AngularPdf operator/(float probability) const noexcept {
		return AngularPdf{ m_pdf / probability };
	}
	__host__ __device__ AreaPdf to_area_pdf(Real cos, Real distSqr) const noexcept {
		return AreaPdf{ m_pdf * ei::abs(cos) / distSqr };
	}
	__host__ __device__ static constexpr AngularPdf infinite() {
		return AngularPdf{ 4294967296.f }; // In theory infinite, but must behave well in float32 (power of two does not change the mantissa, shortens out anyway)
	}

	__host__ __device__ bool is_infinite() const { return m_pdf >= float(infinite()); }
	__host__ __device__ bool is_zero() const { return m_pdf <= 0.0f; }
private:
	Real m_pdf = 0.f;
};

__host__ __device__ inline AngularPdf AreaPdf::to_angular_pdf(Real cos, Real distSqr) const noexcept {
	return AngularPdf{ m_pdf * distSqr / ei::abs(cos) };
}


// Save division (avoids division by 0). Required in robust implementations of materials
// MIS, ...
template<typename T, typename D>
__host__ __device__ inline decltype(std::declval<T>() / std::declval<D>())
sdiv(const T& x, const D& d) {
	return x / (d + T{ 1e-20f } * ei::sgn(d));
}

} // namespace mufflon
