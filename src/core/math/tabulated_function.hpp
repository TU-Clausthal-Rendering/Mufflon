#pragma once

#include <array>

namespace mufflon { namespace math {

// Helper for lookup table functions.
// Proviedes and interpolated sampling on the table including a bounds check.
template<int N>
class TabulatedFunction {
//	static_assert(XMAX > XMIN, "Invalid difinition range.");
	float m_xmin, m_xmax;
	float m_interval;
	std::array<float, N> m_table;
public:
	constexpr TabulatedFunction(float xmin, float xmax, const std::array<float, N>& table) :
		m_xmin{xmin},
		m_xmax{xmax},
		m_interval{xmax - xmin},
		m_table{table}
	{}

	constexpr float operator () (float x) const noexcept {
		if(x <= m_xmin) return m_table[0];
		if(x >= m_xmax) return m_table[N-1];
		x = (N-1) * (x - m_xmin) / m_interval;	// Rescale
		int i = static_cast<int>(x);			// Trancate
		if(i == N-1) --i;						// Only possible if x == XMAX or very close
		float t = x - i;						// Interpolant
		return m_table[i] * (1-t) + m_table[i+1] * t;
	}
};

}} // namespace mufflon::math