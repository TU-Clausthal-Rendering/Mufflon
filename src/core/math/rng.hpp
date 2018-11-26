#pragma once

#include "util/types.hpp"
#include "export/api.hpp"
#include "sfcurves.hpp"


namespace mufflon { namespace math {

// Splitmix 64 to generate seeding values
// http://dx.doi.org/10.1145/2714064.2660195
CUDA_FUNCTION u64 scramble_seed(u32 x) {
	u64 z = x + 0x9e3779b97f4a7c15ull;
	z = (z ^ (z >> 30u)) * 0xbf58476d1ce4e5b9ull;
	z = (z ^ (z >> 27u)) * 0x94d049bb133111ebull;
	return z ^ (z >> 31u);
}
// TODO: test good old wang hash in comparison?

// Xoroshiro128+ generator from David Blackman and Sebastiano Vigna
// http://vigna.di.unimi.it/xorshift/xoroshiro128plus.c
class Xoroshiro128 {
public:
	__host__ __device__ Xoroshiro128() {
		m_state[0] = 0x2f0ae9bc;
		m_state[1] = 0x6431af73;
	}

	__host__ __device__ Xoroshiro128(u32 seed) {
		m_state[1] = scramble_seed(seed);
		m_state[0] = scramble_seed(seed+1);
		next();
	}

	// Restore the RNG from a pure state
	__host__ __device__ Xoroshiro128(const ei::UVec4& state) {
		m_state[0] = u64(state.x) | (u64(state.y) << 32ull);
		m_state[1] = u64(state.z) | (u64(state.w) << 32ull);
	}
	// Dump the state (e.g. for storing in textures)
	ei::UVec4 get_state() const {
		return {m_state[0] & 0xfffffff, m_state[0] >> 32ull,
				m_state[1] & 0xfffffff, m_state[1] >> 32ull};
	}

	__host__ __device__ u64 next() {
		const u64 s0 = m_state[0];
		u64 s1 = m_state[1];
		const u64 result = s0 + s1;
		// Use the following line for the higher quality Xoroshiro**
		//const u64 result = rotl(s0 * 5, 7) * 9;

		s1 ^= s0;
		m_state[0] = rotl(s0, 24) ^ s1 ^ (s1 << 16); // a, b
		m_state[1] = rotl(s1, 37); // c

		return result;
	}
private:
	u64 m_state[2];		// The state: 128bit

	__host__ __device__ u64 rotl(const u64 x, const u64 k) const {
		return (x << k) | (x >> (64u - k));
	}
};

// A variant of the PCG generator http://www.pcg-random.org with 64bit
// state and an output of 64bit numbers as the above genertor.
// Output permutation: RXS-M-XS (see https://en.wikipedia.org/wiki/Permuted_congruential_generator)
// TODO: test which one is faster.
class PCG64 {
public:
	__host__ __device__ PCG64() {
		m_state = 0x2f0ae9bc6431af73ull;
	}

	__host__ __device__ PCG64(u32 seed) {
		m_state = scramble_seed(seed);
		next();
		m_state += seed;
		next();
	}

	__host__ __device__ u64 next() {
		u64 x = m_state;
		const u64 count = m_state >> 59ull;
		m_state = m_state * 6364136223846793005ull + 19823657895641ull;
		x ^= x >> (5 + count);
		x *= 12605985483714917081ull;
		return x ^ (x >> 43);
	}
private:
	u64 m_state;		// The state: 64bit
};


// Low discrepancy sampler: golden ratio (additive recurrence) sequence 1D
class GoldenRatio1D {
public:
	__host__ __device__ GoldenRatio1D(u32 seed) {
		u64 z = scramble_seed(seed);
		m_state = static_cast<u32>(z ^ (z >> 32u));
	}

	__host__ __device__ u32 next() {
		u32 oldState = m_state;
		m_state += 2654435769u;
		return oldState;
	}
private:
	u32 m_state;
};

// Low discrepancy sampler: golden ratio (additive recurrence) sequence 2D
// Generalizes the sequence along a space filling curve like in:
// "Van der Corput and Golden Ratio Sequences Along the Hilbert Space-Filling Curve"
class GoldenRatio2D {
public:
	__host__ __device__ GoldenRatio2D(u32 seed) {
		m_state = scramble_seed(seed);
	}

	// Get 2 independent 32bit random values packed in 64bit
	__host__ __device__ u64 next() {
		u64 s = m_state;
		m_state += 11400714819323198485ull;
		// Inverse transformation of Hilbert-Curve to 2D point
		s = binary_to_gray(s);	// Hilbert -> Morton
		// Inverse morton: Upack ....BABABABA into two separate
		// words (i.e. each second bit belongs to one coordinate).
		s = ((s >>  1) & 0x2222222222222222ull) | ((s <<  1) & 0x4444444444444444ull) | (s & 0x9999999999999999ull);	// ...BBAABBAA
		s = ((s >>  2) & 0x0c0c0c0c0c0c0c0cull) | ((s <<  2) & 0x3030303030303030ull) | (s & 0xc3c3c3c3c3c3c3c3ull);	// ...BBBBAAAA
		s = ((s >>  4) & 0x00f000f000f000f0ull) | ((s <<  4) & 0x0f000f000f000f00ull) | (s & 0xf00ff00ff00ff00full);	// ...BBBBBBBBAAAAAAAA
		s = ((s >>  8) & 0x0000ff000000ff00ull) | ((s <<  8) & 0x00ff000000ff0000ull) | (s & 0xff0000ffff0000ffull);
		s = ((s >> 16) & 0x00000000ffff0000ull) | ((s << 16) & 0x0000ffff00000000ull) | (s & 0xffff00000000ffffull);
		return s;
	}
private:
	u64 m_state;
};



// Use Rng globally, this point can be used to switch between the two
// generators Xoroshiro128 and PCG64.
// Low discrepancy sequences should be used independently of this
using Rng = Xoroshiro128;

}} // namespace mufflon::math