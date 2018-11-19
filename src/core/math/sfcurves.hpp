#pragma once

#include "util/types.hpp"
#include "export/api.hpp"


namespace mufflon { namespace math {

// The following functions are taken from (taken from https://github.com/Jojendersie/Bim/blob/master/src/bim_sbvh.cpp)
// Two sources to derive the z-order comparator
// (floats - unused) http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.150.9547&rep=rep1&type=pdf
// (ints - the below one uses this int-algorithm on floats) http://dl.acm.org/citation.cfm?id=545444
// http://www.forceflow.be/2013/10/07/morton-encodingdecoding-through-bit-interleaving-implementations/ Computing morton codes
CUDA_FUNCTION constexpr u64 part_by_two(const u16 x) {
	u64 r = x;
	r = (r | (r << 16)) & 0x0000ff0000ff;
	r = (r | (r << 8)) & 0x00f00f00f00f;
	r = (r | (r << 4)) & 0x0c30c30c30c3;
	r = (r | (r << 2)) & 0x249249249249;
	return r;
}

// Converts gray-code to regular binary
CUDA_FUNCTION constexpr u64 gray_to_binary(u64 num) {
	num = num ^ (num >> 32);
	num = num ^ (num >> 16);
	num = num ^ (num >> 8);
	num = num ^ (num >> 4);
	num = num ^ (num >> 2);
	num = num ^ (num >> 1);
	return num;
};

// Converts binary to gray-code
CUDA_FUNCTION constexpr u64 binary_to_gray(u64 num) {
	return num ^ (num >> 1);
}

// Encodes 3 16-bit values into a single 64 bit value on a Z-curve.
CUDA_FUNCTION constexpr u64 get_morton_code(const u16 a, const u16 b, const u16 c) {
	return part_by_two(a) | (part_by_two(b) << 1u) | (part_by_two(c) << 2u);
};

// Encodes 3 16-bit values into a single 64 bit value on a Hilbert-curve.
CUDA_FUNCTION constexpr u64 get_hilbert_code(const u16 a, const u16 b, const u16 c) {
	return gray_to_binary(get_morton_code(a, b, c));
};


}} // namespace mufflon::math