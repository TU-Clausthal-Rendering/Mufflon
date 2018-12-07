#pragma once

#include "util/types.hpp"
#include "util/assert.hpp"
#include "core/export/api.h"


namespace mufflon { namespace math {

// The following functions are taken from (taken from https://github.com/Jojendersie/Bim/blob/master/src/bim_sbvh.cpp)
// Two sources to derive the z-order comparator
// (floats - unused) http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.150.9547&rep=rep1&type=pdf
// (ints - the below one uses this int-algorithm on floats) http://dl.acm.org/citation.cfm?id=545444
// http://www.forceflow.be/2013/10/07/morton-encodingdecoding-through-bit-interleaving-implementations/ Computing morton codes
	// Expand a number 16bit to 48bit by inserting two 0 bits between all other bits.
CUDA_FUNCTION constexpr u64 part_by_two16(const u16 x) {
	u64 r = x;
	r = (r | (r << 16)) & 0x0000ff0000ff;
	r = (r | (r <<  8)) & 0x00f00f00f00f;
	r = (r | (r <<  4)) & 0x0c30c30c30c3;
	r = (r | (r <<  2)) & 0x249249249249;
	return r;
}

// Same partitioning as above but for 21 bits (maximum possible with 63 bit of 64 used in output).
CUDA_FUNCTION constexpr u64 part_by_two21(const u32 x) {
	mAssert((x & 0xffe00000) == 0);
	u64 r = x;
	r = (r | (r << 32)) & 0x001f00000000ffff;
	r = (r | (r << 16)) & 0x001f0000ff0000ff;
	r = (r | (r <<  8)) & 0x100f00f00f00f00f;
	r = (r | (r <<  4)) & 0x10c30c30c30c30c3;
	r = (r | (r <<  2)) & 0x1249249249249249;
	return r;
}

CUDA_FUNCTION constexpr u32 part_by_two10(const u32 x) {
	mAssert((x & 0xfffffc00) == 0);
	u32 r = x;
	r = (r * 0x00010001u) & 0xFF0000FFu;
	r = (r * 0x00000101u) & 0x0F00F00Fu;
	r = (r * 0x00000011u) & 0xC30C30C3u;
	r = (r * 0x00000005u) & 0x49249249u;
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
	return part_by_two16(a) | (part_by_two16(b) << 1u) | (part_by_two16(c) << 2u);
};

// Encodes 3 16-bit values into a single 64 bit value on a Hilbert-curve.
CUDA_FUNCTION constexpr u64 get_hilbert_code(const u16 a, const u16 b, const u16 c) {
	return gray_to_binary(get_morton_code(a, b, c));
};


}} // namespace mufflon::math