#pragma once

#include "util/types.hpp"
#include "util/assert.hpp"

namespace mufflon {
namespace CuLib {

/*
 * Sort the lists of keys and values using CUB's radix sort implementation
 * (http://nvlabs.github.io/cub/).
 * The template is instanciated for:
 *		KeyT	ValueT
 *		u32		i32
 *		u64		i32
 * If you need further type please add them in cu_lib_wrapper.cu.
 */
template < typename KeyT, typename ValueT >
float DeviceSort(u32 numElements, 
	const KeyT* keysIn, KeyT* keysOut,
	const ValueT* valuesIn, ValueT* valuesOut);

// @feng
// Can possibly be used inplace
float DeviceExclusiveSum(i32 numElements, const i32* dataIn, i32* dataOut);

float DeviceInclusiveSum(i32 numElements, const i32* valuesIn, i32* valuesOut);
}
}