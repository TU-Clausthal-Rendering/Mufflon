#pragma once

#include "util/types.hpp"

namespace mufflon {
namespace CuLib {

/// <summary> Sort the lists of keys and values using CUB's radix sort implementation
///           (http://nvlabs.github.io/cub/). Does not preserve input arrays. </summary>
///
/// <remarks> Leonardo, 12/29/2014. </remarks>
///
/// <param name="numElements"> Number of elements. </param>
/// <param name="keysIn">           [in,out] Input keys. </param>
/// <param name="keysOut">          [in,out] Output keys. </param>
/// <param name="valuesIn">         [in,out] Input values. </param>
/// <param name="valuesOut">        [in,out] Output values. </param>
///
/// <returns> Execution time, in milliseconds. </returns>
// 
float DeviceSort(u32 numElements, u32** keysIn, u32** keysOut,
	u32** valuesIn, u32** valuesOut);

// TODO add comment
float DeviceSort(u32 numElements, u64** keysIn, u64** keysOut,
	u32** valuesIn, u32** valuesOut);

float DeviceSort(u32 numElements, u64 ** keysIn, u64 ** keysOut, i32 ** valuesIn, i32 ** valuesOut);

float DeviceSort(u32 numElements, u32 ** keysIn, u32 ** keysOut, i32 ** valuesIn, i32 ** valuesOut);

float DeviceSort(u32 numElements, u32** keysIn, u32** keysOut,
	u32** valuesIn, u32** valuesOut);


/// <summary> Sort the lists of keys and values using CUB's radix sort implementation
///           (http://nvlabs.github.io/cub/). Does not preserve input arrays. </summary>
///
/// <remarks> Feng, 15/11/2017. </remarks>
///
/// <param name="numElements"> Number of elements. </param>
/// <param name="keysIn">           [in,out] Input keys. </param>
/// <param name="keysOut">          [in,out] Output keys. </param>
/// <param name="valuesIn">         [in,out] Input values. </param>
/// <param name="valuesOut">        [in,out] Output values. </param>
///
/// <returns> Execution time, in milliseconds. </returns>
float DeviceSort(u32 numElements, i32** keysIn,
	i32** keysOut, u32** valuesIn, u32** valuesOut);

float DeviceSort(u32 numElements, float** keysIn, float** keysOut,
	u32** valuesIn, u32** valuesOut);

float DeviceSortDescending(u32 numElements, float** keysIn, float** keysOut,
	u32** valuesIn, u32** valuesOut);

/// <summary> Wrapper for cub::DeviceReduce::Sum(). If a temporary memory array is not specified, 
///           calculates the required temporary memory size and returns.</summary>
///
/// <remarks> Leonardo, 12/29/2014. </remarks>
///
/// <param name="numElements"> Number of triangles. </param>
/// <param name="in">                Input values. </param>
/// <param name="out">               [out] Output value. </param>
/// <param name="tempMemorySize">    [in,out] If non-null, the size. </param>
/// <param name="tempMemory">        [in,out] (Optional) If non-null, the temporary memory array. 
///                                  </param>
void DeviceSum(u32 numElements, i32* in, i32* out, size_t* size,
	void* tempMemory = nullptr);

/// <summary> Sum the elements from the specified array. Wrapper for cub::DeviceReduce::Sum().
///           </summary>
///
/// <remarks> Leonardo, 12/29/2014. </remarks>
///
/// <param name="numElements"> Number of elements in the array. </param>
/// <param name="elements">         Elements to be summed. </param>
///
/// <returns> The sum of the input values. </returns>
i32 DeviceSum(u32 numElements, i32* elements);

/// <summary> Sum the elements from the specified array. Wrapper for cub::DeviceReduce::Sum().
///           </summary>
///
/// <remarks> Leonardo, 12/29/2014. </remarks>
///
/// <param name="numElements"> Number of elements in the array. </param>
/// <param name="elements">         Elements to be summed. </param>
///
/// <returns> The sum of the input values. </returns>
i32 DeviceSum(u32 numElements, u32* elements);

/// <summary> Sum the elements from the specified array. Wrapper for cub::DeviceReduce::Sum().
///           </summary>
///
/// <remarks> Leonardo, 12/29/2014. </remarks>
///
/// <param name="numElements"> Number of elements in the array. </param>
/// <param name="elements">         Elements to be summed. </param>
///
/// <returns> The sum of the input values. </returns>
float DeviceSum(u32 numElements, float* elements);

// @feng
// Can possibly be used inplace
float DeviceExclusiveSum(i32 numElements, const i32* dataIn, i32* dataOut);

float DeviceInclusiveSum(i32 numElements, const i32* valuesIn, i32* valuesOut);
}
}