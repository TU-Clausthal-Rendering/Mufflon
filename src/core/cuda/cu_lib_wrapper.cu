#include "cu_lib_wrapper.h"

#include <cub\cub.cuh>
//#include "Utilities\cudaHeaders.h"

#include <iostream>
#include <fstream>
#include <string>

namespace mufflon {namespace CuLib {


// In and out buffers may be swaped.
// Original data is not kept.
template <typename KeyT, typename ValueT> float DeviceSort(u32 numElements, 
	KeyT** keysIn, KeyT** keysOut,
	ValueT** valuesIn, ValueT** valuesOut)
{
	KeyT* tmpKeysOut;
	ValueT *tmpValuesOut;
	if (keysIn == keysOut) {
		cudaMalloc((void **)&tmpKeysOut, numElements * sizeof(KeyT));
	}
	else {
		tmpKeysOut = *keysOut;
	}

	if (valuesIn == valuesOut) {
		cudaMalloc((void **)&tmpValuesOut, numElements * sizeof(ValueT));
	}
	else {
		tmpValuesOut = *valuesOut;
	}

	cub::DoubleBuffer<KeyT> keysBuffer(*keysIn, tmpKeysOut);
	cub::DoubleBuffer<ValueT> valuesBuffer(*valuesIn, tmpValuesOut);

	// Check how much temporary memory will be required.
	void* tempStorage = nullptr;
	size_t storageSize = 0;
	cub::DeviceRadixSort::SortPairs(tempStorage, storageSize, keysBuffer, valuesBuffer,
		numElements);

	// Allocate temporary memory.
	cudaMalloc(&tempStorage, storageSize);

	float elapsedTime = 0.0f;
#ifdef MEASURE_EXECUTION_TIMES
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
#endif

	// Sort
	cub::DeviceRadixSort::SortPairs(tempStorage, storageSize, keysBuffer, valuesBuffer,
		numElements);

#ifdef MEASURE_EXECUTION_TIMES
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
#endif

	// Free temporary memory.
	cudaFree(tempStorage);

	// Update in buffers.
	KeyT* current = keysBuffer.d_buffers[1 - keysBuffer.selector];
	if (keysIn != keysOut) {
		*keysIn = current;
	}
	else {
		cudaFree(current);
	}
	ValueT* current2 = valuesBuffer.d_buffers[1 - valuesBuffer.selector];
	if (valuesIn != valuesOut) {
		*valuesIn = current2;
	}
	else {
		cudaFree(current2);
	}

	// Update out buffers.
	current = keysBuffer.Current();
	*keysOut = current;
	current2 = valuesBuffer.Current();
	*valuesOut = current2;

	return elapsedTime;
}

#if 0
// In and out buffers may be swaped.
// Original data is not kept.
template <typename T> float DeviceSort(u32 numElements, T** keysIn, T** keysOut,
	u32** valuesIn, u32** valuesOut)
{
	T* tmpKeysOut;
	u32 *tmpValuesOut;
	if (keysIn == keysOut) {
		cudaMalloc((void **)&tmpKeysOut, numElements * sizeof(T));
	}
	else {
		tmpKeysOut = *keysOut;
	}

	if (valuesIn == valuesOut) {
		cudaMalloc((void **)&tmpValuesOut, numElements * sizeof(u32));
	}
	else {
		tmpValuesOut = *valuesOut;
	}

	cub::DoubleBuffer<T> keysBuffer(*keysIn, tmpKeysOut);
	cub::DoubleBuffer<u32> valuesBuffer(*valuesIn, tmpValuesOut);

	// Check how much temporary memory will be required.
	void* tempStorage = nullptr;
	size_t storageSize = 0;
	cub::DeviceRadixSort::SortPairs(tempStorage, storageSize, keysBuffer, valuesBuffer,
		numElements);

	// Allocate temporary memory.
	cudaMalloc(&tempStorage, storageSize);

	float elapsedTime = 0.0f;
#ifdef MEASURE_EXECUTION_TIMES
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
#endif

	// Sort
	cub::DeviceRadixSort::SortPairs(tempStorage, storageSize, keysBuffer, valuesBuffer,
		numElements);

#ifdef MEASURE_EXECUTION_TIMES
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
#endif

	// Free temporary memory.
	cudaFree(tempStorage);

	// Update in buffers.
	T* current = keysBuffer.d_buffers[1 - keysBuffer.selector];
	if (keysIn != keysOut) {
		*keysIn = current;
	}
	else {
		cudaFree(current);
	}
	u32* current2 = valuesBuffer.d_buffers[1 - valuesBuffer.selector];
	if (valuesIn != valuesOut) {
		*valuesIn = current2;
	}
	else {
		cudaFree(current2);
	}

	// Update out buffers.
	current = keysBuffer.Current();
	*keysOut = current;
	current2 = valuesBuffer.Current();
	*valuesOut = current2;

	return elapsedTime;
}
#endif // 0


// In and out buffers may be swaped.
// Original data is not kept.
template <typename T> float DeviceSortDescending(u32 numElements, T** keysIn, T** keysOut,
	u32** valuesIn, u32** valuesOut)
{
	T* tmpKeysOut;
	u32 *tmpValuesOut;
	if (keysIn == keysOut) {
		cudaMalloc((void **)&tmpKeysOut, numElements * sizeof(T));
	}
	else {
		tmpKeysOut = *keysOut;
	}

	if (valuesIn == valuesOut) {
		cudaMalloc((void **)&tmpValuesOut, numElements * sizeof(u32));
	}
	else {
		tmpValuesOut = *valuesOut;
	}

	cub::DoubleBuffer<T> keysBuffer(*keysIn, tmpKeysOut);
	cub::DoubleBuffer<u32> valuesBuffer(*valuesIn, tmpValuesOut);

	// Check how much temporary memory will be required
	void* tempStorage = nullptr;
	size_t storageSize = 0;
	cub::DeviceRadixSort::SortPairsDescending(tempStorage, storageSize, keysBuffer, valuesBuffer,
		numElements);

	// Allocate temporary memory
	cudaMalloc(&tempStorage, storageSize);

	float elapsedTime = 0.0f;
#ifdef MEASURE_EXECUTION_TIMES
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
#endif

	// Sort.
	cub::DeviceRadixSort::SortPairs(tempStorage, storageSize, keysBuffer, valuesBuffer,
		numElements);

#ifdef MEASURE_EXECUTION_TIMES
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
#endif

	// Free temporary memory
	cudaFree(tempStorage);

	// Update in buffers
	T* current = keysBuffer.d_buffers[1 - keysBuffer.selector];
	if (keysIn != keysOut) {
		*keysIn = current;
	}
	else {
		cudaFree(current);
	}
	u32* current2 = valuesBuffer.d_buffers[1 - valuesBuffer.selector];
	if (valuesIn != valuesOut) {
		*valuesIn = current2;
	}
	else {
		cudaFree(current2);
	}

	// Update out buffers
	current = keysBuffer.Current();
	*keysOut = current;
	current2 = valuesBuffer.Current();
	*valuesOut = current2;

	return elapsedTime;
}


float DeviceSort(u32 numElements, u64 ** keysIn, u64 ** keysOut, u32 ** valuesIn, u32 ** valuesOut)
{
	return DeviceSort<u64, u32>(numElements, keysIn, keysOut, valuesIn, valuesOut);
}

float DeviceSort(u32 numElements, u64 ** keysIn, u64 ** keysOut, i32 ** valuesIn, i32 ** valuesOut)
{
	return DeviceSort<u64, i32>(numElements, keysIn, keysOut, valuesIn, valuesOut);
}

float DeviceSort(u32 numElements, u32 ** keysIn, u32 ** keysOut, i32 ** valuesIn, i32 ** valuesOut)
{
	return DeviceSort<u32, i32>(numElements, keysIn, keysOut, valuesIn, valuesOut);
}

float DeviceSort(u32 numElements, u32** keysIn, u32** keysOut,
	u32** valuesIn, u32** valuesOut)
{
	return DeviceSort<u32, u32>(numElements, keysIn, keysOut, valuesIn, valuesOut);
}

float DeviceSort(u32 numElements, float** keysIn, float** keysOut,
	u32** valuesIn, u32** valuesOut)
{
	return DeviceSort<float, u32>(numElements, keysIn, keysOut, valuesIn, valuesOut);
}

float DeviceSortDescending(u32 numElements, float** keysIn, float** keysOut,
	u32** valuesIn, u32** valuesOut)
{
	return DeviceSortDescending<float>(numElements, keysIn, keysOut, valuesIn, valuesOut);
}

//void DeviceSum(u32 numTriangles, int* in, int* out, size_t* tempMemorySize, void* tempMemory)
//{
//    cub::DeviceReduce::Sum(tempMemory, *tempMemorySize, in, out, numTriangles);
//}

template <class T> T DeviceSum(u32 numElements, T* elements)
{
	T* deviceElementsSum;
	cudaMalloc(&deviceElementsSum, sizeof(T));

	// Calculate the required temporary memory size
	void* tempStorage = nullptr;
	size_t tempStorageSize = 0;
	cub::DeviceReduce::Sum(tempStorage, tempStorageSize, elements, deviceElementsSum,
		numElements);

	// Allocate temporary memory
	cudaMalloc(&tempStorage, tempStorageSize);

	// Sum priorities
	cub::DeviceReduce::Sum(tempStorage, tempStorageSize, elements, deviceElementsSum,
		numElements);

	// Read priorities sum from device memory
	T elementsSum;
	cudaMemcpy(&elementsSum, deviceElementsSum, sizeof(T), cudaMemcpyDeviceToHost);

	// Free temporary memory
	cudaFree(tempStorage);
	cudaFree(deviceElementsSum);

	return elementsSum;
}

template <class T> T DeviceMax(u32 numElements, T* elements)
{
	T* deviceElementsSum;
	cudaMalloc(&deviceElementsSum, sizeof(T));

	// Calculate the required temporary memory size
	void* tempStorage = nullptr;
	size_t tempStorageSize = 0;
	cub::DeviceReduce::Max(tempStorage, tempStorageSize, elements, deviceElementsSum,
		numElements);

	// Allocate temporary memory
	cudaMalloc(&tempStorage, tempStorageSize);

	// Sum priorities
	cub::DeviceReduce::Max(tempStorage, tempStorageSize, elements, deviceElementsSum,
		numElements);

	// Read priorities sum from device memory
	T elementsSum;
	cudaMemcpy(&elementsSum, deviceElementsSum, sizeof(T), cudaMemcpyDeviceToHost);

	// Free temporary memory
	cudaFree(tempStorage);
	cudaFree(deviceElementsSum);

	return elementsSum;
}

int DeviceSum(u32 numElements, int* elements)
{
	return DeviceSum<int>(numElements, elements);
}

float DeviceSum(u32 numElements, float* elements)
{
	return DeviceSum<float>(numElements, elements);
}

int DeviceMax(u32 numElements, int* elements)
{
	return DeviceMax<int>(numElements, elements);
}

// ref. https://nvlabs.github.io/cub/structcub_1_1_device_scan.html#a83236fc272c0b573a2bb2c5b47e0867d
template <typename T> float DeviceExclusiveSum(int numItems, T** valuesIn, T** valuesOut)
{
	T* tmpValuesOut;
	if (valuesIn == valuesOut) {
		cudaMalloc((void **)&tmpValuesOut, numItems * sizeof(T));
	}
	else {
		tmpValuesOut = *valuesOut;
	}

	// Check how much temporary memory will be required
	void* tempStorage = nullptr;
	size_t storageSize = 0;
	cub::DeviceScan::ExclusiveSum(tempStorage, storageSize, *valuesIn, tmpValuesOut, numItems);

	// Allocate temporary memory
	cudaMalloc(&tempStorage, storageSize);

	float elapsedTime = 0.0f;
#ifdef MEASURE_EXECUTION_TIMES
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
#endif

	// Scan
	cub::DeviceScan::ExclusiveSum(tempStorage, storageSize, *valuesIn, tmpValuesOut, numItems);

#ifdef MEASURE_EXECUTION_TIMES
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
#endif

	// Free temporary memory
	cudaFree(tempStorage);

	if (valuesIn == valuesOut) {
		cudaFree(*valuesIn);
		*valuesIn = tmpValuesOut;
	}

	return elapsedTime;
}

// ref. https://nvlabs.github.io/cub/structcub_1_1_device_scan.html#a83236fc272c0b573a2bb2c5b47e0867d
template <typename T> float DeviceInclusiveSum(int numItems, T** valuesIn, T** valuesOut)
{
	T* tmpValuesOut;
	if (valuesIn == valuesOut) {
		cudaMalloc((void **)&tmpValuesOut, numItems * sizeof(T));
	}
	else {
		tmpValuesOut = *valuesOut;
	}

	// Check how much temporary memory will be required
	void* tempStorage = nullptr;
	size_t storageSize = 0;
	cub::DeviceScan::InclusiveSum(tempStorage, storageSize, *valuesIn, tmpValuesOut, numItems);

	// Allocate temporary memory
	cudaMalloc(&tempStorage, storageSize);

	float elapsedTime = 0.0f;
#ifdef MEASURE_EXECUTION_TIMES
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
#endif

	// Scan
	cub::DeviceScan::InclusiveSum(tempStorage, storageSize, *valuesIn, tmpValuesOut, numItems);

#ifdef MEASURE_EXECUTION_TIMES
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
#endif

	// Free temporary memory
	cudaFree(tempStorage);

	if (valuesIn == valuesOut) {
		cudaFree(*valuesIn);
		*valuesIn = tmpValuesOut;
	}

	return elapsedTime;
}


float DeviceExclusiveSum(int numElements, int** valuesIn, int** valuesOut)
{
	return DeviceExclusiveSum<int>(numElements, valuesIn, valuesOut);
}

float DeviceInclusiveSum(int numElements, int** valuesIn, int** valuesOut)
{
	return DeviceInclusiveSum<int>(numElements, valuesIn, valuesOut);
}

}}
