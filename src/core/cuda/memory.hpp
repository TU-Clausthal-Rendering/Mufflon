#pragma once

#include <cuda_runtime.h>
#include "error.hpp"
#include <memory>

namespace mufflon::cuda {

// TODO: Better version that uses pool allocation for memory continuity
template < class T >
class AttributeAllocator {
public:
	using ValueType = T;
	using SizeType = std::size_t;
	using DifferenceType = std::ptrdiff_t;

	AttributeAllocator() = default;
	AttributeAllocator(const AttributeAllocator&) = default;
	AttributeAllocator(AttributeAllocator&&) = default;
	AttributeAllocator& operator=(const AttributeAllocator&) = default;
	AttributeAllocator& operator=(AttributeAllocator&&) = default;
	virtual ~AttributeAllocator() = default;

	[[nodiscard]] ValueType* allocate(std::size_t n) {
		ValueType* ptr = nullptr;
		check_error(cudaMalloc(&ptr, sizeof(ValueType) * n));
		return ptr;
	}

	void deallocate(ValueType* ptr, std::size_t n) {
		(void)n;
		check_error(cudaFree(ptr));
	}

	bool operator==(const AttributeAllocator& lhs, const AttributeAllocator& rhs) {
		return true;
	}

	bool operator!=(const AttributeAllocator& lhs, const AttributeAllocator& rhs) {
		return !(this->operator==(lhs, rhs));
	}

private:
};


} // namespace mufflon::cuda 