#pragma once

#include "util/assert.hpp"
#include "util/int_types.hpp"
#include "core/export/core_api.h"
#include <cuda_runtime.h>

namespace mufflon { namespace renderer { namespace decimaters {

struct FloatOctreeNode {
	float data = 0.f;

	// Creates a node that is pointing to other nodes as a parent
	CUDA_FUNCTION static FloatOctreeNode as_parent(const u32 offset) noexcept {
		return FloatOctreeNode{ -static_cast<float>(offset) };
	}
	CUDA_FUNCTION static FloatOctreeNode as_split_child(const float initViewCum) noexcept {
		return FloatOctreeNode{ initViewCum };
	}

	CUDA_FUNCTION bool is_parent() const noexcept { return data < 0.f; }
	CUDA_FUNCTION bool is_leaf() const noexcept { return !is_parent(); }

	// This function is purely for the spinlock functionality in case
	// the capacity limit has been reached
	CUDA_FUNCTION bool is_parent_or_fresh() const noexcept { return data <= 0.f; }

	CUDA_FUNCTION u32 get_child_offset() const noexcept {
		mAssert(this->is_parent());
		return static_cast<u32>(-data);
	}
	CUDA_FUNCTION float get_value() const noexcept {
		mAssert(this->is_leaf());
		return data;
	}
	CUDA_FUNCTION float get_sample() const noexcept {
		mAssert(this->is_leaf());
		return data;
	}
};

struct SampleOctreeNode {
	i32 count = 0;
	float data = 0.f;

	// Creates a node that is pointing to other nodes as a parent
	CUDA_FUNCTION static SampleOctreeNode as_parent(const u32 offset) noexcept {
		return SampleOctreeNode{ -static_cast<i32>(offset), 0.f };
	}
	CUDA_FUNCTION static SampleOctreeNode as_split_child(const u32 initCount, const float initValue) noexcept {
		return SampleOctreeNode{ static_cast<i32>(initCount), initValue };
	}

	CUDA_FUNCTION bool is_parent() const noexcept { return count < 0; }
	CUDA_FUNCTION bool is_leaf() const noexcept { return !is_parent(); }

	// This function is purely for the spinlock functionality in case
	// the capacity limit has been reached
	CUDA_FUNCTION bool is_parent_or_fresh() const noexcept { return count <= 0; }

	CUDA_FUNCTION u32 get_child_offset() const noexcept {
		mAssert(this->is_parent());
		return static_cast<u32>(-count);
	}
	CUDA_FUNCTION u32 get_count() const noexcept {
		mAssert(this->is_leaf());
		return count;
	}
	CUDA_FUNCTION float get_value() const noexcept {
		mAssert(this->is_leaf());
		return data;
	}
	CUDA_FUNCTION float get_sample() const noexcept {
		mAssert(this->is_leaf());
		return data / static_cast<float>(ei::max(1, count));
	}
};

}}} // namespace mufflon::renderer::decimaters