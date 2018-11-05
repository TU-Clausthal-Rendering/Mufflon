#pragma once

#include "lights.hpp"
#include "core/scene/types.hpp"

namespace mufflon { namespace scene { namespace lights {

class LightTree {
public:

//private:
	/**
	 * Node of the light tree. Stores the accumulated flux and position of its children.
	 * Additionally, it stores whether the children are leafs or interior nodes
	 * and, depending on that information, either the leaf's type or the
	 * index of the node.
	 */
#pragma pack(push, 1)
	struct alignas(16) Node {

		// Struct holding either a light type or an array index
		class Child {
		public:
			__host__ __device__ Child() = default;

			__host__ __device__ constexpr bool is_leaf() const noexcept {
				return m_data & (1 << 31);
			}

			__host__ __device__ constexpr LightType get_type() const noexcept {
				return static_cast<LightType>(m_data & ~(1 << 31));
			}
			__host__ __device__ void set_type(LightType type) noexcept {
				mAssert(is_leaf());
				m_data = (1 << 31) | static_cast<u32>(type);
			}

			__host__ __device__ constexpr u32 get_index() const noexcept {
				mAssert(!is_leaf());
				return m_data & ~(1 << 31);
			}
			__host__ __device__ void set_index(u32 index) noexcept {
				mAssert(!(index & (1 << 31)));
				m_data = index;
			}

		private:
			u32 m_data;
		};

		ei::Vec3 flux;
		Child left;
		ei::Vec3 position;
		Child right;
	};
#pragma pack(pop)
	static_assert(sizeof(Node) == 32);

	// Helper struct to enable synchronization


	util::DirtyFlags<Device> m_flags;
	util::TaggedTuple<DeviceArrayHandle<Device::CPU, Node>,
		DeviceArrayHandle<Device::CUDA, Node>> m_tree;
};

}}} // namespace mufflon::scene::lights