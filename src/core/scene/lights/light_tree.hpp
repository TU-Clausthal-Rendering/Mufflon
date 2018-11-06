#pragma once

#include "lights.hpp"
#include "core/scene/residency.hpp"
#include "core/scene/synchronize.hpp"
#include "core/scene/types.hpp"
#include "core/scene/textures/texture.hpp"
#include <optional>

// Forward declaration
namespace ei {
struct Box;
} // namespace ei

namespace mufflon { namespace scene { namespace lights {

class LightTree {
public:
	/**
	 * Node of the light tree. Stores the accumulated flux and position of its children.
	 * Additionally, it stores whether the children are leaves or interior nodes
	 * and, depending on that information, either the leaf's type or the
	 * index of the node.
	 */
#pragma pack(push, 1)
	// Node type: either a light-tree node or a light
	struct alignas(16) PosNode {
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

		ei::Vec3 intensity;
		Child left;
		ei::Vec3 position;
		Child right;
	};

	struct alignas(16) DirNode {
		// Struct holding an index minus one bit to tell us if a leaf follows
		class Child {
		public:
			__host__ __device__ Child() = default;

			__host__ __device__ constexpr bool is_leaf() const noexcept {
				return m_data & (1 << 31);
			}

			__host__ __device__ void make_leaf() noexcept {
				m_data |= (1 << 31);
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

		ei::Vec3 intensity;
		Child left;
		ei::Vec3 direction;
		Child right;
	};
#pragma pack(pop)
	static_assert(sizeof(PosNode) == 32);

	template < Device dev >
	struct Tree {
		static constexpr Device DEVICE = dev;

		std::size_t numDirLights;
		std::size_t numPosLights;
		LightType singlePosLightType; // If there's only one positional light, this gives us the type
		EnvMapLight<dev> envLight;
		DeviceArrayHandle<DEVICE, char> dirLights;
		DeviceArrayHandle<DEVICE, char> posLights;
	};

	void build(std::vector<PositionalLights>&& posLights,
			   std::vector<DirectionalLight>&& dirLights,
			   const ei::Box& boundingBox,
			   std::optional<textures::TextureHandle> envLight = std::nullopt);

	template < Device dev >
	const Tree<dev>& aquire_tree() const noexcept {
		this->synchronize<dev>();
		return m_trees.get<Tree<dev>>();
	}

	template < Device dev >
	void synchronize() {
		//m_envMapTexture.synchronize<dev>();
		//mufflon::scene::synchronize<dev>(m_trees, m_flags, m_trees.get<Tree<dev>>(), m_envMapTexture);
	}

	template < Device dev >
	void unload() {
		// TODO: m_envMapTexture.unload();
		unload(m_trees.get<Tree<dev>>());
	}

private:
	textures::TextureHandle m_envMapTexture;
	util::DirtyFlags<Device> m_flags;
	util::TaggedTuple<Tree<Device::CPU>, Tree<Device::CUDA>> m_trees;
};

// Functions for synchronizing a light tree
void synchronize(const LightTree::Tree<Device::CPU>& changed, LightTree::Tree<Device::CUDA>& sync, textures::TextureHandle hdl);
void synchronize(const LightTree::Tree<Device::CUDA>& changed, LightTree::Tree<Device::CPU>& sync, textures::TextureHandle hdl);
void unload(LightTree::Tree<Device::CPU>& tree);
void unload(LightTree::Tree<Device::CUDA>& tree);

// TODO: proper type
struct Photon {
	ei::Vec3 position;
	AreaPdf posPdf;
	ei::Vec3 direction;
	AngularPdf dirPdf;
};

struct NextEventEstimation {
	ei::Vec3 position;
	AreaPdf posPdf;
	ei::Vec3 intensity;
	AngularPdf dirPdf;
};

// Shared code for emitting a single photon from the tree
__host__ __device__ Photon emit(const LightTree::PosNode* nodes) {
	// TODO: traverse tree
	return {};
}
__host__ __device__ Photon emit(const LightTree::DirNode* nodes) {
	// TODO: traverse tree
	return {};
}
// Shared code for emitting multiple photons from the tree
__host__ __device__ void emit(const LightTree::PosNode* nodes, Photon* photons, std::size_t n) {
	// TODO: traverse tree
}
__host__ __device__ void emit(const LightTree::DirNode* nodes, Photon* photons, std::size_t n) {
	// TODO: traverse tree
}

// Emit a single photon from an environment-map light source
template < Device dev >
__host__ __device__ Photon emit(const EnvMapLight<dev>& envLight) {
	// TODO: sample envmap
	return {};
}
// Emit a single photon from an environment-map light source
template < Device dev >
__host__ __device__ void emit(const EnvMapLight<dev>& envLight, Photon* photons, std::size_t n) {
	// TODO: sample envmap
	return {};
}

 // Emits a single photon from a light source.
template < Device dev >
__host__ __device__ Photon emit(const LightTree::Tree<dev>& tree) {
	// TODO
	return {};
}
//  Emits a number of photons from the light tree.
template < Device dev >
__host__ __device__ void emit(const LightTree::Tree<dev>& tree, Photon* photons, std::size_t n) {
	// TODO
	return {};
}

}}} // namespace mufflon::scene::lights