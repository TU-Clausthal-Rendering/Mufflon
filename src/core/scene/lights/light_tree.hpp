#pragma once

#include "lights.hpp"
#include "export/dll_export.hpp"
#include "core/memory/allocator.hpp"
#include "core/memory/residency.hpp"
#include "core/memory/synchronize.hpp"
#include "core/scene/types.hpp"
#include "core/scene/textures/texture.hpp"
#include <optional>

// Forward declaration
namespace ei {
struct Box;
} // namespace ei

namespace mufflon { namespace scene { namespace lights {

class LIBRARY_API LightTree {
public:
	LightTree();
	~LightTree();

	/**
	 * Node of the light tree. Stores the accumulated flux and position of its children.
	 * Additionally, it stores whether the children are leaves or interior nodes
	 * and, depending on that information, either the leaf's type or the
	 * index of the node.
	 */
#pragma pack(push, 1)
	// Struct holding either a light type or an array index
	class Child {
	public:
		__host__ __device__ Child() = default;

		__host__ __device__ constexpr bool is_leaf() const noexcept {
			return m_data & (1 << 31);
		}

		__host__ __device__ constexpr LightType get_type() const noexcept {
			mAssert(is_leaf());
			return static_cast<LightType>(m_data & ~(1 << 31));
		}
		__host__ __device__ void set_type(LightType type) noexcept {
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

	// Node for positional lights
	struct alignas(16) PosNode {
		ei::Vec3 intensity;
		Child left;
		ei::Vec3 position;
		Child right;
	};
	// Node for directional lights
	struct alignas(16) DirNode {
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

		LightType singlePosLightType; // If there's only one positional light, this gives us the type
		EnvMapLight<dev> envLight;
		// Pointer to the tree elements
		struct DirLightTree {
			std::size_t lightCount;
			DirNode* nodes;
			char* lights;
		} dirLights;
		struct PosLightTree {
			std::size_t lightCount;
			PosNode* nodes;
			char* lights;
		} posLights;
		// Actual memory
		std::size_t length;
		DeviceArrayHandle<DEVICE, char> memory;
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
		Tree<dev>& tree = m_trees.get<Tree<dev>>();
		tree.memory.handle = Allocator<dev>::free(tree.memory.handle, tree.length);
		// TODO: unload envmap handle
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
__host__ __device__ Photon emit(const LightTree::PosNode* nodes, u64 rng) {
	// Traverse the tree

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
