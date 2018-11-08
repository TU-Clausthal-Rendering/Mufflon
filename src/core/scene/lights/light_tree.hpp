#pragma once

#include "lights.hpp"
#include "light_sampling.hpp"
#include "export/api.hpp"
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

#pragma pack(push, 1)
	/**
	 * Child pointer of light tree node. Stores whether a leaf follows and,
	 * depending on that, additional data identifying either the next node
	 * or light.
	 */
	class Child {
	public:
		// Ensure that our largest light-type doesn't exceed the number of bits
		// a light type may have in the child pointer
		static_assert(static_cast<u16>(LightType::NUM_LIGHTS) < std::numeric_limits<u16>::max());

		__host__ __device__ Child() = default;
		__host__ __device__ constexpr bool is_leaf() const noexcept {
			return m_type != INVALID_TYPE;
		}
		__host__ __device__ void mark_node(std::size_t index)  noexcept {
			mAssert(index <= std::numeric_limits<u32>::max());
			m_type = INVALID_TYPE;
			m_offset = static_cast<u32>(index);
		}
		__host__ __device__ void mark_leaf(LightType type, u32 offset) {
			m_type = static_cast<u16>(type);
			m_offset = offset;
		}
		__host__ __device__ constexpr LightType get_type() const noexcept {
			mAssert(is_leaf());
			return static_cast<LightType>(m_type);
		}
		__host__ __device__ constexpr u64 get_offset() const noexcept {
			return m_offset;
		}

	private:
		static constexpr u16 INVALID_TYPE = std::numeric_limits<u16>::max();

		u32 m_offset;
		u16 m_type;
	};
	
	struct alignas(16) Node {
		static constexpr u16 INVALID_TYPE = std::numeric_limits<u16>::max();

		Node(const Node& left, const Node& right);
		Node(const Node& left, const PositionalLights& right,
			 const ei::Vec3& bounds);
		Node(const Node& left, const DirectionalLight& right,
			 const ei::Vec3& bounds);
		Node(const PositionalLights& left, const Node& right,
			 const ei::Vec3& bounds);
		Node(const DirectionalLight& left, const Node& right,
			 const ei::Vec3& bounds);
		Node(const PositionalLights& left, const PositionalLights& right,
			 const ei::Vec3& bounds);
		Node(const DirectionalLight& left, const DirectionalLight& right,
			 const ei::Vec3& bounds);

		// Layout: [4,4,2]=10, [2,4,4]=10, [4,4,4]=12 bytes
		// Necessary duplication due to memory layout (2x32+16 and 16+2x32 bits)
		struct {
			__device__ __host__ void set_offset(std::size_t off) noexcept {
				mAssert(off < std::numeric_limits<u32>::max());
				offset = static_cast<u32>(off);
			}

			float flux;
			u32 offset;
			u16 type;
		} left;
		struct {
			__device__ __host__ void set_offset(std::size_t off) noexcept {
				mAssert(off < std::numeric_limits<u32>::max());
				offset = static_cast<u32>(off);
			}

			u16 type;
			u32 offset;
			float flux;
		} right;

		ei::Vec3 center;
	};
#pragma pack(pop)
	static_assert(sizeof(Node) == 32);

	struct LightTypeTree {
		struct {
			float flux;
			u16 type;
		} root;
		std::size_t lightCount;
		Node* nodes;
		char* lights;
	};

	template < Device dev >
	struct Tree {
		static constexpr Device DEVICE = dev;

		EnvMapLight<dev> envLight;
		// Pointer to the tree elements
		LightTypeTree dirLights;
		LightTypeTree posLights;
		// Actual memory
		std::size_t length;
		DeviceArrayHandle<DEVICE, char> memory;
	};

	void build(std::vector<PositionalLights>&& posLights,
			   std::vector<DirectionalLight>&& dirLights,
			   const ei::Box& boundingBox,
			   std::optional<textures::TextureHandle> envLight = std::nullopt);

	template < Device dev >
	const Tree<dev>& aquire_tree() noexcept {
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

// TODO: extract
namespace lighttree_detail {

// Converts the typeless memory into the given light type and samples it
__host__ __device__ Photon sample_light(LightType type, const char* light,
										const ei::Box& bounds,
										const RndSet& rnd) {
	mAssert(static_cast<u16>(type) < static_cast<u16>(LightType::NUM_LIGHTS));
	switch(type) {
		case LightType::POINT_LIGHT: return sample_light(*reinterpret_cast<const PointLight*>(light), rnd);
		case LightType::SPOT_LIGHT: return sample_light(*reinterpret_cast<const SpotLight*>(light), rnd);
		case LightType::AREA_LIGHT_TRIANGLE: return sample_light(*reinterpret_cast<const AreaLightTriangle*>(light), rnd);
		case LightType::AREA_LIGHT_QUAD: return sample_light(*reinterpret_cast<const AreaLightQuad*>(light), rnd);
		case LightType::AREA_LIGHT_SPHERE: return sample_light(*reinterpret_cast<const AreaLightSphere*>(light), rnd);
		case LightType::DIRECTIONAL_LIGHT: return sample_light(*reinterpret_cast<const DirectionalLight*>(light), bounds, rnd);
		default: mAssert(false); return {};
	}
}

} // namespace lighttree_detail

/** Shared code for emitting a single photon from the tree.
 * Takes the light tree, initial interval limits, and RNG number as inputs.
 */
__host__ __device__ Photon emit(const LightTree::LightTypeTree& tree, u64 left, u64 right,
								u64 rng, const ei::Box& bounds, const RndSet& rnd) {
	// Check: do we have more than one light here?
	if(tree.lightCount == 1u) {
		// Nothing to do but sample the photon
		mAssert(tree.root.type < static_cast<u16>(LightType::NUM_LIGHTS));
		return lighttree_detail::sample_light(static_cast<LightType>(tree.root.type),
											  &tree.lights[0u], bounds, rnd);
	}

	// Traverse the tree to split chance between lights
	const LightTree::Node* currentNode = tree.nodes;
	u16 type = LightTree::Node::INVALID_TYPE;
	u32 offset = 0u;
	/**
	 * Initial interval
	 * Top level [0, leftFlux0*MAX_INT, MAX_INT]
	 * Next: left [0, leftFlux1l*leftFlux0*MAX_INT, leftFlux0*MAX_INT]
	 *		 right [leftFlux0*MAX_INT, leftFlux1r*leftFlux0*MAX_INT, MAX_INT]
	 * And so on...
	 */
	u64 intervalLeft = left;
	u64 intervalRight = right;

	// Iterate until we hit a leaf
	while(type == LightTree::Node::INVALID_TYPE) {
		mAssert(currentNode != nullptr);
		mAssert(intervalLeft <= intervalRight);
		mAssert(rng >= intervalLeft && rng <= intervalRight);
		// Scale the flux up
		float fluxSum = currentNode->left.flux + currentNode->right.flux;
		u64 split = static_cast<u64>(intervalLeft + (intervalRight - intervalLeft) * currentNode->left.flux / fluxSum);

		// Decide what path to take and recompute RNG interval by rescaling the interval
		if(rng < split) {
			if(currentNode->left.type != LightTree::Node::INVALID_TYPE) {
				type = currentNode->left.type;
				offset = currentNode->left.offset;
				break;
			}
			currentNode = &tree.nodes[currentNode->left.offset];
			intervalRight = split;
		} else {
			if(currentNode->right.type != LightTree::Node::INVALID_TYPE) {
				type = currentNode->right.type;
				offset = currentNode->right.offset;
				break;
			}
			currentNode = &tree.nodes[currentNode->right.offset];
			intervalLeft = split;
		}
	}

	mAssert(type != LightTree::Node::INVALID_TYPE);
	// We got a light source! Sample it
	return lighttree_detail::sample_light(static_cast<LightType>(type),
										  &tree.lights[offset], bounds, rnd);
}

__host__ __device__ Photon emit(const LightTree::LightTypeTree& tree, u64 left, u64 right,
								u64 index, u64 rng, const ei::Box& bounds,
								const RndSet& rnd) {
	// Check: do we have more than one light here?
	if(tree.lightCount == 1u) {
		// Nothing to do but sample the photon
		mAssert(tree.root.type < static_cast<u16>(LightType::NUM_LIGHTS));
		return lighttree_detail::sample_light(static_cast<LightType>(tree.root.type),
											  &tree.lights[0u], bounds, rnd);
	}

	// Traverse the tree to split chance between lights
	const LightTree::Node* currentNode = tree.nodes;
	u16 type = LightTree::Node::INVALID_TYPE;
	u32 offset = 0u;
	/**
	 * Initial interval
	 * Top level [0, leftFlux0*MAX_INT, MAX_INT]
	 * Next: left [0, leftFlux1l*leftFlux0*MAX_INT, leftFlux0*MAX_INT]
	 *		 right [leftFlux0*MAX_INT, leftFlux1r*leftFlux0*MAX_INT, MAX_INT]
	 * And so on...
	 */
	u64 intervalLeft = left;
	u64 intervalRight = right;

	// Iterate until we hit a leaf
	while(type == LightTree::Node::INVALID_TYPE) {
		mAssert(currentNode != nullptr);
		mAssert(intervalLeft <= intervalRight);

		// Scale the flux up
		float fluxSum = currentNode->left.flux + currentNode->right.flux;
		// Compute the integer bounds: once rounded down, once rounded up
		u64 leftRight = static_cast<u64>(intervalLeft + (intervalRight - intervalLeft)
										 * currentNode->left.flux / fluxSum);
		u64 rightLeft = static_cast<u64>(std::ceilf(intervalLeft + (intervalRight - intervalLeft)
													* currentNode->left.flux / fluxSum));
		// Check if our index falls into one of these
		if(index < leftRight) {
			if(currentNode->left.type != LightTree::Node::INVALID_TYPE) {
				type = currentNode->left.type;
				offset = currentNode->left.offset;
				break;
			}
			currentNode = &tree.nodes[currentNode->left.offset];
			intervalRight = leftRight;
		} else if(index >= rightLeft) {
			if(currentNode->right.type != LightTree::Node::INVALID_TYPE) {
				type = currentNode->right.type;
				offset = currentNode->right.offset;
				break;
			}
			currentNode = &tree.nodes[currentNode->right.offset];
			intervalLeft = rightLeft;
		} else {
			// In the middle: gotta let RNG decide this one
			intervalLeft = 0u;
			intervalRight = std::numeric_limits<u64>::max();
			std::size_t split = static_cast<u64>(intervalLeft + (intervalRight - intervalLeft)
												 * currentNode->left.flux / fluxSum);
			if(rng < split) {
				if(currentNode->left.type != LightTree::Node::INVALID_TYPE) {
					type = currentNode->left.type;
					offset = currentNode->left.offset;
					break;
				}
				currentNode = &tree.nodes[currentNode->left.offset];
				intervalRight = split;
			} else {
				if(currentNode->right.type != LightTree::Node::INVALID_TYPE) {
					type = currentNode->right.type;
					offset = currentNode->right.offset;
					break;
				}
				currentNode = &tree.nodes[currentNode->right.offset];
				intervalLeft = split;
			}

			// Make sure that for following iterations, we use index as well
			index = rng;
		}
	}

	mAssert(type != LightTree::Node::INVALID_TYPE);
	// We got a light source! Sample it
	return lighttree_detail::sample_light(static_cast<LightType>(type),
										  &tree.lights[offset], bounds, rnd);
}

 // Emits a single photon from a light source.
template < Device dev >
__host__ __device__ Photon emit(const LightTree::Tree<dev>& tree, u64 rng,
								const ei::Box& bounds, const RndSet& rnd) {
	// Figure out which of the three top-level light types get the photon
	// Implicit left boundary of 0 for the interval
	u64 intervalRight = std::numeric_limits<u64>::max();

	float fluxSum = tree.dirLights.root.flux + tree.posLights.root.flux;
	// TODO: way to check handle's validity!
	if(tree.envLight.texHandle.handle != nullptr)
		fluxSum += ei::sum(tree.envLight.flux);

	// Now split up based on flux
	// First is envmap...
	u64 splitEnv = 0u;
	if(tree.envLight.texHandle.handle != nullptr)
		splitEnv = static_cast<u64>(intervalRight * ei::sum(tree.envLight.flux) / fluxSum);
	if(rng < splitEnv)
		return sample_light(tree.envLight, rnd);
	// ...then come directional lights...
	u64 splitDir = splitEnv + static_cast<u64>(intervalRight * tree.dirLights.root.flux / fluxSum);
	if(rng < splitDir) {
		mAssert(tree.dirLights.lightCount > 0u);
		return emit(tree.dirLights, splitEnv, splitDir, rng, bounds, rnd);
	}
	// ...and last positional lights
	mAssert(tree.posLights.lightCount > 0u);
	return emit(tree.posLights, splitDir, intervalRight, rng, bounds, rnd);
}

/**
 * Emits a single photon from a light source.
 * To ensure a good distribution, we also take an index, which is used to guide
 * the descent into the tree when it is possible to do so without using RNG.
 */
template < Device dev >
__host__ __device__ Photon emit(const LightTree::Tree<dev>& tree, u64 index, u64 indexMax,
								u64 rng, const ei::Box& bounds, const RndSet& rnd) {
	// Figure out which of the three top-level light types get the photon
	// Implicit left boundary of 0 for the interval
	u64 intervalRight = indexMax;

	float fluxSum = tree.dirLights.root.flux + tree.posLights.root.flux;
	// TODO: way to check handle's validity!
	if(tree.envLight.texHandle.handle != nullptr)
		fluxSum += ei::sum(tree.envLight.flux);

	// Now split up based on flux
	// First is envmap...
	u64 rightEnv = static_cast<u64>(intervalRight * ei::sum(tree.envLight.flux) / fluxSum);
	if(index < rightEnv) {
		mAssert(tree.envLight.texHandle.handle != nullptr);
		return sample_light(tree.envLight, rnd);
	}
	// ...then come directional lights...
	u64 leftDir = static_cast<u64>(std::ceilf(intervalRight * (ei::sum(tree.envLight.flux) / fluxSum)));
	u64 rightDir = static_cast<u64>(intervalRight * (ei::sum(tree.envLight.flux)
													 + tree.dirLights.root.flux) / fluxSum);
	if(index >= leftDir && index < rightDir) {
		mAssert(tree.dirLights.lightCount > 0u);
		return emit(tree.dirLights, leftDir, rightDir, index, rng, bounds, rnd);
	}
	// ...and last positional lights
	u64 leftPos = static_cast<u64>(std::ceilf(intervalRight * (ei::sum(tree.envLight.flux)
															   + tree.dirLights.root.flux / fluxSum)));
	if(index >= leftPos) {
		mAssert(tree.posLights.lightCount > 0u);
		return emit(tree.posLights, leftPos, intervalRight,
					index, rng, bounds, rnd);
	}

	// If we made it until here, it means that we fell between the integer bounds of photon distribution
	// Thus we need RNG to decide
	u64 splitEnv = static_cast<u64>(std::numeric_limits<u64>::max() * ei::sum(tree.envLight.flux) / fluxSum);
	u64 splitDir = static_cast<u64>(std::numeric_limits<u64>::max() * (ei::sum(tree.envLight.flux)
																	   + tree.dirLights.root.flux) / fluxSum);
	if(rng < splitEnv) {
		mAssert(tree.envLight.texHandle.handle != nullptr);
		return sample_light(tree.envLight, rnd);
	} else if(rng < splitDir) {
		mAssert(tree.dirLights.lightCount > 0u);
		return emit(tree.dirLights, splitEnv, splitDir, rng, bounds, rnd);
	} else {
		mAssert(tree.posLights.lightCount > 0u);
		return emit(tree.posLights, splitDir, std::numeric_limits<u64>::max(),
					rng, bounds, rnd);
	}
}

}}} // namespace mufflon::scene::lights
