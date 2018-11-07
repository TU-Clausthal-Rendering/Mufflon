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
			 const ei::Vec3& aabbDiag);
		Node(const Node& left, const DirectionalLight& right,
			 const ei::Vec3& aabbDiag);
		Node(const PositionalLights& left, const Node& right,
			 const ei::Vec3& aabbDiag);
		Node(const DirectionalLight& left, const Node& right,
			 const ei::Vec3& aabbDiag);
		Node(const PositionalLights& left, const PositionalLights& right,
			 const ei::Vec3& aabbDiag);
		Node(const DirectionalLight& left, const DirectionalLight& right,
			 const ei::Vec3& aabbDiag);

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

// TODO: extract
namespace lighttree_detail {

// TODO
// Sample a light source for either one or many photons
__host__ __device__ inline Photon sample_light(const PointLight& light,
											   float r0, float r1) {
	return {};
}
__host__ __device__ inline Photon sample_light(const SpotLight& light,
											   float r0, float r1) {
	return {};
}
__host__ __device__ inline Photon sample_light(const AreaLightTriangle& light,
											   float r0, float r1) {
	return {};
}
__host__ __device__ inline Photon sample_light(const AreaLightQuad& light,
											   float r0, float r1) {
	return {};
}
__host__ __device__ inline Photon sample_light(const AreaLightSphere& light,
											   float r0, float r1) {
	return {};
}
__host__ __device__ inline Photon sample_light(const DirectionalLight& light,
											   float r0, float r1) {
	return {};
}
template < Device dev >
__host__ __device__ inline Photon sample_light(const EnvMapLight<dev>& light,
											   float r0, float r1) {
	return {};
}
__host__ __device__ inline void sample_light(const PointLight& light,
											 Photon* photons, std::size_t n,
											 const float* photonRng) {}
__host__ __device__ inline void sample_light(const SpotLight& light,
											 Photon* photons, std::size_t n,
											 const float* photonRng) {}
__host__ __device__ inline void sample_light(const AreaLightTriangle& light,
											 Photon* photons, std::size_t n,
											 const float* photonRng) {}
__host__ __device__ inline void sample_light(const AreaLightQuad& light,
											 Photon* photons, std::size_t n,
											 const float* photonRng) {}
__host__ __device__ inline void sample_light(const AreaLightSphere& light,
											 Photon* photons, std::size_t n,
											 const float* photonRng) {}
__host__ __device__ inline void sample_light(const DirectionalLight& light,
											 Photon* photons, std::size_t n,
											 const float* photonRng) {}
template < Device dev >
__host__ __device__ inline void sample_light(const EnvMapLight<dev>& light,
											   Photon* photons, std::size_t n,
											 const float* photonRng) {}

// Converts the typeless memory into the given light type and samples it
__host__ __device__ Photon sample_light(LightType type, const char* light,
										float r0, float r1) {
	mAssert(static_cast<u16>(type) < static_cast<u16>(LightType::NUM_LIGHTS));
	switch(type) {
		case LightType::POINT_LIGHT: return sample_light(*reinterpret_cast<const PointLight*>(light), r0, r1);
		case LightType::SPOT_LIGHT: return sample_light(*reinterpret_cast<const SpotLight*>(light), r0, r1);
		case LightType::AREA_LIGHT_TRIANGLE: return sample_light(*reinterpret_cast<const AreaLightTriangle*>(light), r0, r1);
		case LightType::AREA_LIGHT_QUAD: return sample_light(*reinterpret_cast<const AreaLightQuad*>(light), r0, r1);
		case LightType::AREA_LIGHT_SPHERE: return sample_light(*reinterpret_cast<const AreaLightSphere*>(light), r0, r1);
		case LightType::DIRECTIONAL_LIGHT: return sample_light(*reinterpret_cast<const DirectionalLight*>(light), r0, r1);
		default: mAssert(false); return {};
	}
}
__host__ __device__ void sample_light(LightType type, const char* light, Photon* photons, std::size_t n,
									  const float* photonRng) {
	mAssert(static_cast<u16>(type) < static_cast<u16>(LightType::NUM_LIGHTS));
	switch(type) {
		case LightType::POINT_LIGHT: sample_light(*reinterpret_cast<const PointLight*>(light),
												  photons, n, photonRng);
		case LightType::SPOT_LIGHT: sample_light(*reinterpret_cast<const SpotLight*>(light),
												 photons, n, photonRng);;
		case LightType::AREA_LIGHT_TRIANGLE: sample_light(*reinterpret_cast<const AreaLightTriangle*>(light),
														  photons, n, photonRng);
		case LightType::AREA_LIGHT_QUAD: sample_light(*reinterpret_cast<const AreaLightQuad*>(light),
													  photons, n, photonRng);
		case LightType::AREA_LIGHT_SPHERE: sample_light(*reinterpret_cast<const AreaLightSphere*>(light),
														photons, n, photonRng);
		case LightType::DIRECTIONAL_LIGHT: sample_light(*reinterpret_cast<const DirectionalLight*>(light),
														photons, n, photonRng);
		default: mAssert(false);
	}
}

} // namespace lighttree_detail

/** Shared code for emitting a single photon from the tree.
 * Takes the light tree, initial interval limits, and RNG number as inputs.
 */
__host__ __device__ Photon emit(const LightTree::LightTypeTree& tree, u64 left, u64 right,
								u64 rng, float r0, float r1) {
	// Check: do we have more than one light here?
	if(tree.lightCount == 1u) {
		// Nothing to do but sample the photon
		mAssert(tree.root.type < static_cast<u16>(LightType::NUM_LIGHTS));
		return lighttree_detail::sample_light(static_cast<LightType>(tree.root.type),
											  &tree.lights[0u], r0, r1);
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
										  &tree.lights[offset], r0, r1);
}
/**
 * Shared code for emitting multiple photons from the tree.
 * Takes the light tree, photon buffer, number of photons,
 * initial interval limits, and RNG number as inputs.
 */
__host__ __device__ void emit(const LightTree::LightTypeTree& tree, Photon* photons,
							  std::size_t n, u64 left, u64 right, u64 rng,
							  const float* photonRng) {
	// TODO: traverse tree
}

 // Emits a single photon from a light source.
template < Device dev >
__host__ __device__ Photon emit(const LightTree::Tree<dev>& tree, u64 rng, float r0, float r1) {
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
		return lighttree_detail::sample_light(tree.envLight, r0, r1);
	// ...then come directional lights...
	u64 splitDir = splitEnv + static_cast<u64>(intervalRight * tree.dirLights.root.flux / fluxSum);
	if(rng < splitDir) {
		mAssert(tree.dirLights.lightCount > 0u);
		return emit(tree.dirLights, splitEnv, splitDir, rng, r0, r1);
	}
	// ...and last positional lights
	mAssert(tree.posLights.lightCount > 0u);
	return emit(tree.posLights, splitDir, intervalRight, rng, r0, r1);
}
//  Emits a number of photons from the light tree.
template < Device dev >
__host__ __device__ void emit(const LightTree::Tree<dev>& tree, Photon* photons,
							  std::size_t n, u64 rng, const float* photonRng) {
	// TODO: Device-agnostic type for photon buffer!
	mAssert(photons != nullptr);

	// Figure out how to split the n photons onto the top-level light types
	// Implicit left boundary of 0 for the interval
	u64 intervalRight = std::numeric_limits<u64>::max();

	float fluxSum = tree.dirLights.root.flux + tree.posLights.root.flux;
	// TODO: way to check handle's validity!
	if(tree.envLight.texHandle.handle != nullptr)
		fluxSum += ei::sum(tree.envLight.flux);

	// Now split up based on flux
	// First is envmap...
	u64 splitEnv = splitEnv = static_cast<u64>(intervalRight * ei::sum(tree.envLight.flux) / fluxSum);
	u64 envPhotons = static_cast<std::size_t>(n * ei::sum(tree.envLight.flux) / fluxSum);
	// ...then come directional lights...
	u64 splitDir = splitEnv + static_cast<u64>(intervalRight * tree.dirLights.root.flux / fluxSum);
	u64 dirPhotons = static_cast<std::size_t>(n * tree.dirLights.root.flux / fluxSum);
	// ...and last positional lights
	u64 posPhotons = static_cast<std::size_t>(n * tree.posLights.root.flux / fluxSum);

	// Now find out where to put the (up to 1) extra photon
	if(envPhotons + dirPhotons + posPhotons < n) {
		mAssert(n - (envPhotons + dirPhotons + posPhotons) <= 1u);
		if(rng < splitEnv)
			++envPhotons;
		else if(rng < splitDir)
			++dirPhotons;
		else
			++posPhotons;
	}

	// Sample the photons from the respective light sources
	if(envPhotons > 0u && tree.envLight.texHandle.handle != nullptr)
		lighttree_detail::sample_light(tree.envLight, photons, envPhotons, photonRng);
	if(dirPhotons > 0u) {
		mAssert(tree.dirLights.lightCount > 0u);
		emit(tree.dirLights, &photons[envPhotons], dirPhotons, splitEnv, splitDir,
			 rng, &photonRng[envPhotons]);
	}
	if(posPhotons > 0u) {
		mAssert(tree.posLights.lightCount > 0u);
		emit(tree.posLights, &photons[envPhotons + dirPhotons], posPhotons, splitDir,
			 intervalRight, rng, &photonRng[envPhotons + dirPhotons]);
	}
}

}}} // namespace mufflon::scene::lights
