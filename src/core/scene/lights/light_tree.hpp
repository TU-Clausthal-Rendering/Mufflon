#pragma once

#include "lights.hpp"
#include "light_sampling.hpp"
#include "export/api.hpp"
#include "core/memory/allocator.hpp"
#include "core/memory/residency.hpp"
#include "core/memory/synchronize.hpp"
#include "core/scene/handles.hpp"
#include "core/scene/types.hpp"
#include "core/scene/textures/texture.hpp"
#include "util/flag.hpp"
#include <vector>

#ifndef __CUDACC__
#include <optional>
#endif // __CUDACC__

// Forward declaration
namespace ei {
struct Box;
} // namespace ei

namespace mufflon { namespace scene { namespace lights {

struct LightSubTree {
#pragma pack(push, 1)
	struct alignas(16) Node {
		static constexpr u16 INVALID_TYPE = std::numeric_limits<u16>::max();

#ifndef __CUDACC__
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
#endif // __CUDACC__

		// Layout: [4,4,2]=10, [2,4,4]=10, [4,4,4]=12 bytes
		// Necessary duplication due to memory layout (2x32+16 and 16+2x32 bits)
		struct {
			CUDA_FUNCTION __forceinline__ void set_offset(std::size_t off) noexcept {
				mAssert(off < std::numeric_limits<u32>::max());
				offset = static_cast<u32>(off);
			}
			CUDA_FUNCTION __forceinline__ void mark_node() noexcept {
				type = INVALID_TYPE;
			}
			CUDA_FUNCTION __forceinline__ void set_light_type(LightType t) noexcept {
				mAssert(static_cast<u16>(t) < static_cast<u16>(LightType::NUM_LIGHTS));
				type = static_cast<u16>(t);
			}
			CUDA_FUNCTION __forceinline__ constexpr bool is_node() const noexcept {
				return type == INVALID_TYPE;
			}
			CUDA_FUNCTION __forceinline__ constexpr bool is_light() const noexcept {
				return !is_node();
			}
			CUDA_FUNCTION __forceinline__ constexpr LightType get_light_type() const noexcept {
				return static_cast<LightType>(type);
			}

			float flux;
			u32 offset;
			u16 type;
		} left;
		struct {
			CUDA_FUNCTION __forceinline__ void set_offset(std::size_t off) noexcept {
				mAssert(off < std::numeric_limits<u32>::max());
				offset = static_cast<u32>(off);
			}
			CUDA_FUNCTION __forceinline__ void mark_node() noexcept {
				type = INVALID_TYPE;
			}
			CUDA_FUNCTION __forceinline__ void set_light_type(LightType t) noexcept {
				mAssert(static_cast<u16>(t) < static_cast<u16>(LightType::NUM_LIGHTS));
				type = static_cast<u16>(t);
			}
			CUDA_FUNCTION __forceinline__ constexpr bool is_node() const noexcept {
				return type == INVALID_TYPE;
			}
			CUDA_FUNCTION __forceinline__ constexpr bool is_light() const noexcept {
				return !is_node();
			}
			CUDA_FUNCTION __forceinline__ constexpr LightType get_light_type() const noexcept {
				return static_cast<LightType>(type);
			}

			u16 type;
			u32 offset;
			float flux;
		} right;

		ei::Vec3 center;
	};
#pragma pack(pop)
	static_assert(sizeof(Node) == 32 && alignof(Node) == 16,
				  "Invalid node struct size/alignment");

	struct {
		ei::Vec3 center;
		float flux;
		u16 type;
	} root;
	std::size_t lightCount;
	Node* nodes;
	char* lights;
};

template < Device dev >
struct LightTree {
	static constexpr Device DEVICE = dev;

	EnvMapLight<dev> envLight;
	// Pointer to the tree elements
	LightSubTree dirLights;
	LightSubTree posLights;
	// Actual memory
	std::size_t length;
	ArrayDevHandle_t<DEVICE, char> memory;
};

#ifndef __CUDACC__
class LightTreeBuilder {
public:
	LightTreeBuilder();
	~LightTreeBuilder();
	LightTreeBuilder(LightTreeBuilder&&) = default;
	LightTreeBuilder& operator=(LightTreeBuilder&&) = default;

	// Builds the light tree from lists of positional and directional lights and
	// optionally an envmap light
	void build(std::vector<PositionalLights>&& posLights,
			   std::vector<DirectionalLight>&& dirLights,
			   const ei::Box& boundingBox,
			   std::optional<TextureHandle> envLight = std::nullopt);

	template < Device dev >
	const LightTree<dev>& aquire_tree() noexcept {
		this->synchronize<dev>();
		return m_trees.get<LightTree<dev>>();
	}

	template < Device dev >
	void synchronize() {
		//m_envMapTexture.synchronize<dev>();
		//mufflon::scene::synchronize<dev>(m_trees, m_flags, m_trees.get<LightTree<dev>>(), m_envMapTexture);
	}

	template < Device dev >
	void unload() {
		LightTree<dev>& tree = m_trees.get<LightTree<dev>>();
		tree.memory.handle = Allocator<dev>::free(tree.memory.handle, tree.length);
		// TODO: unload envmap handle
	}

private:
	TextureHandle m_envMapTexture;
	util::DirtyFlags<Device> m_flags;
	util::TaggedTuple<LightTree<Device::CPU>, LightTree<Device::CUDA>> m_trees;
};

// Functions for synchronizing a light tree
void synchronize(const LightTree<Device::CPU>& changed, LightTree<Device::CUDA>& sync, TextureHandle hdl);
void synchronize(const LightTree<Device::CUDA>& changed, LightTree<Device::CPU>& sync, TextureHandle hdl);
void unload(LightTree<Device::CPU>& tree);
void unload(LightTree<Device::CUDA>& tree);

#endif // __CUDACC__

namespace lighttree_detail {

// Helper to adjust PDF by the chance to pick light type
CUDA_FUNCTION __forceinline__ Photon adjustPdf(Photon&& sample, float chance) {
	sample.pos.pdf *= chance;
	return sample;
}

template < class ChildType >
CUDA_FUNCTION __forceinline__ ei::Vec3 get_cluster_center(const ChildType& child,
														  const LightSubTree& tree) {
	if(child.is_node()) {
		// Look up node
		return tree.nodes[child.offset].center;
	} else {
		mAssert(child.is_light());
		const char* light = &tree.lights[child.offset];
		switch(child.get_light_type()) {
			case LightType::POINT_LIGHT: return get_center(*reinterpret_cast<const PointLight*>(light));
			case LightType::SPOT_LIGHT: return get_center(*reinterpret_cast<const SpotLight*>(light));
			case LightType::AREA_LIGHT_TRIANGLE: return get_center(*reinterpret_cast<const AreaLightTriangle<CURRENT_DEV>*>(light));
			case LightType::AREA_LIGHT_QUAD: return get_center(*reinterpret_cast<const AreaLightQuad<CURRENT_DEV>*>(light));
			case LightType::AREA_LIGHT_SPHERE: return get_center(*reinterpret_cast<const AreaLightSphere<CURRENT_DEV>*>(light));
			case LightType::DIRECTIONAL_LIGHT: return get_center(*reinterpret_cast<const DirectionalLight*>(light));
			default: mAssert(false); return {};
		}
	}
}

CUDA_FUNCTION __forceinline__ ei::Vec3 get_flux(const char* light, LightType type, const ei::Vec3& aabbDiag) {
	switch(type) {
		case LightType::POINT_LIGHT: return get_flux(*reinterpret_cast<const PointLight*>(light));
		case LightType::SPOT_LIGHT: return get_flux(*reinterpret_cast<const SpotLight*>(light));
		case LightType::AREA_LIGHT_TRIANGLE: return get_flux(*reinterpret_cast<const AreaLightTriangle<CURRENT_DEV>*>(light));
		case LightType::AREA_LIGHT_QUAD: return get_flux(*reinterpret_cast<const AreaLightQuad<CURRENT_DEV>*>(light));
		case LightType::AREA_LIGHT_SPHERE: return get_flux(*reinterpret_cast<const AreaLightSphere<CURRENT_DEV>*>(light));
		case LightType::DIRECTIONAL_LIGHT: return get_flux(*reinterpret_cast<const DirectionalLight*>(light), aabbDiag);
		default: mAssert(false); return {};
	}
}

template < class ChildType >
CUDA_FUNCTION __forceinline__ ei::Vec3 get_cluster_flux(const ChildType& child,
														const LightSubTree& tree,
														const ei::Vec3& aabbDiag) {
	if(child.is_node()) {
		// Look up node
		return tree.nodes[child.offset].flux;
	} else {
		mAssert(child.is_light());
		return ei::sum(get_flux(&tree.lights[child.offset], child.get_light_type(), aabbDiag));
	}
}

// Converts the typeless memory into the given light type and samples it
CUDA_FUNCTION Photon sample_light(LightType type, const char* light,
										const ei::Box& bounds,
										const RndSet& rnd) {
	mAssert(static_cast<u16>(type) < static_cast<u16>(LightType::NUM_LIGHTS));
	switch(type) {
		case LightType::POINT_LIGHT: return sample_light(*reinterpret_cast<const PointLight*>(light), rnd);
		case LightType::SPOT_LIGHT: return sample_light(*reinterpret_cast<const SpotLight*>(light), rnd);
		case LightType::AREA_LIGHT_TRIANGLE: return sample_light(*reinterpret_cast<const AreaLightTriangle<CURRENT_DEV>*>(light), rnd);
		case LightType::AREA_LIGHT_QUAD: return sample_light(*reinterpret_cast<const AreaLightQuad<CURRENT_DEV>*>(light), rnd);
		case LightType::AREA_LIGHT_SPHERE: return sample_light(*reinterpret_cast<const AreaLightSphere<CURRENT_DEV>*>(light), rnd);
		case LightType::DIRECTIONAL_LIGHT: return sample_light(*reinterpret_cast<const DirectionalLight*>(light), bounds, rnd);
		default: mAssert(false); return {};
	}
}

// Converts the typeless memory into the given light type and samples it
CUDA_FUNCTION NextEventEstimation connect_light(LightType type, const char* light,
								  const ei::Vec3& position, float distSqr,
								  const ei::Box& bounds, const NEERndSet& rnd) {
	mAssert(static_cast<u16>(type) < static_cast<u16>(LightType::NUM_LIGHTS));
	switch(type) {
		case LightType::POINT_LIGHT: return connect_light(*reinterpret_cast<const PointLight*>(light), position, distSqr, rnd);
		case LightType::SPOT_LIGHT: return connect_light(*reinterpret_cast<const SpotLight*>(light), position, distSqr, rnd);
		case LightType::AREA_LIGHT_TRIANGLE: return connect_light(*reinterpret_cast<const AreaLightTriangle<CURRENT_DEV>*>(light), position, rnd);
		case LightType::AREA_LIGHT_QUAD: return connect_light(*reinterpret_cast<const AreaLightQuad<CURRENT_DEV>*>(light), position, rnd);
		case LightType::AREA_LIGHT_SPHERE: return connect_light(*reinterpret_cast<const AreaLightSphere<CURRENT_DEV>*>(light), position, rnd);
		case LightType::DIRECTIONAL_LIGHT: return connect_light(*reinterpret_cast<const DirectionalLight*>(light), position, bounds, rnd);
		default: mAssert(false); return {};
	}
}

} // namespace lighttree_detail

/** Shared code for emitting a single photon from the tree.
 * Takes the light tree, initial interval limits, and RNG number as inputs.
 * Also takes an index, which is initially used to distribute the photon
 * until it cannot uniquely identify a subtree (ie. index 1 for interval [0,2]
 * and flux distribution of 50/50).
 */
CUDA_FUNCTION Photon emit(const LightSubTree& tree, u64 left, u64 right,
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
	const LightSubTree::Node* currentNode = tree.nodes;
	u16 type = LightSubTree::Node::INVALID_TYPE;
	u32 offset = 0u;
	u64 intervalLeft = left;
	u64 intervalRight = right;
	float lightPdf = 1.f;

	// Iterate until we hit a leaf
	while(type == LightSubTree::Node::INVALID_TYPE) {
		mAssert(currentNode != nullptr);
		mAssert(intervalLeft <= intervalRight);

		// Scale the flux up
		const float fluxSum = currentNode->left.flux + currentNode->right.flux;
		const float probLeft = currentNode->left.flux / fluxSum;
		// Compute the integer bounds: once rounded down, once rounded up
		u64 leftRight = static_cast<u64>(intervalLeft + (intervalRight - intervalLeft)
										 * probLeft);
		u64 rightLeft = static_cast<u64>(std::ceilf(intervalLeft + (intervalRight - intervalLeft)
													* probLeft));
		// Check if our index falls into one of these
		if(index < leftRight) {
			lightPdf *= probLeft;
			if(currentNode->left.is_light()) {
				type = currentNode->left.type;
				offset = currentNode->left.offset;
				break;
			}
			currentNode = &tree.nodes[currentNode->left.offset];
			intervalRight = leftRight;
		} else if(index >= rightLeft) {
			lightPdf *= 1.f - probLeft;
			if(currentNode->right.is_light()) {
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
												 * probLeft);
			if(rng < split) {
				lightPdf *= probLeft;
				if(currentNode->left.is_light()) {
					type = currentNode->left.type;
					offset = currentNode->left.offset;
					break;
				}
				currentNode = &tree.nodes[currentNode->left.offset];
				intervalRight = split;
			} else {
				lightPdf *= 1.f - probLeft;
				if(currentNode->right.is_light()) {
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

	mAssert(type != LightSubTree::Node::INVALID_TYPE);
	// We got a light source! Sample it
	using namespace lighttree_detail;
	return adjustPdf(sample_light(static_cast<LightType>(type),
								  &tree.lights[offset], bounds, rnd),
					 lightPdf);
}


/** Shared code for emitting a single photon from the tree.
 * Takes the light tree, initial interval limits, and RNG number as inputs.
 * Also takes an index, which is initially used to distribute the photon
 * until it cannot uniquely identify a subtree (ie. index 1 for interval [0,2]
 * and flux distribution of 50/50).
 */
template < class Guide >
CUDA_FUNCTION NextEventEstimation connect(const LightSubTree& tree, u64 left, u64 right,
										  u64 index, u64 rng, const ei::Vec3& position,
										  const ei::Box& bounds, const NEERndSet& rnd,
										  Guide&& guide) {
	using namespace lighttree_detail;
	// Check: do we have more than one light here?
	if(tree.lightCount == 1u) {
		// Nothing to do but sample the photon
		mAssert(tree.root.type < static_cast<u16>(LightType::NUM_LIGHTS));
		return lighttree_detail::connect_light(static_cast<LightType>(tree.root.type),
											   &tree.lights[0u], position,
											   ei::lensq(tree.root.center - position),
											   bounds, rnd);
	}

	// Traverse the tree to split chance between lights
	const LightSubTree::Node* currentNode = tree.nodes;
	u16 type = LightSubTree::Node::INVALID_TYPE;
	u32 offset = 0u;
	u64 intervalLeft = left;
	u64 intervalRight = right;
	float lightPdf = 1.f;

	// Iterate until we hit a leaf
	while(type == LightSubTree::Node::INVALID_TYPE) {
		mAssert(currentNode != nullptr);
		mAssert(intervalLeft <= intervalRight);
		
		// Find out the two cluster centers
		const ei::Vec3 leftCenter = get_cluster_center(currentNode->left, tree);
		const ei::Vec3 rightCenter = get_cluster_center(currentNode->right, tree);

		// Scale the flux up
		float probLeft = guide(position, leftCenter, rightCenter, currentNode->left.flux, currentNode->right.flux);
		// Compute the integer bounds: once rounded down, once rounded up
		u64 leftRight = static_cast<u64>(intervalLeft + (intervalRight - intervalLeft)
										 * probLeft);
		u64 rightLeft = static_cast<u64>(std::ceilf(intervalLeft + (intervalRight - intervalLeft)
													* probLeft));
		// Check if our index falls into one of these
		if(index < leftRight) {
			lightPdf *= probLeft;
			if(currentNode->left.is_light()) {
				type = currentNode->left.type;
				offset = currentNode->left.offset;
				break;
			}
			currentNode = &tree.nodes[currentNode->left.offset];
			intervalRight = leftRight;
		} else if(index >= rightLeft) {
			lightPdf *= 1.f - probLeft;
			if(currentNode->right.is_light()) {
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
												 * probLeft);
			if(rng < split) {
				lightPdf *= probLeft;
				if(currentNode->left.is_light()) {
					type = currentNode->left.type;
					offset = currentNode->left.offset;
					break;
				}
				currentNode = &tree.nodes[currentNode->left.offset];
				intervalRight = split;
			} else {
				lightPdf *= 1.f - probLeft;
				if(currentNode->right.is_light()) {
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

	mAssert(type != LightSubTree::Node::INVALID_TYPE);
	// We got a light source! Sample it
	// TODO: incorporate light selection probability
	return connect_light(static_cast<LightType>(type), &tree.lights[offset],
						 position, ei::lensq(currentNode->center - position),
						 bounds, rnd);
}

/**
 * Emits a single photon from a light source.
 * To ensure a good distribution, we also take an index, which is used to guide
 * the descent into the tree when it is possible to do so without using RNG.
 */
CUDA_FUNCTION Photon emit(const LightTree<CURRENT_DEV>& tree, u64 index,
								u64 indexMax, u64 rng, const ei::Box& bounds,
								const RndSet& rnd) {
	// Figure out which of the three top-level light types get the photon
	// Implicit left boundary of 0 for the interval
	u64 intervalRight = indexMax;

	float fluxSum = tree.dirLights.root.flux + tree.posLights.root.flux;
	float envProb = 0.f;
	if(is_valid(tree.envLight.texHandle)) {
		fluxSum += ei::sum(tree.envLight.flux);
		envProb = ei::sum(tree.envLight.flux) / fluxSum;
	}
	float dirProb = tree.dirLights.root.flux / fluxSum;
	float posProb = tree.posLights.root.flux / fluxSum;

	// Now split up based on flux
	// First is envmap...
	u64 rightEnv = static_cast<u64>(intervalRight * envProb);
	if(index < rightEnv) {
		mAssert(is_valid(tree.envLight.texHandle));
		return lighttree_detail::adjustPdf(sample_light(tree.envLight,
														rnd), envProb);
	}
	// ...then come directional lights...
	u64 leftDir = static_cast<u64>(std::ceilf(intervalRight * envProb));
	u64 rightDir = static_cast<u64>(intervalRight
									* (ei::sum(tree.envLight.flux)
									   + tree.dirLights.root.flux) / fluxSum);
	if(index >= leftDir && index < rightDir) {
		mAssert(tree.dirLights.lightCount > 0u);
		return lighttree_detail::adjustPdf(emit(tree.dirLights, leftDir,
												rightDir, index, rng,
												bounds, rnd), dirProb);
	}
	// ...and last positional lights
	u64 leftPos = static_cast<u64>(std::ceilf(intervalRight
											  * (ei::sum(tree.envLight.flux)
												 + tree.dirLights.root.flux / fluxSum)));
	if(index >= leftPos) {
		mAssert(tree.posLights.lightCount > 0u);
		return lighttree_detail::adjustPdf(emit(tree.posLights, leftPos,
												intervalRight, index, rng,
												bounds, rnd), posProb);
	}

	// If we made it until here, it means that we fell between
	// the integer bounds of photon distribution
	// Thus we need RNG to decide
	// TODO: it could fall onto a boundary repeatedly, but that costs
	// performance (even more divergence)
	u64 splitEnv = static_cast<u64>(std::numeric_limits<u64>::max()
									* envProb);
	u64 splitDir = static_cast<u64>(std::numeric_limits<u64>::max()
									* (ei::sum(tree.envLight.flux)
									   + tree.dirLights.root.flux) / fluxSum);
	if(rng < splitEnv) {
		mAssert(is_valid(tree.envLight.texHandle));
		return lighttree_detail::adjustPdf(sample_light(tree.envLight, rnd),
										   envProb);
	} else if(rng < splitDir) {
		mAssert(tree.dirLights.lightCount > 0u);
		return lighttree_detail::adjustPdf(emit(tree.dirLights, splitEnv,
												splitDir, rng, rng, bounds, rnd),
										   dirProb);
	} else {
		mAssert(tree.posLights.lightCount > 0u);
		return lighttree_detail::adjustPdf(emit(tree.posLights, splitDir,
												std::numeric_limits<u64>::max(),
												rng, rng, bounds, rnd), posProb);
	}
}

// Emits a single photon from a light source.
template < Device dev >
CUDA_FUNCTION Photon emit(const LightTree<dev>& tree, u64 rng,
						  const ei::Box& bounds, const RndSet& rnd) {
	// No index means our RNG serves as an index
	return emit(tree, rng, std::numeric_limits<u64>::max(), rng, bounds, rnd);
}

/**
 * Performs next-event estimation.
 * For selecting the light source we want to connect against we try to maximize
 * the radiance.
 */
template < Device dev, class Guide >
CUDA_FUNCTION NextEventEstimation connect(const LightTree<dev>& tree, u64 index,
										  u64 indexMax, u64 rng, const ei::Vec3& position,
										  const ei::Box& bounds, const RndSet& rnd,
										  Guide&& guide) {
	// Figure out which of the three top-level light types get the photon
	// Implicit left boundary of 0 for the interval
	u64 intervalRight = indexMax;

	float fluxSum = tree.dirLights.root.flux + tree.posLights.root.flux;
	float envProb = 0.f;
	if(is_valid(tree.envLight.texHandle)) {
		fluxSum += ei::sum(tree.envLight.flux);
		envProb = ei::sum(tree.envLight.flux) / fluxSum;
	}
	float dirProb = tree.dirLights.root.flux / fluxSum;
	float posProb = tree.posLights.root.flux / fluxSum;

	// Now split up based on flux
	// First is envmap...
	u64 rightEnv = static_cast<u64>(intervalRight * envProb);
	if(index < rightEnv) {
		mAssert(is_valid(tree.envLight.texHandle));
		// TODO: adjust light probability
		return connect_light(tree.envLight, position, rnd);
	}
	// ...then come directional lights...
	u64 leftDir = static_cast<u64>(std::ceilf(intervalRight * envProb));
	u64 rightDir = static_cast<u64>(intervalRight
									* (ei::sum(tree.envLight.flux)
									   + tree.dirLights.root.flux) / fluxSum);
	if(index >= leftDir && index < rightDir) {
		mAssert(tree.dirLights.lightCount > 0u);
		return connect(tree.dirLights, leftDir, rightDir, index, rng, position,
					   bounds, rnd, guide);
	}
	// ...and last positional lights
	u64 leftPos = static_cast<u64>(std::ceilf(intervalRight
											  * (ei::sum(tree.envLight.flux)
												 + tree.dirLights.root.flux / fluxSum)));
	if(index >= leftPos) {
		mAssert(tree.posLights.lightCount > 0u);
		// TODO: adjust light probability
		return connect(tree.posLights, leftPos, intervalRight, index, rng, position,
					   bounds, rnd, guide);
	}

	// If we made it until here, it means that we fell between
	// the integer bounds of photon distribution
	// Thus we need RNG to decide
	u64 splitEnv = static_cast<u64>(std::numeric_limits<u64>::max()
									* envProb);
	u64 splitDir = static_cast<u64>(std::numeric_limits<u64>::max()
									* (ei::sum(tree.envLight.flux)
									   + tree.dirLights.root.flux) / fluxSum);
	if(rng < splitEnv) {
		mAssert(is_valid(tree.envLight.texHandle));
		// TODO: adjust light probability
		return connect_light(tree.envLight, position, rnd);
	} else if(rng < splitDir) {
		mAssert(tree.dirLights.lightCount > 0u);
		// TODO: adjust light probability
		return connect(tree.dirLights, splitEnv,
					   splitDir, rng, rng, position,
					   bounds, rnd, guide);
	} else {
		mAssert(tree.posLights.lightCount > 0u);
		// TODO: adjust light probability
		return connect(tree.posLights, splitDir,
					   std::numeric_limits<u64>::max(),
					   rng, rng, position, bounds,
					   rnd, guide);
	}
}

}}} // namespace mufflon::scene::lights
