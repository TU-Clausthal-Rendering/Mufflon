#pragma once

#include "lights.hpp"
#include "light_sampling.hpp"
#include "export/api.hpp"
#include "core/memory/allocator.hpp"
#include "core/memory/residency.hpp"
#include "core/memory/synchronize.hpp"
#include "core/memory/dyntype_memory.hpp"
#include "core/memory/hashmap.hpp"
#include "core/scene/handles.hpp"
#include "core/scene/types.hpp"
#include "core/scene/textures/texture.hpp"
#include "util/flag.hpp"
#include <vector>
#include <unordered_map>

// Forward declaration
namespace ei {
struct Box;
} // namespace ei

namespace mufflon { namespace scene { namespace lights {

struct LightSubTree {
#pragma pack(push, 1)
	struct alignas(16) Node {
		static constexpr u16 INTERNAL_NODE_TYPE = std::numeric_limits<u16>::max();

#ifndef __CUDACC__
		Node(const char* base,
			 u32 leftOffset, u16 leftType,
			 u32 rightOffset, u16 rightType,
			 const ei::Vec3& bounds);
#endif // __CUDACC__

		// Layout: [4,4,2]=10, [2,4,4]=10, [4,4,4]=12 bytes
		// Necessary duplication due to memory layout (2x32+16 and 16+2x32 bits)
		struct {
			CUDA_FUNCTION __forceinline__ constexpr bool is_light() const noexcept {
				return type < u16(LightType::NUM_LIGHTS);
			}

			float flux;
			u32 offset;
			u16 type;
		} left;
		struct {
			CUDA_FUNCTION __forceinline__ constexpr bool is_light() const noexcept {
				return type < u16(LightType::NUM_LIGHTS);
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
		float flux { 0.0f };
		u16 type;
	} root;
	std::size_t lightCount { 0 };
	char* memory { nullptr };

	CUDA_FUNCTION __forceinline__ Node* get_node(u32 offset) { return as<Node>(memory + offset); }
	CUDA_FUNCTION __forceinline__ const Node* get_node(u32 offset) const { return as<Node>(memory + offset); }
};

template < Device dev >
struct LightTree {
	static constexpr Device DEVICE = dev;

	EnvMapLight<dev> envLight;
	// Pointer to the tree elements
	LightSubTree dirLights;
	LightSubTree posLights;
	// Actual memory
	std::size_t length { 0 };
	ArrayDevHandle_t<DEVICE, char> memory { nullptr };
	// A map to find the node of a given primitve.
	// The map stores an encoded path to the node. Storing its pointer/offset
	// would be useless in terms of finding its probability. Therefore,
	// the tree must be traversed
	HashMap<dev, PrimitiveHandle, u32> primToNodePath;

	LightTree(std::size_t numPosLights) : primToNodePath{u32(numPosLights)} {}
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
			   TextureHandle envLight = nullptr);

	template < Device dev >
	const LightTree<dev>& aquire_tree() noexcept {
		this->synchronize<dev>();
		if constexpr(dev == Device::CPU) return *m_treeCpu;
		else return *m_treeCuda;
	}

	template < Device dev >
	void synchronize();

	template < Device dev >
	void unload() {
		if(dev == Device::CPU && m_treeCpu) {
			Allocator<dev>::free(m_treeCpu->memory, m_treeCpu->length);
			m_treeCpu = nullptr;
		} else if(m_treeCuda) {
			Allocator<dev>::free(m_treeCuda->memory, m_treeCuda->length);
			m_treeCuda->primToNodePath.free();
			m_treeCuda = nullptr;
		}
		// TODO: unload envmap handle
	}

private:
	util::DirtyFlags<Device> m_flags;
	std::unique_ptr<LightTree<Device::CPU>> m_treeCpu;
	std::unique_ptr<LightTree<Device::CUDA>> m_treeCuda;
	// The tree is build on CPU side. For synchronization we need a possiblity to
	// find the CUDA textures.
	std::unordered_map<textures::ConstTextureDevHandle_t<Device::CPU>, TextureHandle> m_textureMap;

	void remap_textures(const char* cpuMem, u32 offset, u16 type, char* cudaMem);
};

#endif // __CUDACC__

namespace lighttree_detail {

// Helper to adjust PDF by the chance to pick light type
CUDA_FUNCTION __forceinline__ Photon adjustPdf(Photon&& sample, float chance) {
	sample.pos.pdf *= chance;
	return sample;
}

CUDA_FUNCTION __forceinline__ ei::Vec3 get_center(const void* node, u16 type) {
	switch(type) {
		case u16(LightType::POINT_LIGHT): return get_center(*as<PointLight>(node));
		case u16(LightType::SPOT_LIGHT): return get_center(*as<SpotLight>(node));
		case u16(LightType::AREA_LIGHT_TRIANGLE): return get_center(*as<AreaLightTriangle<CURRENT_DEV>>(node));
		case u16(LightType::AREA_LIGHT_QUAD): return get_center(*as<AreaLightQuad<CURRENT_DEV>>(node));
		case u16(LightType::AREA_LIGHT_SPHERE): return get_center(*as<AreaLightSphere<CURRENT_DEV>>(node));
		case u16(LightType::DIRECTIONAL_LIGHT): return get_center(*as<DirectionalLight>(node));
		case u16(LightType::ENVMAP_LIGHT): return get_center(*as<EnvMapLight<CURRENT_DEV>>(node));
		case LightSubTree::Node::INTERNAL_NODE_TYPE: return as<LightSubTree::Node>(node)->center;
	}
	mAssert(false);
	return ei::Vec3{0.0f};
}


// Converts the typeless memory into the given light type and samples it
CUDA_FUNCTION Photon sample_light(LightType type, const char* light,
										const ei::Box& bounds,
										const math::RndSet2& rnd) {
	mAssert(static_cast<u16>(type) < static_cast<u16>(LightType::NUM_LIGHTS));
	switch(type) {
		case LightType::POINT_LIGHT: return sample_light_pos(*reinterpret_cast<const PointLight*>(light), rnd);
		case LightType::SPOT_LIGHT: return sample_light_pos(*reinterpret_cast<const SpotLight*>(light), rnd);
		case LightType::AREA_LIGHT_TRIANGLE: return sample_light_pos(*reinterpret_cast<const AreaLightTriangle<CURRENT_DEV>*>(light), rnd);
		case LightType::AREA_LIGHT_QUAD: return sample_light_pos(*reinterpret_cast<const AreaLightQuad<CURRENT_DEV>*>(light), rnd);
		case LightType::AREA_LIGHT_SPHERE: return sample_light_pos(*reinterpret_cast<const AreaLightSphere<CURRENT_DEV>*>(light), rnd);
		case LightType::DIRECTIONAL_LIGHT: return sample_light_pos(*reinterpret_cast<const DirectionalLight*>(light), bounds, rnd);
		default: mAssert(false); return {};
	}
}

// Converts the typeless memory into the given light type and samples it
CUDA_FUNCTION NextEventEstimation connect_light(LightType type, const char* light,
								  const ei::Vec3& position, float distSqr,
								  const ei::Box& bounds, const math::RndSet2& rnd) {
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
								const math::RndSet2& rnd) {
	// Check: do we have more than one light here?
	if(tree.lightCount == 1u) {
		// Nothing to do but sample the photon
		mAssert(tree.root.type < static_cast<u16>(LightType::NUM_LIGHTS));
		return lighttree_detail::sample_light(static_cast<LightType>(tree.root.type),
											  tree.memory, bounds, rnd);
	}

	// Traverse the tree to split chance between lights
	const LightSubTree::Node* currentNode = as<LightSubTree::Node>(tree.memory);
	u16 type = LightSubTree::Node::INTERNAL_NODE_TYPE;
	u32 offset = 0u;
	u64 intervalLeft = left;
	u64 intervalRight = right;
	float lightPdf = 1.f;

	// Iterate until we hit a leaf
	while(type == LightSubTree::Node::INTERNAL_NODE_TYPE) {
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
			currentNode = tree.get_node(currentNode->left.offset);
			intervalRight = leftRight;
		} else if(index >= rightLeft) {
			lightPdf *= 1.f - probLeft;
			if(currentNode->right.is_light()) {
				type = currentNode->right.type;
				offset = currentNode->right.offset;
				break;
			}
			currentNode = tree.get_node(currentNode->right.offset);
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
				currentNode = tree.get_node(currentNode->left.offset);
				intervalRight = split;
			} else {
				lightPdf *= 1.f - probLeft;
				if(currentNode->right.is_light()) {
					type = currentNode->right.type;
					offset = currentNode->right.offset;
					break;
				}
				currentNode = tree.get_node(currentNode->right.offset);
				intervalLeft = split;
			}

			// Make sure that for following iterations, we use index as well
			index = rng;
		}
	}

	mAssert(type != LightSubTree::Node::INTERNAL_NODE_TYPE);
	// We got a light source! Sample it
	using namespace lighttree_detail;
	return adjustPdf(sample_light(static_cast<LightType>(type),
								  tree.memory + offset, bounds, rnd),
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
										  const ei::Box& bounds, const math::RndSet2& rnd,
										  Guide&& guide) {
	using namespace lighttree_detail;
	// Check: do we have more than one light here?
	if(tree.lightCount == 1u) {
		// Nothing to do but sample the photon
		mAssert(tree.root.type < static_cast<u16>(LightType::NUM_LIGHTS));
		return lighttree_detail::connect_light(static_cast<LightType>(tree.root.type),
											   tree.memory, position,
											   ei::lensq(tree.root.center - position),
											   bounds, rnd);
	}

	// Traverse the tree to split chance between lights
	const LightSubTree::Node* currentNode = as<LightSubTree::Node>(tree.memory);
	u16 type = LightSubTree::Node::INTERNAL_NODE_TYPE;
	u32 offset = 0u;
	u64 intervalLeft = left;
	u64 intervalRight = right;
	float lightPdf = 1.f;

	// Iterate until we hit a leaf
	while(type == LightSubTree::Node::INTERNAL_NODE_TYPE) {
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
			currentNode = tree.get_node(currentNode->left.offset);
			intervalRight = leftRight;
		} else if(index >= rightLeft) {
			lightPdf *= 1.f - probLeft;
			if(currentNode->right.is_light()) {
				type = currentNode->right.type;
				offset = currentNode->right.offset;
				break;
			}
			currentNode = tree.get_node(currentNode->right.offset);
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
				currentNode = tree.get_node(currentNode->left.offset);
				intervalRight = split;
			} else {
				lightPdf *= 1.f - probLeft;
				if(currentNode->right.is_light()) {
					type = currentNode->right.type;
					offset = currentNode->right.offset;
					break;
				}
				currentNode = tree.get_node(currentNode->right.offset);
				intervalLeft = split;
			}

			// Make sure that for following iterations, we use index as well
			index = rng;
		}
	}

	mAssert(type != LightSubTree::Node::INTERNAL_NODE_TYPE);
	// We got a light source! Sample it
	// TODO: incorporate light selection probability
	return connect_light(static_cast<LightType>(type), tree.memory + offset,
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
								const math::RndSet2_1& rnd) {
	using namespace lighttree_detail;
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
		return adjustPdf(sample_light_pos(tree.envLight, bounds, rnd), envProb);
	}
	// ...then come directional lights...
	u64 leftDir = static_cast<u64>(std::ceilf(intervalRight * envProb));
	u64 rightDir = static_cast<u64>(intervalRight
									* (ei::sum(tree.envLight.flux)
									   + tree.dirLights.root.flux) / fluxSum);
	if(index >= leftDir && index < rightDir) {
		mAssert(tree.dirLights.lightCount > 0u);
		return adjustPdf(emit(tree.dirLights, leftDir, rightDir,
							  index, rng, bounds, rnd), dirProb);
	}
	// ...and last positional lights
	u64 leftPos = static_cast<u64>(std::ceilf(intervalRight
											  * (ei::sum(tree.envLight.flux)
												 + tree.dirLights.root.flux / fluxSum)));
	if(index >= leftPos) {
		mAssert(tree.posLights.lightCount > 0u);
		return adjustPdf(emit(tree.posLights, leftPos, intervalRight,
							  index, rng, bounds, rnd), posProb);
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
		return adjustPdf(sample_light_pos(tree.envLight, bounds, rnd), envProb);
	} else if(rng < splitDir) {
		mAssert(tree.dirLights.lightCount > 0u);
		return adjustPdf(emit(tree.dirLights, splitEnv, splitDir, rng, rng, bounds, rnd),
						 dirProb);
	} else {
		mAssert(tree.posLights.lightCount > 0u);
		return adjustPdf(emit(tree.posLights, splitDir, std::numeric_limits<u64>::max(),
							  rng, rng, bounds, rnd), posProb);
	}
}

// Emits a single photon from a light source.
CUDA_FUNCTION Photon emit(const LightTree<CURRENT_DEV>& tree, u64 rng,
						  const ei::Box& bounds, const math::RndSet2_1& rnd) {
	// No index means our RNG serves as an index
	return emit(tree, rng, std::numeric_limits<u64>::max(), rng, bounds, rnd);
}

/**
 * Performs next-event estimation.
 * For selecting the light source we want to connect against we try to maximize
 * the radiance.
 */
template < class Guide >
CUDA_FUNCTION NextEventEstimation connect(const LightTree<CURRENT_DEV>& tree, u64 index,
										  u64 indexMax, u64 rng, const ei::Vec3& position,
										  const ei::Box& bounds, const math::RndSet2& rnd,
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

/*
 * Hitable light source (area lights) must provide MIS helpers which are
 * called if a surface is hit randomly. This method computes the area pdf
 * which would be produced by the above connect_light() samplers.
 */
template < class Guide >
CUDA_FUNCTION AreaPdf connect_pdf(const LightTree<CURRENT_DEV>& tree,
								  PrimitiveHandle primitive,
								  const ei::Vec3& refPosition, Guide&& guide) {
	mAssert(primitive != ~0u);
	using namespace lighttree_detail;

	float p = tree.posLights.root.flux / (tree.dirLights.root.flux + tree.posLights.root.flux + ei::sum(tree.envLight.flux));
	u32 code = *tree.primToNodePath.find(primitive); // If crash here, you have hit an emissive surface which is not in the light tree. This is a fundamental problem and not only an access violation.

	// Travers through the tree to compute the complete, guide dependent pdf
	u32 offset = 0u;
	u16 type = tree.posLights.root.type;
	// Iterate until we hit a leaf
	while(type == LightSubTree::Node::INTERNAL_NODE_TYPE) {
		const LightSubTree::Node* currentNode = tree.posLights.get_node(offset);
		// Find out the two cluster centers
		const ei::Vec3 leftCenter = get_center(tree.posLights.memory + currentNode->left.offset, currentNode->left.type);
		const ei::Vec3 rightCenter = get_center(tree.posLights.memory + currentNode->right.offset, currentNode->right.type);

		float pLeft = guide(refPosition, leftCenter, rightCenter, currentNode->left.flux, currentNode->right.flux);
		// Go right? The code has stored the path to the primitive (beginning with the most significant bit).
		if(code & 0x80000000) {
			p *= (1.0f - pLeft);
			type = currentNode->right.type;
			offset = currentNode->right.offset;
		} else {
			p *= pLeft;
			type = currentNode->left.type;
			offset = currentNode->left.offset;
		}
		code <<= 1;
	}

	// Now, p is the choice probability, but we also need the surface area
	switch(static_cast<LightType>(type)) {
		case LightType::AREA_LIGHT_TRIANGLE: {
			auto& a = *as<AreaLightTriangle<CURRENT_DEV>>(tree.posLights.memory + offset);
			float area = ei::surface(ei::Triangle{a.points[0], a.points[1], a.points[2]});
			return AreaPdf{ p / area };
		}
		case LightType::AREA_LIGHT_QUAD: {
			auto& a = *as<AreaLightQuad<CURRENT_DEV>>(tree.posLights.memory + offset);
			float area = ei::surface(ei::Triangle{a.points[0], a.points[1], a.points[2]})
					   + ei::surface(ei::Triangle{a.points[0], a.points[2], a.points[3]});
			return AreaPdf{ p / area };
		}
		case LightType::AREA_LIGHT_SPHERE: {
			auto& a = *as<AreaLightSphere<CURRENT_DEV>>(tree.posLights.memory + offset);
			float area = 4 * ei::PI * ei::sq(a.radius);
			return AreaPdf{ p / area };
		}
		default:
			mAssertMsg(false, "Decoded node must be some hitable area light.");
	}
	return AreaPdf{0.0f};
}

// Guide the light tree traversal based on flux only
CUDA_FUNCTION float guide_flux(const scene::Point&, const scene::Point&, const scene::Point&,
							   float leftFlux, float rightFlux) {
	return leftFlux / (leftFlux + rightFlux);
}

// Guide the light tree traversal based on expected contribution
CUDA_FUNCTION float guide_flux_pos(const scene::Point& refPosition,
								   const scene::Point& leftPosition,
								   const scene::Point& rightPosition,
								   float leftFlux, float rightFlux) {
	leftFlux /= lensq(leftPosition - refPosition);
	rightFlux /= lensq(rightPosition - refPosition);
	return leftFlux / (leftFlux + rightFlux);
}


}}} // namespace mufflon::scene::lights
