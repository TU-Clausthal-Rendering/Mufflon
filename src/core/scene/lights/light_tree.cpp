#include "light_tree.hpp"
#include "ei/3dtypes.hpp"
#include "util/assert.hpp"
#include "core/concepts.hpp"
#include "core/memory/allocator.hpp"
#include "core/cuda/error.hpp"
#include "core/math/sfcurves.hpp"
#include "core/scene/descriptors.hpp"
#include "core/scene/accel_structs/intersection.hpp"
#include "core/scene/materials/material.hpp"
#include "core/scene/lights/light_medium.hpp"
#include "core/scene/lights/light_tree_sampling.hpp"
#include "background.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <functional>


namespace mufflon { namespace scene { namespace lights {

namespace {

float get_flux(const void* light, u16 type, const ei::Vec3& aabbDiag, const int* materials) {
	switch(type) {
		case u16(LightType::POINT_LIGHT): return ei::sum(get_flux(*as<PointLight>(light)));
		case u16(LightType::SPOT_LIGHT): return ei::sum(get_flux(*as<SpotLight>(light)));
		case u16(LightType::AREA_LIGHT_TRIANGLE): return ei::sum(get_flux(*as<AreaLightTriangle<Device::CPU>>(light), materials));
		case u16(LightType::AREA_LIGHT_QUAD): return ei::sum(get_flux(*as<AreaLightQuad<Device::CPU>>(light), materials));
		case u16(LightType::AREA_LIGHT_SPHERE): return ei::sum(get_flux(*as<AreaLightSphere<Device::CPU>>(light), materials));
		case u16(LightType::DIRECTIONAL_LIGHT): return ei::sum(get_flux(*as<DirectionalLight>(light), aabbDiag));
		case u16(LightType::ENVMAP_LIGHT): return ei::sum(as<BackgroundDesc<Device::CPU>>(light)->flux);
		case LightSubTree::Node::INTERNAL_NODE_TYPE: return as<LightSubTree::Node>(light)->left.flux + as<LightSubTree::Node>(light)->right.flux;
	}
	return 0.0f;
}

// Computes the offset of the i-th point light - positional ones need a sum table!
// The offsets are relative to the trees node-memory (which points to the internal nodes
// first, then the lights).
template < class LT >
class LightOffset;

template <>
class LightOffset<DirectionalLight> {
public:
	LightOffset(const std::vector<DirectionalLight>& lights, std::size_t globalOffset) :
		m_lightCount{ lights.size() },
		m_globalOffset{ globalOffset }
	{}
	
	constexpr u32 operator[](std::size_t lightIndex) const noexcept {
		return static_cast<u32>(m_globalOffset + sizeof(DirectionalLight) * lightIndex);
	}

	std::size_t light_count() const noexcept {
		return m_lightCount;
	}

	std::size_t mem_size() const noexcept {
		return m_globalOffset + m_lightCount * sizeof(DirectionalLight);
	}

	u16 type(std::size_t /*lightIndex*/) const noexcept {
		return u16(LightType::DIRECTIONAL_LIGHT);
	}
private:
	std::size_t m_lightCount;
	std::size_t m_globalOffset;
};

// helper from https://en.cppreference.com/w/cpp/utility/variant/visit
template<class... Ts> struct overloaded : Ts... { using Ts::operator()...; };
template<class... Ts> overloaded(Ts...) -> overloaded<Ts...>;
template <>
class LightOffset<PositionalLights> {
public:
	LightOffset(const std::vector<PositionalLights>& lights, std::size_t globalOffset) :
		m_offsets(lights.size()) {
		u32 prevOffset = u32(globalOffset);
		for(std::size_t i = 0u; i < lights.size(); ++i) {
			m_offsets[i] = std::visit(overloaded{
					[&prevOffset](const PointLight& /*light*/) constexpr { LightRef res{prevOffset, u16(LightType::POINT_LIGHT)}; prevOffset += sizeof(PointLight); return res; },
					[&prevOffset](const SpotLight& /*light*/) constexpr { LightRef res{prevOffset, u16(LightType::SPOT_LIGHT)}; prevOffset += sizeof(SpotLight); return res; },
					[&prevOffset](const AreaLightTriangleDesc& /*light*/) constexpr { LightRef res{prevOffset, u16(LightType::AREA_LIGHT_TRIANGLE)}; prevOffset += sizeof(AreaLightTriangle<Device::CPU>); return res; },
					[&prevOffset](const AreaLightQuadDesc& /*light*/) constexpr { LightRef res{prevOffset, u16(LightType::AREA_LIGHT_QUAD)}; prevOffset += sizeof(AreaLightQuad<Device::CPU>); return res; },
					[&prevOffset](const AreaLightSphereDesc& /*light*/) constexpr { LightRef res{prevOffset, u16(LightType::AREA_LIGHT_SPHERE)}; prevOffset += sizeof(AreaLightSphere<Device::CPU>); return res; }
				}, lights[i].light);
		}
		m_memSize = prevOffset;
	}

	u32 operator[](std::size_t lightIndex) const noexcept {
		mAssert(lightIndex < m_offsets.size());
		return m_offsets[lightIndex].offset;
	}

	std::size_t light_count() const noexcept {
		return m_offsets.size();
	}
	std::size_t mem_size() const noexcept {
		return m_memSize;
	}

	u16 type(std::size_t lightIndex) const noexcept {
		return m_offsets[lightIndex].type;
	}
private:
	struct LightRef {
		u32 offset;
		u16 type;
	};
	std::vector<LightRef> m_offsets;
	std::size_t m_memSize;
};


template < class LightT >
void create_light_tree(LightOffset<LightT>& lightOffsets, LightSubTree& tree,
					   const ei::Vec3& aabbDiag,
					   const int* materials) {
	using Node = LightSubTree::Node;

	tree.lightCount = lightOffsets.light_count();
	if(lightOffsets.light_count() == 0u)
		return;

	// Only one light -> no actual tree, only light
	if(lightOffsets.light_count() == 1u) {
		tree.root.type = static_cast<u16>(lightOffsets.type(0));
		tree.root.flux = get_flux(tree.memory + lightOffsets[0], lightOffsets.type(0), aabbDiag, materials);
		return;
	}

	/* Example tree: (I are intrenal nodes, L are light sources/leaves)
	 *                                 I0
	 *                   I1                        I2
	 *           I3                I4          I5      I6
	 *      I7       I8        I9       L0  L1   L2  L3   L4
	 *    L5  L6   L7  L8   L9   L10
	 *
	 */

	// Compute the height of the tree IF the tree would be perfectly balanced, without
	// the extra nodes (e.g. without L5-L10, so height == 3)
	const std::size_t height = static_cast<std::size_t>(std::log2(lightOffsets.light_count()));
	mAssert(height > 0u);
	// We can then compute how many additional internal nodes there need to be (e.g. I7 - I9, so 3)
	const std::size_t extraNodes = lightOffsets.light_count() - (1ull << height);
	if(extraNodes > 0u) {
		// Compute starting positions for the extra internal nodes and their lights
		// In the example, they would be startNode == 8 and startLight == 5
		const std::size_t startNode = (1lu << height) - 1u;
		const std::size_t startLight = lightOffsets.light_count() - 2u * extraNodes;
		// Create the internal nodes putting together two lights
		for(std::size_t i = 0u; i < extraNodes; ++i) {
			mAssert(startLight + 2u * i + 1u < lightOffsets.light_count());
			mAssert(startNode + i < tree.internalNodeCount);
			const std::size_t left = startLight + 2u * i;
			const std::size_t right = startLight + 2u * i + 1u;
			Node& interiorNode = as<Node>(tree.memory)[startNode + i];

			interiorNode = Node{ tree.memory, lightOffsets[left], lightOffsets.type(left),
								 lightOffsets[right], lightOffsets.type(right), aabbDiag, materials };
		}

		// To ensure we can later uniformly create all remaining internal nodes, we also form
		// the "second-lowest" layer of internal nodes (e.g. I3, which may occur multiple times,
		// and I4, of which there will at most be one kind)
		const std::size_t startInnerNode = (1lu << (height - 1u)) - 1u;
		for(std::size_t i = 0u; i < extraNodes / 2u; ++i) {
			mAssert(startNode + 2u*i + 1u < tree.internalNodeCount);
			const std::size_t left = startNode + 2u * i;
			const std::size_t right = startNode + 2u * i + 1u;
			Node& node = as<Node>(tree.memory)[startInnerNode + i];

			node = Node{ tree.memory, u32(left * sizeof(Node)), Node::INTERNAL_NODE_TYPE,
						 u32(right * sizeof(Node)), Node::INTERNAL_NODE_TYPE, aabbDiag, materials };
		}

		// Create the one possible internal node that has another internal node as left child
		// and a light source as right child (I4)
		if(extraNodes % 2 != 0u) {
			// One interior leftover; must be first light and last internal node
			mAssert(startNode + extraNodes - 1u < tree.internalNodeCount);
			const std::size_t left = startNode + extraNodes - 1u;
			const std::size_t right = 0;
			Node& node = as<Node>(tree.memory)[startInnerNode + extraNodes / 2u];

			node = Node{ tree.memory, u32(left * sizeof(Node)), Node::INTERNAL_NODE_TYPE,
						 lightOffsets[right], lightOffsets.type(right), aabbDiag, materials };
		}
	}

	// Merge together the light nodes on the level featuring both light sources and internal nodes
	// (which is the level == height, e.g. 3; the computed nodes in the example are I5 and I6)
	// Take into account that up to one light has been merged already at the beginning (only ever L0)
	// This step will also be executed in case of a fully balanced tree
	const std::size_t startLight = (extraNodes % 2 == 0u) ? 0u : 1u;
	// Compute the starting internal node index by computing how many lights there could be in
	// this level and how many there are, which is given by the extraNodes (e.g. extraNodes is
	// 3 (I7, I8, I9), the level can take 2^3 == 8 lights, leaves 5 lights, of which L0 was already
	// accounted for; thus the internal node start index == 2 in the next-higher level, 4 in the tree)
	const std::size_t levelNodeCount = lightOffsets.light_count() - 2u * extraNodes - startLight;
	const std::size_t nodeCount = levelNodeCount / 2u;
	const std::size_t startNodeIndex = (1ull << height) - nodeCount - 1u;
	for(std::size_t i = 0u; i < nodeCount; ++i) {
		mAssert(startLight + 2u * i + 1u < lightOffsets.light_count());
		mAssert(startNodeIndex + i < tree.internalNodeCount);
		const std::size_t left = startLight + 2u * i;
		const std::size_t right = startLight + 2u * i + 1u;
		Node& node = as<Node>(tree.memory)[startNodeIndex + i];

		node = Node{ tree.memory, lightOffsets[left], lightOffsets.type(left),
			lightOffsets[right], lightOffsets.type(right), aabbDiag, materials };
	}

	// Now for the rest of the levels (ie. inner nodes, no more lights nowhere)
	if(height > 1u) {
		for(std::size_t level = height - 1u; level >= 1u; --level) {
			// Compute the number of nodes on the current and the level below
			const std::size_t nodes = (1ull << level);
			const std::size_t innerNode = nodes / 2u - 1u;
			// Accumulate for higher-up node
			for(std::size_t i = 0u; i < nodes / 2u; ++i) {
				mAssert(innerNode + i < tree.internalNodeCount);
				mAssert(nodes + 2u * i < tree.internalNodeCount);
				const std::size_t left = nodes - 1u + 2u * i;
				const std::size_t right = nodes - 1u + 2u * i + 1u;
				Node& node = as<Node>(tree.memory)[innerNode + i];

				node = Node{ tree.memory, u32(left * sizeof(Node)), Node::INTERNAL_NODE_TYPE,
							 u32(right * sizeof(Node)), Node::INTERNAL_NODE_TYPE, aabbDiag, materials };
			}
		}
	}

	// Last, set the root properties: guaranteed two lights
	tree.root.type = Node::INTERNAL_NODE_TYPE;
	tree.root.flux = tree.get_node(0u)->left.flux + tree.get_node(0u)->right.flux;
}

void fill_map(const std::vector<PositionalLights>& lights, HashMap<Device::CPU, PrimitiveHandle, u32>& map) {
	if(lights.size() == 0)
		return;
	// 'Height' is the 'lower' height of the tree, ie. on what height the first light node is
	int height = ei::ilog2(lights.size());
	// Calculate the number of additional internal nodes compared to a full tree of 'height'
	const u32 extraNodes = u32(lights.size()) - (1u << height);
	// From this we can infer how many lights are on the lower level
	const u32 lowerLightCount = u32(1u << height) - extraNodes;

	u32 i = 0u;
	// Special case: non-full light tree (not so special actually...)
	if(extraNodes > 0) {
		for(; i < lowerLightCount; ++i) {
			const auto& light = lights[i];
			if(light.primitive.instanceId != -1) {		// Hitable light source?
			// We can compute the code 'backwards': (2^(height + 1) - 1) - (n-1 - i)
				u32 code = ((1u << height) - 1) - (lowerLightCount - i - 1);
				// Shift to left-align the code
				code <<= (32 - height);
				map.insert(light.primitive, code);
			}
		}
		height += 1;
	}

	// Left-overs on the final tree level: count up the in-level index and shift
	for(; i < lights.size(); ++i) {
		const auto& light = lights[i];
		if(light.primitive.instanceId != -1) {		// Hitable light source?
			const u32 code = (i - lowerLightCount) << (32 - height);
			map.insert(light.primitive, code);
		}
	}
}


} // namespace

namespace lighttree_detail {

// Computes the number of nodes of a balanced tree with elems leaves
std::size_t get_num_internal_nodes(std::size_t elems) {
	if(elems <= 1u)
		return 0u;
	// Height of a balanced binary tree is log2(N) + 1
	std::size_t height = static_cast<std::size_t>(std::log2(elems));
	// Interior nodes are then 2^(height-1) - 1u
	std::size_t nodes = (1lu << height) - 1u;
	// Check if our tree is "filled" or if we have extra nodes on the bottom level
	return nodes + (elems - (1ull << height));
}

} // namespace lighttree_detail

using namespace lighttree_detail;

// Node constructors
LightSubTree::Node::Node(const char* base,
						 u32 leftOffset, u16 leftType,
						 u32 rightOffset, u16 rightType,
						 const ei::Vec3& aabbDiag,
						 const int* materials) :
	left{ get_flux(base + leftOffset, leftType, aabbDiag, materials), leftOffset, leftType },
	right{ rightType, rightOffset, get_flux(base + rightOffset, rightType, aabbDiag, materials) },
	center{ (get_center(base + leftOffset, leftType) + get_center(base + rightOffset, rightType)) / 2.0f }
{}


LightTree<Device::OPENGL>::LightTree(const std::vector<SmallLight>& smallLights, const std::vector<BigLight>& bigLights) {
	numSmallLights = uint32_t(smallLights.size());
	numBigLights = uint32_t(bigLights.size());
    // alloc array storage (increase size to 1 if 0)
	this->smallLights = Allocator<DEVICE>::alloc_array<SmallLight>(std::max<size_t>(numSmallLights, 1) );
	this->bigLights = Allocator<DEVICE>::alloc_array<BigLight>(std::max<size_t>(numBigLights, 1));
    // copy data
	if(numSmallLights)
		copy(this->smallLights, smallLights.data(), numSmallLights * sizeof(SmallLight));
	if(numBigLights)
		copy(this->bigLights, bigLights.data(), numBigLights * sizeof(BigLight));
}

/*LightTree<Device::OPENGL>::~LightTree() {
	if(smallLights != nullptr)
		gl::deleteBuffer(smallLights.id);
	if(bigLights != nullptr)
		gl::deleteBuffer(bigLights.id);
}*/

LightTreeBuilder::LightTreeBuilder() {}

LightTreeBuilder::~LightTreeBuilder() {
	unload<Device::CPU>();
	unload<Device::CUDA>();
	unload<Device::OPENGL>();
}

void LightTreeBuilder::build(std::vector<PositionalLights>&& posLights,
					  std::vector<DirectionalLight>&& dirLights,
					  const ei::Box& boundingBox,
					  const int* materials) {
	logInfo("[LightTreeBuilder::build] Start building light tree.");
	// Make sure the hashmap memory is allocated
	m_primToNodePath.resize(int(posLights.size()));

	unload<Device::CPU>();
	m_treeCpu = std::make_unique<LightTree<Device::CPU>>();
	m_primToNodePath.clear();
	m_treeCpu->primToNodePath = m_primToNodePath.acquire<Device::CPU>();
	//m_treeCpu->guide = &guide_flux;
	m_treeCpu->posGuide = false;

	if(posLights.size() == 0u && dirLights.size() == 0u) {
		// Shortcut if no lights are specified
		return;
	}

	// Create our spatial sorting to get a good tree
	// TODO: sort the directional lights by direction
	ei::Vec3 scale = boundingBox.max - boundingBox.min;
	std::sort(dirLights.begin(), dirLights.end(), [](const DirectionalLight& a, const DirectionalLight& b) {
		// TODO: better sorting scheme!
		constexpr ei::Vec3 BASE_DIRECTION{ 1.f, 0.f, 0.f };
		float distA = -ei::dot(a.direction, BASE_DIRECTION);
		float distB = -ei::dot(b.direction, BASE_DIRECTION);
		return distA < distB;
	});
	std::sort(posLights.begin(), posLights.end(), [&scale](const PositionalLights& a, const PositionalLights& b) {
		// Rescale and round the light centers to fall onto positive integers
		ei::UVec3 x = ei::UVec3(get_center(&a, u16(get_light_type(a))) * scale);
		ei::UVec3 y = ei::UVec3(get_center(&b, u16(get_light_type(b))) * scale);
		// Interleave packages of 16 bit, convert to inverse Gray code and compare.
		u64 codeA = math::get_hilbert_code(x.x >> 16u, x.y >> 16u, x.z >> 16u);
		u64 codeB = math::get_hilbert_code(y.x >> 16u, y.y >> 16u, y.z >> 16u);
		// If they are equal take the next 16 less significant bit.
		if(codeA == codeB) {
			codeA = math::get_hilbert_code(x.x & 0xffff, x.y & 0xffff, x.z & 0xffff);
			codeB = math::get_hilbert_code(y.x & 0xffff, y.y & 0xffff, y.z & 0xffff);
		}
		return codeA < codeB;
	});

	// Allocate enough space to fit all lights - this assumes that the tree will be balanced!
	// Node memory is easy since fixed size
	m_treeCpu->dirLights.internalNodeCount = static_cast<u32>(get_num_internal_nodes(dirLights.size()));
	m_treeCpu->posLights.internalNodeCount = static_cast<u32>(get_num_internal_nodes(posLights.size()));

	// Compute a sum table for the light offsets for positional lights,
	// nothing for directional ones (just create a compatible interface).
	LightOffset<DirectionalLight> dirLightOffsets(dirLights, m_treeCpu->dirLights.internalNodeCount * sizeof(LightSubTree::Node));
	LightOffset<PositionalLights> posLightOffsets(posLights, m_treeCpu->posLights.internalNodeCount * sizeof(LightSubTree::Node));

	std::size_t treeMemSize = sizeof(LightSubTree::Node) * (m_treeCpu->dirLights.internalNodeCount + m_treeCpu->posLights.internalNodeCount)
		+ dirLightOffsets.mem_size() + posLightOffsets.mem_size();
	m_treeMemory.resize(treeMemSize);
	m_treeMemory.mark_changed(Device::CPU);
	char* memory = m_treeMemory.acquire<Device::CPU>();
	// Set up the node pointers
	m_treeCpu->dirLights.memory = memory;
	m_treeCpu->posLights.memory = memory + dirLightOffsets.mem_size();

	// Copy the lights into the tree
	// Directional lights are easier, because they have a fixed size
	if(dirLights.size() > 0u) {
		std::memcpy(m_treeCpu->dirLights.memory + dirLightOffsets[0], dirLights.data(),
					dirLightOffsets.mem_size() - dirLightOffsets[0]);
	}
	// Positional lights are more difficult since we don't know the concrete size
	if(posLights.size() > 0u) {
		char* mem = m_treeCpu->posLights.memory + posLightOffsets[0];
		for(const PositionalLights& light : posLights) {
			std::visit(overloaded{
				[&mem](const AreaLightTriangleDesc& desc) {
					auto* dst = as<AreaLightTriangle<Device::CPU>>(mem);
					mem += sizeof(*dst);
					*dst = desc;
				},
				[&mem](const AreaLightQuadDesc& desc) {
					auto* dst = as<AreaLightQuad<Device::CPU>>(mem);
					mem += sizeof(*dst);
					*dst = desc;
				},
				[&mem](const AreaLightSphereDesc& desc) {
					auto* dst = as<AreaLightSphere<Device::CPU>>(mem);
					mem += sizeof(*dst);
					*dst = desc;
				},
				[&mem](const auto& desc) {
					std::memcpy(mem, &desc, sizeof(desc));
					mem += sizeof(desc);
				}
			}, light.light);
		}
	}

	// Now we gotta construct the proper nodes by recursively merging them together
	create_light_tree(dirLightOffsets, m_treeCpu->dirLights, scale, materials);
	create_light_tree(posLightOffsets, m_treeCpu->posLights, scale, materials);
	fill_map(posLights, m_treeCpu->primToNodePath);
	m_primToNodePath.mark_changed(Device::CPU);
	m_treeCuda.reset();
	m_treeOpengl.reset();

	m_lightCount = static_cast<u32>(posLights.size() + dirLights.size());
}

//__device__ GuideFunction cudaGuideFlux = guide_flux;
//__device__ GuideFunction cudaGuideFluxPos = guide_flux_pos;
//__device__ float pi_gpu = 0;

std::size_t LightTreeBuilder::descriptor_size() const noexcept {
	return m_treeMemory.size() + m_primToNodePath.size()
		+ (m_envLight != nullptr ? m_envLight->descriptor_size() : 0u);
}

template < Device dev >
void LightTreeBuilder::synchronize(const ei::Box& sceneBounds) {
	// Background is no longer outdated
	m_backgroundDirty = false;

	// We never change the light tree on another device aside from the CPU
	if(dev == Device::CUDA && !m_treeCuda) {
		if(!m_treeCpu)
			throw std::runtime_error("[LightTreeBuilder::synchronize] There is no source for light-tree synchronization!");
		// Easiest way to synchronize this complicated type (espacially, if the size changed)
		// is simply to start over.
		m_treeCuda = std::make_unique<LightTree<Device::CUDA>>();
		m_treeCuda->primToNodePath = m_primToNodePath.acquire<Device::CUDA>(); // Includes synchronization
		m_treeCuda->posGuide = false;
		//void* symbolAddress;
		//cudaGetSymbolAddress(&symbolAddress, guide_flux);
		//cudaMemcpyFromSymbol(&symbolAddress, &pi_gpu, sizeof(float));
		//(cudaMemcpyFromSymbol(&m_treeCuda->guide, cudaGuideFlux, sizeof(GuideFunction)));
		//cudaMemcpy(&m_treeCuda->guide, &cudaGuideFlux, sizeof(GuideFunction), cudaMemcpyDeviceToHost);

		// Equalize bookkeeping of subtrees
		char* lightMem = m_treeMemory.template is_resident<Device::CPU>() ? m_treeMemory.template acquire<Device::CUDA>() : nullptr;
		m_treeCuda->dirLights = m_treeCpu->dirLights;
		m_treeCuda->dirLights.memory = lightMem;
		m_treeCuda->posLights = m_treeCpu->posLights;
		m_treeCuda->posLights.memory = lightMem + (m_treeCpu->posLights.memory - m_treeCpu->dirLights.memory);
	} else if(dev == Device::OPENGL) {
		if(!m_treeCpu)
			throw std::runtime_error("[LightTreeBuilder::synchronize] There is no source for light-tree synchronization!");

		std::vector<LightTree<Device::OPENGL>::SmallLight> smallLights;
		std::vector<LightTree<Device::OPENGL>::BigLight> bigLights;

        // traverse tree
        auto addLight = [&](u32 offset, u16 type, LightSubTree& subTree) {
			switch(LightType(type)) {
			case LightType::POINT_LIGHT: {
				auto light = reinterpret_cast<const PointLight*>(subTree.memory + offset);
				auto& dst = smallLights.emplace_back();
				dst.type = uint32_t(LightType::POINT_LIGHT);
			    dst.intensity = light->intensity;
				dst.position = light->position;
			} break;
			case LightType::SPOT_LIGHT: {
				auto light = reinterpret_cast<const SpotLight*>(subTree.memory + offset);
				auto& dst = smallLights.emplace_back();
				dst.type = uint32_t(LightType::SPOT_LIGHT);
				dst.intensity = light->intensity;
				dst.position = light->position;
				dst.direction = light->direction;
				dst.cosFalloffStart = light->cosFalloffStart;
				dst.cosThetaMax = light->cosThetaMax;
			} break;
			case LightType::AREA_LIGHT_TRIANGLE: {
				auto light = reinterpret_cast<const AreaLightTriangle<Device::CPU>*>(subTree.memory + offset);
				auto& dst = bigLights.emplace_back();
				dst.material = light->material;
				dst.numPoints = 3;
				dst.pos = light->posV[0];
				dst.v1 = light->posV[1];
				dst.v2 = light->posV[2];
                // test upload point light
				//auto& dst2 = smallLights.emplace_back();
                //dst2.type = uint32_t(LightType::POINT_LIGHT);
				//dst2.intensity = ei::Vec3(1.0f);
				//dst2.position = dst.pos + (dst.v1 + dst.v2) * 0.5f;
   			} break;
			case LightType::AREA_LIGHT_QUAD: {
				auto light = reinterpret_cast<const AreaLightQuad<Device::CPU>*>(subTree.memory + offset);
				auto& dst = bigLights.emplace_back();
				dst.material = light->material;
				dst.numPoints = 4;
				dst.pos = light->posV[0];
				dst.v3 = light->posV[1];
				dst.v1 = light->posV[2];
				dst.v2 = light->posV[3] + dst.v1 + dst.v3;
                // test upload point light
				//auto& dst2 = smallLights.emplace_back();
				//dst2.type = uint32_t(LightType::POINT_LIGHT);
				//dst2.intensity = ei::Vec3(1.0f);
				//dst2.position = dst.pos + (dst.v1 + dst.v2 + dst.v3) * 0.33f;
			} break;
			case LightType::AREA_LIGHT_SPHERE: {
				auto light = reinterpret_cast<const AreaLightSphere<Device::CPU>*>(subTree.memory + offset);
				auto& dst = smallLights.emplace_back();
				dst.type = uint32_t(LightType::AREA_LIGHT_SPHERE);
				dst.position = light->position;
				dst.radius = light->radius;
				dst.material = light->material;
			} break;
			case LightType::DIRECTIONAL_LIGHT: {
				auto light = reinterpret_cast<const DirectionalLight*>(subTree.memory + offset);
				auto& dst = smallLights.emplace_back();
				dst.type = uint32_t(LightType::DIRECTIONAL_LIGHT);
				dst.intensity = light->irradiance;
				dst.direction = light->direction;
			} break;
			case LightType::ENVMAP_LIGHT:
			default:
				break;
            }
		};

        // helper to extract lights from the tree
		std::function<void(LightSubTree::Node*, LightSubTree&)> extractNodes;
        extractNodes = [&](LightSubTree::Node* node, LightSubTree& subTree) {
            if(node->left.is_light()) {
				addLight(node->left.offset, node->left.type, subTree);
            } else {
				extractNodes(m_treeCpu->posLights.get_node(node->left.offset), subTree);
            }
            if(node->right.is_light()) {
				addLight(node->right.offset, node->right.type, subTree);
            } else {
				extractNodes(m_treeCpu->posLights.get_node(node->right.offset), subTree);
            }
		};

        // do extraction
		if(m_treeCpu->dirLights.internalNodeCount >= 1)
			extractNodes(m_treeCpu->dirLights.get_node(0), m_treeCpu->dirLights);
		else if(m_treeCpu->dirLights.lightCount == 1)
			addLight(0, m_treeCpu->dirLights.root.type, m_treeCpu->dirLights);

		if(m_treeCpu->posLights.internalNodeCount >= 1)
			extractNodes(m_treeCpu->posLights.get_node(0), m_treeCpu->posLights);
		else if(m_treeCpu->posLights.lightCount == 1)
			addLight(0, m_treeCpu->posLights.root.type, m_treeCpu->posLights);

        // upload buffers
		unload<Device::OPENGL>();
		m_treeOpengl = std::make_unique<LightTree<Device::OPENGL>>(smallLights, bigLights);
	}

	// The background can always be outdated
	mAssertMsg(m_envLight != nullptr, "Background should always be set!");
	if(dev == Device::CUDA)
		m_treeCuda->background = m_envLight->acquire_const<Device::CUDA>(sceneBounds);
	else
		m_treeCpu->background = m_envLight->acquire_const<Device::CPU>(sceneBounds);
	// TODO: backsync? Would need another hashmap for texture mapping.
}

/*template <>
GuideFunction LightTreeBuilder::get_guide_fptr<Device::CPU>(bool posGuide) {
	if(posGuide)
		return &guide_flux_pos;
	else
		return &guide_flux;
}

template <>
GuideFunction LightTreeBuilder::get_guide_fptr<Device::CUDA>(bool posGuide) {
	GuideFunction fptr;
	if(posGuide)
		cudaMemcpyFromSymbol(&fptr, cudaGuideFluxPos, sizeof(GuideFunction));
	else
		cudaMemcpyFromSymbol(&fptr, cudaGuideFlux, sizeof(GuideFunction));
	return fptr;
}*/


void LightTreeBuilder::update_media_cpu(const SceneDescriptor<Device::CPU>& scene) {
	char* currLightMem = m_treeCpu->posLights.memory;
	mAssert(m_treeCpu->posLights.lightCount == 0 || currLightMem != nullptr);
	mAssert(m_treeCpu->posLights.lightCount < std::numeric_limits<u32>::max());
	if(m_treeCpu->posLights.lightCount == 0)
		return;
	if(m_treeCpu->posLights.lightCount == 1) {
		// Special case: root stores type
		mAssert(m_treeCpu->posLights.root.type < static_cast<u16>(LightType::NUM_LIGHTS));
		set_light_medium(currLightMem, static_cast<LightType>(m_treeCpu->posLights.root.type), scene);
		return;
	}
	// TODO: do we need to determine the medium for area lights (given by the geometry)
	const u32 NODE_COUNT = static_cast<u32>(m_treeCpu->posLights.internalNodeCount);
	// Walk backwards in the nodes to iterate over lights, but leave out the odd one (if it exists)
	const u32 exclusiveLightNodes = static_cast<u32>(m_treeCpu->posLights.lightCount / 2u);
	for(u32 i = 1u; i <= exclusiveLightNodes; ++i) {
		const LightSubTree::Node& node = *m_treeCpu->posLights.get_node((NODE_COUNT - i) * static_cast<u32>(sizeof(LightSubTree::Node)));
		mAssert(node.left.type < static_cast<u16>(LightType::NUM_LIGHTS));
		mAssert(node.right.type < static_cast<u16>(LightType::NUM_LIGHTS));
		set_light_medium(&currLightMem[node.left.offset], static_cast<LightType>(node.left.type), scene);
		set_light_medium(&currLightMem[node.right.offset], static_cast<LightType>(node.right.type), scene);
	}
	if(exclusiveLightNodes * 2u < m_treeCpu->posLights.lightCount) {
		// One extra light
		const LightSubTree::Node& node = *m_treeCpu->posLights.get_node((NODE_COUNT - exclusiveLightNodes - 1u) * sizeof(LightSubTree::Node));
		mAssert(node.right.type < static_cast<u16>(LightType::NUM_LIGHTS));
		set_light_medium(&currLightMem[node.right.offset], static_cast<LightType>(node.right.type), scene);
	}
}

template < Device dev >
void LightTreeBuilder::update_media(const SceneDescriptor<dev>& scene) {
	this->synchronize<dev>(scene.aabb);
	if constexpr(dev == Device::CPU)
		this->update_media_cpu(scene);
	else if constexpr(dev == Device::CUDA)
		update_media_cuda(scene, m_treeCuda->posLights);
	else if constexpr(dev == Device::OPENGL)
		; // TODO
	else
		mAssert(false);
}

template < Device dev >
void LightTreeBuilder::unload() {
	m_treeMemory.unload<dev>();
	m_primToNodePath.unload<dev>();
	switch(dev) {
	case Device::CPU: m_treeCpu.reset(); break;
	case Device::CUDA: m_treeCuda.reset(); break;
	case Device::OPENGL: if(m_treeOpengl) {
		gl::deleteBuffer(m_treeOpengl->smallLights.id);
		gl::deleteBuffer(m_treeOpengl->bigLights.id);
		m_treeOpengl.reset();
	} break;
	}
	// TODO: unload envmap handle
}

template struct DeviceManagerConcept<scene::lights::LightTreeBuilder>;

template void LightTreeBuilder::synchronize<Device::CPU>(const ei::Box&);
template void LightTreeBuilder::synchronize<Device::CUDA>(const ei::Box&);
template void LightTreeBuilder::synchronize<Device::OPENGL>(const ei::Box&);
template void LightTreeBuilder::unload<Device::CPU>();
template void LightTreeBuilder::unload<Device::CUDA>();
template void LightTreeBuilder::unload<Device::OPENGL>();
template void LightTreeBuilder::update_media<Device::CPU>(const SceneDescriptor<Device::CPU>& descriptor);
template void LightTreeBuilder::update_media<Device::CUDA>(const SceneDescriptor<Device::CUDA>& descriptor);
template void LightTreeBuilder::update_media<Device::OPENGL>(const SceneDescriptor<Device::OPENGL>& descriptor);

}}} // namespace mufflon::scene::lights