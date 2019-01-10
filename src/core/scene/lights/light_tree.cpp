#include "light_tree.hpp"
#include "ei/3dtypes.hpp"
#include "util/assert.hpp"
#include "core/memory/allocator.hpp"
#include "core/cuda/error.hpp"
#include "core/math/sfcurves.hpp"
#include "core/scene/descriptors.hpp"
#include "core/scene/accel_structs/intersection.hpp"
#include "core/scene/materials/material.hpp"
#include "core/scene/lights/light_medium.hpp"
#include "background.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>

namespace mufflon { namespace scene { namespace lights {

namespace {

float get_flux(const void* light, u16 type, const ei::Vec3& aabbDiag) {
	switch(type) {
		case u16(LightType::POINT_LIGHT): return ei::sum(get_flux(*as<PointLight>(light)));
		case u16(LightType::SPOT_LIGHT): return ei::sum(get_flux(*as<SpotLight>(light)));
		case u16(LightType::AREA_LIGHT_TRIANGLE): return ei::sum(get_flux(*as<AreaLightTriangle<Device::CPU>>(light)));
		case u16(LightType::AREA_LIGHT_QUAD): return ei::sum(get_flux(*as<AreaLightQuad<Device::CPU>>(light)));
		case u16(LightType::AREA_LIGHT_SPHERE): return ei::sum(get_flux(*as<AreaLightSphere<Device::CPU>>(light)));
		case u16(LightType::DIRECTIONAL_LIGHT): return ei::sum(get_flux(*as<DirectionalLight>(light), aabbDiag));
		case u16(LightType::ENVMAP_LIGHT): return ei::sum(as<BackgroundDesc<Device::CPU>>(light)->flux);
		case LightSubTree::Node::INTERNAL_NODE_TYPE: return as<LightSubTree::Node>(light)->left.flux + as<LightSubTree::Node>(light)->right.flux;
	}
	return 0.0f;
}

// Computes the offset of the i-th point light - positional ones need a sum table!
// The offsets are relative to the trees node-memory.
template < class LT >
class LightOffset {
public:
	LightOffset(const std::vector<LT>& lights, std::size_t globalOffset) :
		m_lightCount{ lights.size() },
		m_globalOffset{ globalOffset }
	{}
	
	constexpr u32 operator[](std::size_t lightIndex) const noexcept {
		return static_cast<u32>(sizeof(LT) * lightIndex);
	}

	std::size_t light_count() const noexcept {
		return m_lightCount;
	}
	std::size_t mem_size() const noexcept {
		return m_lightCount * sizeof(DirectionalLight);
	}

	u16 type(std::size_t lightIndex) const noexcept {
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
					[&prevOffset](const PointLight& light) constexpr { LightRef res{prevOffset, u16(LightType::POINT_LIGHT)}; prevOffset += sizeof(PointLight); return res; },
					[&prevOffset](const SpotLight& light) constexpr { LightRef res{prevOffset, u16(LightType::SPOT_LIGHT)}; prevOffset += sizeof(SpotLight); return res; },
					[&prevOffset](const AreaLightTriangleDesc& light) constexpr { LightRef res{prevOffset, u16(LightType::AREA_LIGHT_TRIANGLE)}; prevOffset += sizeof(AreaLightTriangle<Device::CPU>); return res; },
					[&prevOffset](const AreaLightQuadDesc& light) constexpr { LightRef res{prevOffset, u16(LightType::AREA_LIGHT_QUAD)}; prevOffset += sizeof(AreaLightQuad<Device::CPU>); return res; },
					[&prevOffset](const AreaLightSphereDesc& light) constexpr { LightRef res{prevOffset, u16(LightType::AREA_LIGHT_SPHERE)}; prevOffset += sizeof(AreaLightSphere<Device::CPU>); return res; }
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
					   const ei::Vec3& aabbDiag) {
	using Node = LightSubTree::Node;

	tree.lightCount = lightOffsets.light_count();
	if(lightOffsets.light_count() == 0u)
		return;

	// Only one light -> no actual tree, only light
	if(lightOffsets.light_count() == 1u) {
		tree.root.type = static_cast<u16>(lightOffsets.type(0));
		tree.root.flux = get_flux(tree.memory + lightOffsets[0], lightOffsets.type(0), aabbDiag);
		tree.root.center = get_center(tree.memory + lightOffsets[0], lightOffsets.type(0));
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
			mAssert(startNode + i < get_num_internal_nodes(lightOffsets.light_count()));
			const std::size_t left = startLight + 2u * i;
			const std::size_t right = startLight + 2u * i + 1u;
			Node& interiorNode = as<Node>(tree.memory)[startNode + i];

			interiorNode = Node{ tree.memory, lightOffsets[left], lightOffsets.type(left),
								 lightOffsets[right], lightOffsets.type(right), aabbDiag };
		}

		// To ensure we can later uniformly create all remaining internal nodes, we also form
		// the "second-lowest" layer of internal nodes (e.g. I3, which may occur multiple times,
		// and I4, of which there will at most be one kind)
		const std::size_t startInnerNode = (1lu << (height - 1u)) - 1u;
		for(std::size_t i = 0u; i < extraNodes / 2u; ++i) {
			mAssert(startNode + 2u*i + 1u < get_num_internal_nodes(lightOffsets.light_count()));
			const std::size_t left = startNode + 2u * i;
			const std::size_t right = startNode + 2u * i + 1u;
			Node& node = as<Node>(tree.memory)[startInnerNode + i];

			node = Node{ tree.memory, u32(left * sizeof(Node)), Node::INTERNAL_NODE_TYPE,
						 u32(right * sizeof(Node)), Node::INTERNAL_NODE_TYPE, aabbDiag };
		}

		// Create the one possible internal node that has another internal node as left child
		// and a light source as right child (I4)
		if(extraNodes % 2 != 0u) {
			// One interior leftover; must be first light and last internal node
			mAssert(startNode + extraNodes - 1u < get_num_internal_nodes(lightOffsets.light_count()));
			const std::size_t left = startNode + extraNodes - 1u;
			const std::size_t right = 0;
			Node& node = as<Node>(tree.memory)[startInnerNode + extraNodes / 2u];

			node = Node{ tree.memory, u32(left * sizeof(Node)), Node::INTERNAL_NODE_TYPE,
						 lightOffsets[right], lightOffsets.type(right), aabbDiag };
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
		mAssert(startNodeIndex + i < get_num_internal_nodes(lightOffsets.light_count()));
		const std::size_t left = startLight + 2u * i;
		const std::size_t right = startLight + 2u * i + 1u;
		Node& node = as<Node>(tree.memory)[startNodeIndex + i];

		node = Node{ tree.memory, lightOffsets[left], lightOffsets.type(left),
			lightOffsets[right], lightOffsets.type(right), aabbDiag };
	}

	// Now for the rest of the levels (ie. inner nodes, no more lights nowhere)
	if(height > 1u) {
		for(std::size_t level = height - 1u; level >= 1u; --level) {
			// Compute the number of nodes on the current and the level below
			const std::size_t nodes = (1ull << level);
			const std::size_t innerNode = nodes / 2u - 1u;
			// Accumulate for higher-up node
			for(std::size_t i = 0u; i < nodes / 2u; ++i) {
				mAssert(innerNode + i < get_num_internal_nodes(lightOffsets.light_count()));
				mAssert(nodes + 2u * i < get_num_internal_nodes(lightOffsets.light_count()));
				const std::size_t left = nodes - 1u + 2u * i;
				const std::size_t right = nodes - 1u + 2u * i + 1u;
				Node& node = as<Node>(tree.memory)[innerNode + i];

				node = Node{ tree.memory, u32(left * sizeof(Node)), Node::INTERNAL_NODE_TYPE,
							 u32(right * sizeof(Node)), Node::INTERNAL_NODE_TYPE, aabbDiag };
			}
		}
	}

	// Last, set the root properties: guaranteed two lights
	tree.root.type = Node::INTERNAL_NODE_TYPE;
	tree.root.flux = tree.get_node(0u)->left.flux + tree.get_node(0u)->right.flux;
	tree.root.center = tree.get_node(0u)->center;
}

void fill_map(const std::vector<PositionalLights>& lights, HashMap<Device::CPU, PrimitiveHandle, u32>& map) {
	int height = ei::ilog2(lights.size());
	u32 extraNodes = u32(lights.size()) - (1u << height);
	u32 lvlOff = extraNodes * 2;		// Index of first node on the height-1 level (all nodes < lvlOff are in level height)
	if(extraNodes > 0) ++height;
	u32 i = 0;
	for(const auto& light : lights) {
		if(light.primitive != ~0u) {		// Hitable light source?
			u32 code = (i <= lvlOff) ? i : (i-lvlOff)*2+lvlOff;
			// Append zero -> most significant bit is the root branch
			code <<= 32 - height;
			map.insert(light.primitive, code);
		}
		++i;
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
						 const ei::Vec3& bounds) :
	left{ get_flux(base + leftOffset, leftType, bounds), leftOffset, leftType },
	right{ rightType, rightOffset, get_flux(base + rightOffset, rightType, bounds) },
	center{ (get_center(base + leftOffset, leftType) + get_center(base + rightOffset, rightType)) / 2.0f }
{}


LightTreeBuilder::LightTreeBuilder() :
	m_dirty()
{}

LightTreeBuilder::~LightTreeBuilder() {
	unload<Device::CPU>();
	unload<Device::CUDA>();
}

void LightTreeBuilder::build(std::vector<PositionalLights>&& posLights,
					  std::vector<DirectionalLight>&& dirLights,
					  const ei::Box& boundingBox) {
	// Make sure the hashmap memory is allocated
	m_primToNodePath.resize(int(posLights.size()));

	unload<Device::CPU>();
	m_treeCpu = std::make_unique<LightTree<Device::CPU>>();
	m_treeCpu->primToNodePath = m_primToNodePath.acquire<Device::CPU>();

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
	std::size_t dirNodes = get_num_internal_nodes(dirLights.size());
	std::size_t posNodes = get_num_internal_nodes(posLights.size());

	// Compute a sum table for the light offsets for positional lights,
	// nothing for directional ones (just create a compatible interface).
	LightOffset<DirectionalLight> dirLightOffsets(dirLights, dirNodes * sizeof(LightSubTree::Node));
	LightOffset<PositionalLights> posLightOffsets(posLights, posNodes * sizeof(LightSubTree::Node));

	std::size_t treeMemSize = sizeof(LightSubTree::Node) * (dirNodes + posNodes)
		+ dirLightOffsets.mem_size() + posLightOffsets.mem_size();
	m_treeMemory.resize(treeMemSize);
	char* memory = m_treeMemory.acquire<Device::CPU>();
	// Set up the node pointers
	m_treeCpu->dirLights.memory = memory;
	m_treeCpu->posLights.memory = memory + dirLightOffsets[0] + dirLightOffsets.mem_size();

	// Copy the lights into the tree
	// Directional lights are easier, because they have a fixed size
	if(dirLights.size() > 0.0f) {
		std::memcpy(m_treeCpu->dirLights.memory + dirLightOffsets[0], dirLights.data(), dirLightOffsets.mem_size());
	}
	// Positional lights are more difficult since we don't know the concrete size
	if(posLights.size() > 0.0f) {
		static_assert(sizeof(AreaLightQuad<Device::CPU>) == 96+16);
		char* mem = m_treeCpu->posLights.memory + posLightOffsets[0];
		std::size_t i = 0u;
		for(const PositionalLights& light : posLights) {
			if(i == 509) {
				const float v = 5;
			}
			++i;
			std::visit(overloaded{
				[&mem,this](const AreaLightTriangleDesc& desc) {
					auto* dst = as<AreaLightTriangle<Device::CPU>>(mem);
					mem += sizeof(*dst);
					*dst = desc;
					// Remember texture for synchronization
					m_textureMap.emplace(dst->radianceTex, desc.radianceTex);
				},
				[&mem,this](const AreaLightQuadDesc& desc) {
					auto* dst = as<AreaLightQuad<Device::CPU>>(mem);
					mem += sizeof(*dst);
					*dst = desc;
					// Remember texture for synchronization
					m_textureMap.emplace(dst->radianceTex, desc.radianceTex);
				},
				[&mem,this](const AreaLightSphereDesc& desc) {
					auto* dst = as<AreaLightSphere<Device::CPU>>(mem);
					mem += sizeof(*dst);
					*dst = desc;
					// Remember texture for synchronization
					m_textureMap.emplace(dst->radianceTex, desc.radianceTex);
				},
				[&mem](const auto& desc) {
					std::memcpy(mem, &desc, sizeof(desc));
					mem += sizeof(desc);
				}
			}, light.light);
		}
	}

	// Now we gotta construct the proper nodes by recursively merging them together
	create_light_tree(dirLightOffsets, m_treeCpu->dirLights, scale);
	create_light_tree(posLightOffsets, m_treeCpu->posLights, scale);
	fill_map(posLights, m_treeCpu->primToNodePath);
	m_dirty.mark_changed(Device::CPU);
}

void LightTreeBuilder::remap_textures(const char* cpuMem, u32 offset, u16 type, char* cudaMem) {
	switch(type) {
		case LightSubTree::Node::INTERNAL_NODE_TYPE: {
			cudaMemcpy(cudaMem + offset, cpuMem + offset, sizeof(LightSubTree::Node), cudaMemcpyDefault);
			// Recursive implementation necessary, because the type is stored at the parents and not known to the node itself
			const auto* node = as<LightSubTree::Node>(cpuMem + offset);
			remap_textures(cpuMem, node->left.offset, node->left.type, cudaMem);
			remap_textures(cpuMem, node->right.offset, node->right.type, cudaMem);
		} break;
		case u16(LightType::AREA_LIGHT_TRIANGLE): {
			const auto* light = as<AreaLightTriangle<Device::CPU>>(cpuMem + offset);
			AreaLightTriangle<Device::CUDA> cudaLight;
			for(int i = 0; i < 3; ++i) {
				cudaLight.points[i] = light->points[i];
				cudaLight.uv[i] = light->uv[i];
			}
			cudaLight.scale = light->scale;
			cudaLight.radianceTex = m_textureMap.find(light->radianceTex)->second->acquire_const<Device::CUDA>();
			cudaMemcpy(cudaMem + offset, &cudaLight, sizeof(cudaLight), cudaMemcpyDefault);
		} break;
		case u16(LightType::AREA_LIGHT_QUAD): {
			const auto* light = as<AreaLightQuad<Device::CPU>>(cpuMem + offset);
			AreaLightQuad<Device::CUDA> cudaLight;
			for(int i = 0; i < 4; ++i) {
				cudaLight.points[i] = light->points[i];
				cudaLight.uv[i] = light->uv[i];
			}
			cudaLight.scale = light->scale;
			cudaLight.radianceTex = m_textureMap.find(light->radianceTex)->second->acquire_const<Device::CUDA>();
			cudaMemcpy(cudaMem + offset, &cudaLight, sizeof(cudaLight), cudaMemcpyDefault);
		} break;
		case u16(LightType::AREA_LIGHT_SPHERE): {
			const auto* light = as<AreaLightSphere<Device::CPU>>(cpuMem + offset);
			AreaLightSphere<Device::CUDA> cudaLight;
			cudaLight.position = light->position;
			cudaLight.radius = light->radius;
			cudaLight.scale = light->scale;
			cudaLight.radianceTex = m_textureMap.find(light->radianceTex)->second->acquire_const<Device::CUDA>();
			cudaMemcpy(cudaMem + offset, &cudaLight, sizeof(cudaLight), cudaMemcpyDefault);
		} break;
		default:; // Other light type - nothing to do
	}
}

template < Device dev >
void LightTreeBuilder::synchronize() {
	if(dev == Device::CUDA && m_dirty.needs_sync(dev)) {
		if(!m_treeCpu)
			throw std::runtime_error("[LightTreeBuilder::synchronize] There is no source for light-tree synchronization!");
		// Easiest way to synchronize this complicated type (espacially, if the size changed)
		// is simply to start over.
		m_treeCuda = std::make_unique<LightTree<Device::CUDA>>();
		m_treeCuda->primToNodePath = m_primToNodePath.acquire<Device::CUDA>(); // Includes synchronization

		// Equalize bookkeeping of subtrees
		m_treeCuda->dirLights = m_treeCpu->dirLights;
		m_treeCuda->dirLights.memory = m_treeMemory.acquire<Device::CUDA>(false);
		m_treeCuda->posLights = m_treeCpu->posLights;
		m_treeCuda->posLights.memory = m_treeCuda->dirLights.memory + (m_treeCpu->posLights.memory - m_treeCpu->dirLights.memory);

		// Remap the environment map
		if(m_envLight)
			m_treeCuda->background = m_envLight->acquire_const<Device::CUDA>();
		else // Default envmap is black
			m_treeCuda->background = Background::black().acquire_const<Device::CUDA>();

		// Replace all texture handles inside the tree's data and
		// synchronize all the remaining tree memory.
		remap_textures(m_treeCpu->posLights.memory, 0, m_treeCpu->posLights.root.type, m_treeCuda->posLights.memory);

		m_dirty.mark_synced(dev);
	} else {
		// The background can always be outdated
		if(m_envLight)
			m_treeCpu->background = m_envLight->acquire_const<Device::CPU>();
		else // Default envmap is black
			m_treeCpu->background = Background::black().acquire_const<Device::CPU>();
		// TODO: backsync? Would need another hashmap for texture mapping.
	}
}

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
	const u32 NODE_COUNT = static_cast<u32>(get_num_internal_nodes(m_treeCpu->posLights.lightCount));
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
	this->synchronize<dev>();
	if constexpr(dev == Device::CPU)
		this->update_media_cpu(scene);
	else if constexpr(dev == Device::CUDA)
		update_media_cuda(scene, m_treeCuda->posLights);
	else
		default: mAssert(false); return;
}

template < Device dev >
void LightTreeBuilder::unload() {
	m_treeMemory.unload<dev>();
	m_primToNodePath.unload<dev>();
	if(dev == Device::CPU && m_treeCpu) {
		m_treeCpu = nullptr;
	} else if(m_treeCuda) {
		m_treeCuda = nullptr;
	}
	// TODO: unload envmap handle
}

template void LightTreeBuilder::synchronize<Device::CPU>();
template void LightTreeBuilder::synchronize<Device::CUDA>();
//template void LightTreeBuilder::synchronize<Device::OPENGL>();
template void LightTreeBuilder::unload<Device::CPU>();
template void LightTreeBuilder::unload<Device::CUDA>();
//template void LightTreeBuilder::unload<Device::OPENGL>();
template void LightTreeBuilder::update_media<Device::CPU>(const SceneDescriptor<Device::CPU>& descriptor);
template void LightTreeBuilder::update_media<Device::CUDA>(const SceneDescriptor<Device::CUDA>& descriptor);
//template void LightTreeBuilder::update_media<Device::OPENGL>(const SceneDescriptor<Device::OPENGL>& descriptor);

}}} // namespace mufflon::scene::lights