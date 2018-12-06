#include "light_tree.hpp"
#include "ei/3dtypes.hpp"
#include "util/assert.hpp"
#include "core/memory/allocator.hpp"
#include "core/cuda/error.hpp"
#include "core/math/sfcurves.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>

namespace mufflon { namespace scene { namespace lights {

namespace {

// Computes the number of nodes of a balanced tree with elems leaves
std::size_t get_num_internal_nodes(std::size_t elems) {
	if(elems <= 1u)
		return 0u;
	// Height of a balanced binary tree is log2(N) + 1
	std::size_t height = static_cast<std::size_t>(std::log2(elems));
	// Interior nodes are then 2^(height-1) - 1u
	std::size_t nodes = static_cast<std::size_t>(std::pow(2u, height)) - 1u;
	// Check if our tree is "filled" or if we have extra nodes on the bottom level
	return nodes + (elems - static_cast<std::size_t>(std::pow(2u, height)));
}



float get_flux(const void* light, u16 type, const ei::Vec3& aabbDiag) {
	switch(type) {
		case u16(LightType::POINT_LIGHT): return ei::sum(get_flux(*as<PointLight>(light)));
		case u16(LightType::SPOT_LIGHT): return ei::sum(get_flux(*as<SpotLight>(light)));
		case u16(LightType::AREA_LIGHT_TRIANGLE): return ei::sum(get_flux(*as<AreaLightTriangle<Device::CPU>>(light)));
		case u16(LightType::AREA_LIGHT_QUAD): return ei::sum(get_flux(*as<AreaLightQuad<Device::CPU>>(light)));
		case u16(LightType::AREA_LIGHT_SPHERE): return ei::sum(get_flux(*as<AreaLightSphere<Device::CPU>>(light)));
		case u16(LightType::DIRECTIONAL_LIGHT): return ei::sum(get_flux(*as<DirectionalLight>(light), aabbDiag));
		case u16(LightType::ENVMAP_LIGHT): return ei::sum(get_flux(*as<EnvMapLight<Device::CPU>>(light)));
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

	// Correctly set the internal nodes
	// Start with nodes that form the last (incomplete!) level
	std::size_t height = static_cast<std::size_t>(std::log2(lightOffsets.light_count()));
	mAssert(height > 0u);
	std::size_t extraNodes = lightOffsets.light_count() - static_cast<std::size_t>(std::pow(2u, height));
	if(extraNodes > 0u) {
		// Compute starting positions for internal nodes and lights
		std::size_t startNode = static_cast<std::size_t>(std::pow(2u, height)) - 1u;
		std::size_t startLight = lightOffsets.light_count() - 2u * extraNodes;
		// "Merge" together two nodes
		for(std::size_t i = 0u; i < extraNodes; ++i) {
			mAssert(startLight + 2u * i + 1u < lightOffsets.light_count());
			mAssert(startNode + i < get_num_internal_nodes(lightOffsets.light_count()));
			std::size_t left = startLight + 2u * i;
			std::size_t right = startLight + 2u * i;
			Node& interiorNode = as<Node>(tree.memory)[startNode + i];

			interiorNode = Node{ tree.memory, lightOffsets[left], lightOffsets.type(left),
								 lightOffsets[right], lightOffsets.type(right), aabbDiag };
		}

		// Also merge together inner nodes for last level!
		// Two cases: Merge two interior nodes each or (for last one) merge interior with light
		std::size_t startInnerNode = static_cast<std::size_t>(std::pow(2u, height - 1u)) - 1u;
		for(std::size_t i = 0u; i < extraNodes / 2u; ++i) {
			mAssert(startNode + 2u*i + 1u < get_num_internal_nodes(lightOffsets.light_count()));
			std::size_t left = startNode + 2u * i;
			std::size_t right = startNode + 2u * i + 1u;
			Node& node = as<Node>(tree.memory)[startInnerNode + i];

			node = Node{ tree.memory, u32(left * sizeof(Node)), Node::INTERNAL_NODE_TYPE,
						 u32(right * sizeof(Node)), Node::INTERNAL_NODE_TYPE, aabbDiag };
		}
		if(extraNodes % 2 != 0u) {
			// One interior leftover; must be very first light
			mAssert(startNode + extraNodes - 1u < get_num_internal_nodes(lightOffsets.light_count()));
			std::size_t left = startNode + extraNodes - 1u;
			std::size_t right = 0;
			Node& node = as<Node>(tree.memory)[startInnerNode + extraNodes / 2u];

			node = Node{ tree.memory, u32(left * sizeof(Node)), Node::INTERNAL_NODE_TYPE,
						 lightOffsets[right], lightOffsets.type(right), aabbDiag };
		}
	}

	// Now the nodes from the next higher (incomplete) level
	// Take into account that up to one light has been merged already at the beginning
	std::size_t startLight = (extraNodes % 2 == 0u) ? 0u : 1u;
	std::size_t nodeCount = (lightOffsets.light_count() - 2u * extraNodes - startLight) / 2u;
	// Start node for completely filled tree, but we may need an offset
	std::size_t lightStartNode = static_cast<std::size_t>(std::pow(2u, height)) - 1u + extraNodes;
	for(std::size_t i = 0u; i < nodeCount; ++i) {
		mAssert(startLight + 2u * i + 1u < lightOffsets.light_count());
		mAssert(lightStartNode + i < get_num_internal_nodes(lightOffsets.light_count()));
		std::size_t left = startLight + 2u * i;
		std::size_t right = startLight + 2u * i + 1u;
		Node& node = as<Node>(tree.memory)[lightStartNode + i];

		node = Node{ tree.memory, lightOffsets[left], lightOffsets.type(left),
			lightOffsets[right], lightOffsets.type(right), aabbDiag };
	}

	// Now for the rest of the levels (ie. inner nodes, no more lights nowhere)
	height -= 1u;
	for(std::size_t level = height; level >= 1u; --level) {
		std::size_t nodes = static_cast<std::size_t>(std::pow(2u, level));
		std::size_t innerNode = static_cast<std::size_t>(std::pow(2u, level - 1u)) - 1u;
		// Accumulate for higher-up node
		for(std::size_t i = 0u; i < nodes / 2u; ++i) {
			mAssert(innerNode + i < get_num_internal_nodes(lightOffsets.light_count()));
			mAssert(nodes + 2u * i < get_num_internal_nodes(lightOffsets.light_count()));
			std::size_t left = nodes - 1u + 2u * i;
			std::size_t right = nodes - 1u + 2u * i + 1u;
			Node& node = as<Node>(tree.memory)[innerNode + i];

			node = Node{ tree.memory, u32(left * sizeof(Node)), Node::INTERNAL_NODE_TYPE,
						 u32(right * sizeof(Node)), Node::INTERNAL_NODE_TYPE, aabbDiag };
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

using namespace lighttree_detail;

} // namespace


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
	m_flags()
{}

LightTreeBuilder::~LightTreeBuilder() {
	unload<Device::CPU>();
	unload<Device::CUDA>();
}

void LightTreeBuilder::build(std::vector<PositionalLights>&& posLights,
					  std::vector<DirectionalLight>&& dirLights,
					  const ei::Box& boundingBox,
					  TextureHandle envLight) {
	unload<Device::CPU>();
	m_treeCpu = std::make_unique<LightTree<Device::CPU>>( posLights.size() );

	// Construct the environment light
	if(envLight) {
		// First create an appropriate summed area table; this depends on if we have a cube or a spherical map
		if(envLight->get_num_layers() == 6)
			m_envmapSum = std::make_unique<textures::Texture>(6u * envLight->get_width(), envLight->get_height(), 1u,
															  envLight->get_format(), textures::SamplingMode::LINEAR,
															  false);
		else
			m_envmapSum = std::make_unique<textures::Texture>(envLight->get_width(), envLight->get_height(), 1u,
															  envLight->get_format(), textures::SamplingMode::LINEAR,
															  false);
		m_treeCpu->background = Background<Device::CPU>::envmap(envLight, m_envmapSum.get());
		m_textureMap.emplace(*envLight->aquireConst<Device::CPU>(), envLight);
	} else {
		// TODO: make more generic (different colors, analytic model...)
		m_treeCpu->background = Background<Device::CPU>::black();
	}

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

	m_treeCpu->length = sizeof(LightSubTree::Node) * (dirNodes + posNodes)
		+ dirLightOffsets.mem_size() + posLightOffsets.mem_size();
	m_treeCpu->memory = Allocator<Device::CPU>::alloc_array<char>(m_treeCpu->length);
	// Set up the node pointers
	m_treeCpu->dirLights.memory = m_treeCpu->memory;
	m_treeCpu->posLights.memory = m_treeCpu->memory + dirLightOffsets[0] + dirLightOffsets.mem_size();

	// Copy the lights into the tree
	// Directional lights are easier, because they have a fixed size
	{
		std::memcpy(m_treeCpu->dirLights.memory + dirLightOffsets[0], dirLights.data(), dirLightOffsets.mem_size());
	}
	// Positional lights are more difficult since we don't know the concrete size
	{
		char* mem = m_treeCpu->posLights.memory + posLightOffsets[0];
		for(const auto& light : posLights) {
			std::visit([&mem,this](const auto& posLight) {
				using T = std::decay<decltype(posLight)>;
				mem += sizeof(posLight);
				if constexpr(std::is_same_v<T,AreaLightTriangleDesc>) {
					auto* dst = as<AreaLightTriangle<Device::CPU>>(mem);
					*dst = posLight;
					dst->radianceTex = *posLight.radianceTex->aquireConst<Device::CPU>();
					// Remember texture for synchronization
					m_textureMap.emplace(dst->radianceTex, posLight.radianceTex);
				}
			}, light.light);
		}
	}

	// Now we gotta construct the proper nodes by recursively merging them together
	create_light_tree(dirLightOffsets, m_treeCpu->dirLights, scale);
	create_light_tree(posLightOffsets, m_treeCpu->posLights, scale);
	fill_map(posLights, m_treeCpu->primToNodePath);
	m_flags.mark_changed(Device::CPU);
}

void LightTreeBuilder::remap_textures(const char* cpuMem, u32 offset, u16 type, char* cudaMem) {
	switch(type) {
		case LightSubTree::Node::INTERNAL_NODE_TYPE: {
			// Recursive implementation necessary, because the type is stored at the parents and not known to the node itself
			const auto* node = as<LightSubTree::Node>(cpuMem + offset);
			remap_textures(cpuMem, node->left.offset, node->left.type, cudaMem);
		} break;
		case u16(LightType::AREA_LIGHT_TRIANGLE): {
			const auto* light = as<AreaLightTriangle<Device::CPU>>(cpuMem + offset);
			textures::ConstTextureDevHandle_t<Device::CUDA> tex =
				*m_textureMap.find(light->radianceTex)->second->aquireConst<Device::CUDA>();
			offset += u32((const char*)&light->radianceTex - (const char*)light);
			cudaMemcpy(cudaMem + offset, &tex, sizeof(tex), cudaMemcpyHostToDevice);
		} break;
		case u16(LightType::AREA_LIGHT_QUAD): {
			const auto* light = as<AreaLightQuad<Device::CPU>>(cpuMem + offset);
			textures::ConstTextureDevHandle_t<Device::CUDA> tex =
				*m_textureMap.find(light->radianceTex)->second->aquireConst<Device::CUDA>();
			offset += u32((const char*)&light->radianceTex - (const char*)light);
			cudaMemcpy(cudaMem + offset, &tex, sizeof(tex), cudaMemcpyHostToDevice);
		} break;
		case u16(LightType::AREA_LIGHT_SPHERE): {
			const auto* light = as<AreaLightSphere<Device::CPU>>(cpuMem + offset);
			textures::ConstTextureDevHandle_t<Device::CUDA> tex =
				*m_textureMap.find(light->radianceTex)->second->aquireConst<Device::CUDA>();
			offset += u32((const char*)&light->radianceTex - (const char*)light);
			cudaMemcpy(cudaMem + offset, &tex, sizeof(tex), cudaMemcpyHostToDevice);
		} break;
		default:; // Other light type - nothing to do
	}
}

template < Device dev >
void LightTreeBuilder::synchronize() {
	if(dev == Device::CUDA) {
		if(!m_treeCpu)
			throw std::runtime_error("[LightTreeBuilder::synchronize] There is no source for light-tree synchronization!");
		// Keep old memory if possible, any size change is better handled by a new allocation.
		if(!m_treeCuda || m_treeCuda->length != m_treeCpu->length
			|| m_treeCuda->primToNodePath.size() != m_treeCpu->primToNodePath.size()) {
			unload<Device::CUDA>();
			m_treeCuda = std::make_unique<LightTree<Device::CUDA>>( m_treeCpu->posLights.lightCount );
			m_treeCuda->memory = Allocator<Device::CUDA>::alloc_array<char>(m_treeCpu->length);
		}

		// Equalize bookkeeping
		m_treeCuda->length = m_treeCpu->length;
		m_treeCuda->dirLights = m_treeCpu->dirLights;
		m_treeCuda->dirLights.memory = m_treeCuda->memory;
		m_treeCuda->posLights = m_treeCpu->posLights;
		m_treeCuda->posLights.memory = m_treeCuda->memory + (m_treeCpu->posLights.memory - m_treeCpu->memory);

		// Remap the environment map
		TextureHandle envMapTex = nullptr;
		auto envMapIter = m_textureMap.find(m_treeCpu->background.get_envmap_light().texHandle);
		if(envMapIter != m_textureMap.end())
			envMapTex = envMapIter->second;
		m_treeCuda->background = m_treeCpu->background.synchronize<Device::CUDA>(envMapTex, m_envmapSum.get());

		// Copy the real data
		m_treeCuda->primToNodePath.synchornize(m_treeCpu->primToNodePath);
		cuda::check_error(cudaMemcpy(m_treeCuda->memory, m_treeCpu->memory, m_treeCpu->length, cudaMemcpyHostToDevice));

		// Replace all texture handles inside the tree's data
		remap_textures(m_treeCpu->posLights.memory, 0, m_treeCpu->posLights.root.type, m_treeCuda->posLights.memory);

		m_flags.mark_synced(dev);
	} else {
		// TODO: backsync? Would need another hashmap for texture mapping.
	}
}

template void LightTreeBuilder::synchronize<Device::CUDA>();
template void LightTreeBuilder::synchronize<Device::CPU>();

}}} // namespace mufflon::scene::lights