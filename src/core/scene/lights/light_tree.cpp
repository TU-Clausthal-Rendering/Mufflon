#include "light_tree.hpp"
#include "core/memory/allocator.hpp"
#include "util/assert.hpp"
#include "core/cuda/error.hpp"
#include "ei/3dtypes.hpp"
#include <cuda_runtime.h>
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

// The following functions are taken from (taken from https://github.com/Jojendersie/Bim/blob/master/src/bim_sbvh.cpp)

// Two sources to derive the z-order comparator
// (floats - unused) http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.150.9547&rep=rep1&type=pdf
// (ints - the below one uses this int-algorithm on floats) http://dl.acm.org/citation.cfm?id=545444
// http://www.forceflow.be/2013/10/07/morton-encodingdecoding-through-bit-interleaving-implementations/ Computing morton codes
auto part_by_two = [](const u16 x) constexpr->u64 {
	u64 r = x;
	r = (r | (r << 16)) & 0x0000ff0000ff;
	r = (r | (r << 8)) & 0x00f00f00f00f;
	r = (r | (r << 4)) & 0x0c30c30c30c3;
	r = (r | (r << 2)) & 0x249249249249;
	return r;
};

// Converts gray-code to regular binary
auto gray_to_binary = [](u64 num) constexpr->u64{
	num = num ^ (num >> 32);
	num = num ^ (num >> 16);
	num = num ^ (num >> 8);
	num = num ^ (num >> 4);
	num = num ^ (num >> 2);
	num = num ^ (num >> 1);
	return num;
};

// Encodes 3 16-bit values into a single 64 bit value
auto get_morton_code = [](const u16 a, const u16 b, const u16 c) constexpr->u64 {
	return part_by_two(a) | (part_by_two(b) << 1u) | (part_by_two(c) << 2u);
};

template < class T >
ei::Vec3 get_light_center(const T& light) {
	auto getCenter = [](const auto& posLight) constexpr {
		using Type = std::decay_t<decltype(posLight)>;
		if constexpr(std::is_same_v<PointLight, Type> || std::is_same_v<SpotLight, Type>
					 || std::is_same_v<AreaLightSphere, Type>) {
			return posLight.position;
		} else if constexpr(std::is_same_v<AreaLightTriangle, Type>) {
			return ei::center(ei::Triangle(posLight.points[0u], posLight.points[1u],
										   posLight.points[2u]));
		} else if constexpr(std::is_same_v<AreaLightQuad, Type>) {
			static_assert(std::is_same_v<AreaLightQuad, Type>,
						  "Unknown positional light type");
			return ei::center(ei::Tetrahedron(posLight.points[0u], posLight.points[1u],
											  posLight.points[2u], posLight.points[3u]));
		}
	};

	if constexpr(std::is_same_v<T, PositionalLights>)
		return std::visit(getCenter, light);
	else
		return getCenter(light);
}

// Computes the offset of the i-th point light - positional ones need a sum table!
template < class LT >
class LightOffset {
public:
	LightOffset(const std::vector<LT>& lights) {}
	
	constexpr u32 operator[](std::size_t lightIndex) const noexcept {
		return static_cast<u32>(sizeof(LT) * lightIndex);
	}
};
template <>
class LightOffset<PositionalLights> {
public:
	LightOffset(const std::vector<PositionalLights>& lights) : m_offsets(lights.size()) {
		m_offsets[0u] = 0u;
		for(std::size_t i = 1u; i < lights.size(); ++i) {
			m_offsets[i] = m_offsets[i - 1u] + std::visit([](const auto& light) constexpr {
				return static_cast<u32>(sizeof(light));
			}, lights[i - 1u]);
		}
	}

	u32 operator[](std::size_t lightIndex) const noexcept {
		mAssert(lightIndex < m_offsets.size());
		return m_offsets[lightIndex];
	}

private:
	std::vector<u32> m_offsets;
};


template < class LightType >
void create_light_tree(const std::vector<LightType>& lights, LightTree::LightTypeTree& tree,
					   const ei::Vec3& aabbDiag) {
	using Node = LightTree::Node;

	tree.lightCount = lights.size();
	if(lights.size() == 0u)
		return;

	// Only one light -> no actual tree, only light
	if(lights.size() == 1u) {
		tree.root.type = static_cast<u16>(get_light_type(lights.front()));
		tree.root.flux = ei::sum(get_flux(lights.front(), aabbDiag));
		return;
	}

	// Compute a sum table for the light offsets for positional lights,
	// nothing for directional ones
	LightOffset<LightType> lightOffsets(lights);

	// Correctly set the internal nodes
	// Start with nodes that form the last (incomplete!) level
	std::size_t height = static_cast<std::size_t>(std::log2(lights.size()));
	mAssert(height > 0u);
	std::size_t extraNodes = lights.size() - static_cast<std::size_t>(std::pow(2u, height));
	if(extraNodes > 0u) {
		// Compute starting positions for internal nodes and lights
		std::size_t startNode = static_cast<std::size_t>(std::pow(2u, height)) - 1u;
		std::size_t startLight = lights.size() - 2u * extraNodes;
		// "Merge" together two nodes
		for(std::size_t i = 0u; i < extraNodes; ++i) {
			mAssert(startLight + 2u * i + 1u < lights.size());
			mAssert(startNode + i < get_num_internal_nodes(lights.size()));
			const LightType& left = lights[startLight + 2u * i];
			const LightType& right = lights[startLight + 2u * i + 1u];
			Node& interiorNode = tree.nodes[startNode + i];

			interiorNode = Node{ left, right, aabbDiag };
			interiorNode.left.set_offset(lightOffsets[startLight + 2u * i]);
			interiorNode.right.set_offset(lightOffsets[startLight + 2u * i + 1u]);
		}

		// Also merge together inner nodes for last level!
		// Two cases: Merge two interior nodes each or (for last one) merge interior with light
		std::size_t startInnerNode = static_cast<std::size_t>(std::pow(2u, height - 1u)) - 1u;
		for(std::size_t i = 0u; i < extraNodes / 2u; ++i) {
			mAssert(startNode + 2u*i + 1u < get_num_internal_nodes(lights.size()));
			const Node& left = tree.nodes[startNode + 2u * i];
			const Node& right = tree.nodes[startNode + 2u * i + 1u];
			Node& node = tree.nodes[startInnerNode + i];

			node = Node{ left, right };
			node.left.set_offset(startNode + 2u * i);
			node.right.set_offset(startNode + 2u * i + 1u);
		}
		if(extraNodes % 2 != 0u) {
			// One interior leftover; must be very first light
			mAssert(startNode + extraNodes - 1u < get_num_internal_nodes(lights.size()));
			const Node& left = tree.nodes[startNode + extraNodes - 1u];
			const LightType& right = lights.front();
			Node& node = tree.nodes[startInnerNode + extraNodes / 2u];

			node = Node{ left, right, aabbDiag };
			node.left.set_offset(startNode + extraNodes - 1u);
			node.right.set_offset(lightOffsets[startInnerNode + extraNodes / 2u]);
		}
	}

	// Now the nodes from the next higher (incomplete) level
	// Take into account that up to one light has been marged already at the beginning
	std::size_t startLight = (extraNodes % 2 == 0u) ? 0u : 1u;
	std::size_t nodeCount = (lights.size() - 2u * extraNodes - startLight) / 2u;
	// Start node for completely filled tree, but we may need an offset
	std::size_t lightStartNode = static_cast<std::size_t>(std::pow(2u, height)) - 1u + extraNodes;
	for(std::size_t i = 0u; i < nodeCount; ++i) {
		mAssert(startLight + 2u * i + 1u < lights.size());
		mAssert(lightStartNode + i < get_num_internal_nodes(lights.size()));
		const LightType& left = lights[startLight + 2u * i];
		const LightType& right = lights[startLight + 2u * i + 1u];
		Node& node = tree.nodes[lightStartNode + i];

		node = Node{ left, right, aabbDiag };
		node.left.set_offset(lightOffsets[startLight + 2u * i]);
		node.right.set_offset(lightOffsets[startLight + 2u * i + 1u]);
	}

	// Now for the rest of the levels (ie. inner nodes, no more lights nowhere)
	height -= 1u;
	for(std::size_t level = height; level >= 1u; --level) {
		std::size_t nodes = static_cast<std::size_t>(std::pow(2u, level));
		std::size_t innerNode = static_cast<std::size_t>(std::pow(2u, level - 1u)) - 1u;
		// Accumulate for higher-up node
		for(std::size_t i = 0u; i < nodes / 2u; ++i) {
			mAssert(innerNode + i < get_num_internal_nodes(lights.size()));
			mAssert(nodes + 2u * i < get_num_internal_nodes(lights.size()));
			const Node& left = tree.nodes[nodes - 1u + 2u * i];
			const Node& right = tree.nodes[nodes - 1u + 2u * i + 1u];
			Node& node = tree.nodes[innerNode + i];

			node = Node{ left, right };
			node.left.set_offset(nodes - 1u + 2u * i);
			node.right.set_offset(nodes - 1u + 2u * i + 1u);
		}
	}

	// Last, set the root properties: guaranteed two lights
	tree.root.type = Node::INVALID_TYPE;
	tree.root.flux = tree.nodes[0u].left.flux + tree.nodes[0u].right.flux;
}

} // namespace


// Node constructors
LightTree::Node::Node(const Node& left, const Node& right) :
	left{ left.left.flux + left.right.flux, 0u, Node::INVALID_TYPE },
	right{ Node::INVALID_TYPE, 0u, right.left.flux + right.right.flux },
	center{ (left.center + right.center) / 2.f }
{}
LightTree::Node::Node(const Node& left, const PositionalLights& right,
					  const ei::Vec3& aabbDiag) :
	left{ left.left.flux + left.right.flux, 0u, Node::INVALID_TYPE },
	right{ static_cast<u16>(get_light_type(right)), 0u, ei::sum(get_flux(right)) },
	center{ (left.center + get_light_center(right)) / 2.f }
{}
LightTree::Node::Node(const Node& left, const DirectionalLight& right,
					  const ei::Vec3& aabbDiag) :
	left{ left.left.flux + left.right.flux, 0u, Node::INVALID_TYPE },
	right{ static_cast<u16>(get_light_type(right)), 0u, ei::sum(get_flux(right, aabbDiag)) },
	center{ (left.center + right.direction) / 2.f }
{}
LightTree::Node::Node(const PositionalLights& left, const Node& right,
					  const ei::Vec3& aabbDiag) :
	left{ ei::sum(get_flux(left)), 0u, static_cast<u16>(get_light_type(left)) },
	right{ Node::INVALID_TYPE, 0u, right.left.flux + right.right.flux  },
	center{ (get_light_center(left) + right.center) / 2.f }
{}
LightTree::Node::Node(const DirectionalLight& left, const Node& right,
					  const ei::Vec3& aabbDiag) :
	left{ ei::sum(get_flux(left, aabbDiag)), 0u, static_cast<u16>(get_light_type(left)) },
	right{ Node::INVALID_TYPE, 0u, right.left.flux + right.right.flux },
	center{ (left.direction + right.center) / 2.f }
{}
LightTree::Node::Node(const PositionalLights& left, const PositionalLights& right,
					  const ei::Vec3& aabbDiag) :
	left{ ei::sum(get_flux(left)), 0u, static_cast<u16>(get_light_type(left)) },
	right{ static_cast<u16>(get_light_type(right)), 0u,  ei::sum(get_flux(right)) },
	center{ (get_light_center(left) + get_light_center(right)) / 2.f }
{}
LightTree::Node::Node(const DirectionalLight& left, const DirectionalLight& right,
					  const ei::Vec3& aabbDiag) :
	left{ ei::sum(get_flux(left, aabbDiag)), 0u, static_cast<u16>(get_light_type(left)) },
	right{ static_cast<u16>(get_light_type(right)), 0u,  ei::sum(get_flux(right, aabbDiag)) },
	center{ (left.direction + right.direction) / 2.f }
{}


LightTree::LightTree() :
	m_envMapTexture(nullptr),
	m_flags(),
	m_trees{
		Tree<Device::CPU>{
			EnvMapLight<Device::CPU>{},
			{ {}, 0u, nullptr, nullptr },
			{ {}, 0u, nullptr, nullptr },
			0u,
			nullptr
		},
		Tree<Device::CUDA>{
			EnvMapLight<Device::CUDA>{},
			{ {}, 0u, nullptr, nullptr },
			{ {}, 0u, nullptr, nullptr },
			0u,
			nullptr
		},
	}
{

}

LightTree::~LightTree() {

}

void LightTree::build(std::vector<PositionalLights>&& posLights,
					  std::vector<DirectionalLight>&& dirLights,
					  const ei::Box& boundingBox,
					  std::optional<textures::TextureHandle> envLight) {
	Tree<Device::CPU>& tree = m_trees.get<Tree<Device::CPU>>();

	// First delete any leftovers
	tree.memory.handle = Allocator<Device::CPU>::free(tree.memory.handle, tree.length);
	tree.envLight = EnvMapLight<Device::CPU>{};

	// Construct the environment light
	if(envLight.has_value()) {
		tree.envLight = EnvMapLight<Device::CPU>{ {{ *envLight.value()->aquireConst<Device::CPU>() }}, ei::Vec3{0, 0, 0} };
		// TODO: accumulate flux
	}

	if(posLights.size() == 0u && dirLights.size() == 0u) {
		// Shortcut if no lights are specified
		tree.dirLights.lightCount = 0u;
		tree.posLights.lightCount = 0u;
		return;
	}

	// Create our spatial sorting to get a good tree
	// TODO: sort the directional lights by direction
	ei::Vec3 scale = boundingBox.max - boundingBox.min;
	std::sort(posLights.begin(), posLights.end(), [&scale](const PositionalLights& a, const PositionalLights& b) {
		// Rescale and round the light centers to fall onto positive integers
		ei::UVec3 x = ei::UVec3(get_light_center(a) * scale);
		ei::UVec3 y = ei::UVec3(get_light_center(b) * scale);
		// Interleave packages of 16 bit, convert to inverse Gray code and compare.
		u64 codeA = gray_to_binary(get_morton_code(x.x >> 16u, x.y >> 16u, x.z >> 16u));
		u64 codeB = gray_to_binary(get_morton_code(y.x >> 16u, y.y >> 16u, y.z >> 16u));
		// If they are equal take the next 16 less significant bit.
		if(codeA == codeB) {
			codeA = gray_to_binary(get_morton_code(x.x & 0xffff, x.y & 0xffff, x.z & 0xffff));
			codeB = gray_to_binary(get_morton_code(y.x & 0xffff, y.y & 0xffff, y.z & 0xffff));
		}
		return codeA < codeB;
	});

	// Allocate enough space to fit all lights - this assumes that the tree will be balanced!
	// Node memory is easy since fixed size
	std::size_t dirNodes = get_num_internal_nodes(dirLights.size());
	std::size_t posNodes = get_num_internal_nodes(posLights.size());
	// For light memory we need to iterate the positional lights
	std::size_t dirLightSize = sizeof(DirectionalLight) * dirLights.size();
	std::size_t posLightSize = 0u;
	for(const auto& light : posLights)
		posLightSize += std::visit([](const auto& posLight) { return sizeof(posLight); }, light);

	tree.length = sizeof(Node) * (dirNodes + posNodes)
		+ dirLightSize + posLightSize;
	tree.memory.handle = Allocator<Device::CPU>::alloc_array<char>(tree.length);
	// Set up the node pointers
	tree.dirLights.nodes = reinterpret_cast<Node*>(tree.memory.handle);
	tree.dirLights.lights = &tree.memory.handle[sizeof(Node) * dirNodes];
	tree.posLights.nodes = reinterpret_cast<Node*>(&tree.memory.handle[sizeof(Node) * dirNodes
																		  + dirLightSize]);
	tree.posLights.lights = &tree.memory.handle[sizeof(Node) * dirNodes
												+ dirLightSize
												+ sizeof(Node) * posNodes];

	// Copy the lights into the tree
	// TODO: how to place this best into memory?
	// Directional lights are easier, because they have a fixed size
	{
		std::memcpy(tree.dirLights.lights, dirLights.data(), dirLightSize);
	}
	// Positional lights are more difficult since we don't know the concrete size
	{
		char* mem = tree.posLights.lights;
		for(const auto& light : posLights) {
			std::visit([&mem](const auto& posLight) {
				std::memcpy(mem, &posLight, sizeof(posLight));
				mem += sizeof(posLight);
			}, light);
		}
	}

	// Now we gotta construct the proper nodes by recursively merging them together
	create_light_tree(dirLights, tree.dirLights, scale);
	create_light_tree(posLights, tree.posLights, scale);
}

// TODO
void synchronize(const LightTree::Tree<Device::CPU>& changed, LightTree::Tree<Device::CUDA>& sync,
				 std::optional<textures::TextureHandle> hdl) {
	if(changed.length == 0u) {
		// Remove all data since there are no lights
		mAssert(changed.memory.handle == nullptr);
		sync.memory.handle = Allocator<Device::CUDA>::free(sync.memory.handle, sync.length);
	} else {
		// Still have data, (re)alloc and copy
		if(sync.memory.handle == nullptr) {
			sync.memory.handle = Allocator<Device::CUDA>::alloc_array<char>(changed.length);
		} else if(sync.length != changed.length) {
			sync.memory.handle = Allocator<Device::CUDA>::realloc(sync.memory.handle, sync.length,
																  changed.length);
		}
		cudaMemcpy(sync.memory.handle, changed.memory.handle, changed.length, cudaMemcpyHostToDevice);
	}
	// Equalize bookkeeping
	sync.length = changed.length;
	sync.dirLights.lightCount = changed.dirLights.lightCount;
	sync.posLights.lightCount = changed.posLights.lightCount;

	// Also copy the environment light
	sync.envLight.flux = changed.envLight.flux;
	if(hdl.has_value())
		sync.envLight.texHandle = textures::ConstDeviceTextureHandle<Device::CUDA>{ {*hdl.value()->aquireConst<Device::CUDA>()} };
	else
		sync.envLight.texHandle = textures::ConstDeviceTextureHandle<Device::CUDA>{};
}

void synchronize(const LightTree::Tree<Device::CUDA>& changed, LightTree::Tree<Device::CPU>& sync,
				 std::optional<textures::TextureHandle> hdl) {
	if(changed.length == 0u) {
		// Remove all data since there are no lights
		mAssert(changed.memory.handle == nullptr);
		sync.memory.handle = Allocator<Device::CPU>::free(sync.memory.handle, sync.length);
	} else {
		// Still have data, (re)alloc and copy
		if(sync.memory.handle == nullptr) {
			sync.memory.handle = Allocator<Device::CPU>::alloc_array<char>(changed.length);
		} else if(sync.length != changed.length) {
			sync.memory.handle = Allocator<Device::CPU>::realloc(sync.memory.handle, sync.length,
																  changed.length);
		}
		cudaMemcpy(sync.memory.handle, changed.memory.handle, changed.length, cudaMemcpyHostToDevice);
	}
	// Equalize bookkeeping
	sync.length = changed.length;
	sync.dirLights.lightCount = changed.dirLights.lightCount;
	sync.posLights.lightCount = changed.posLights.lightCount;

	// Also copy the environment light
	sync.envLight.flux = changed.envLight.flux;
	if(hdl.has_value())
		sync.envLight.texHandle = textures::ConstDeviceTextureHandle<Device::CPU>{ {*hdl.value()->aquireConst<Device::CPU>()} };
	else
		sync.envLight.texHandle = textures::ConstDeviceTextureHandle<Device::CPU>{};
}

}}} // namespace mufflon::scene::lights