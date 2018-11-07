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

// Extracts the center from a positional light source
auto get_light_center = [](const auto& light) -> ei::Vec3 {
	using Type = std::decay_t<decltype(light)>;
	static_assert(is_positional_light_type<Type>(), "Center only available to positional lights");

	// Compute the center of the light source
	if constexpr(std::is_same_v<PointLight, Type> || std::is_same_v<SpotLight, Type>
				 || std::is_same_v<AreaLightSphere, Type>) {
		return light.position;
	} else if constexpr(std::is_same_v<AreaLightTriangle, Type>) {
		return ei::center(ei::Triangle(light.points[0u], light.points[1u], light.points[2u]));
	} else if constexpr(std::is_same_v<AreaLightQuad, Type>) {
		return ei::center(ei::Tetrahedron(light.points[0u], light.points[1u],
										  light.points[2u], light.points[3u]));
	}
};

// Extracts the intensity from a positional light source
auto get_pos_light_intensity = [](const auto& light) -> ei::Vec3 {
	using Type = std::decay_t<decltype(light)>;
	static_assert(is_positional_light_type<Type>(), "Center only available to positional lights");

	if constexpr(std::is_same_v<PointLight, Type> || std::is_same_v<SpotLight, Type>) {
		return light.intensity;
	} else if constexpr(std::is_same_v<AreaLightSphere, Type>) {
		return light.radiance * ei::surface(ei::Sphere{ light.position, light.radius });
	} else if constexpr(std::is_same_v<AreaLightTriangle, Type>) {
		return light.radiance * ei::surface(ei::Triangle(light.points[0u], light.points[1u],
														 light.points[2u]));
	} else if constexpr(std::is_same_v<AreaLightQuad, Type>) {
		return light.radiance * ei::surface(ei::Tetrahedron(light.points[0u], light.points[1u], 
														   light.points[2u], light.points[3u]));
	}
};

// Extracts the intensity of a directional light source with respect to the scene's bounding box
ei::Vec3 get_dir_light_intensity(const DirectionalLight& light, const ei::Vec3& aabbDiag) {
	// TODO: not sure if this is quite right
	// The area we want to sample over for a directional light is the projected
	// scene bounding box (or at least a good approximation)
	mAssert(aabbDiag.x > 0 && aabbDiag.y > 0 && aabbDiag.z > 0);
	float surface = aabbDiag.y*aabbDiag.z*light.direction.x
		+ aabbDiag.x*aabbDiag.z*light.direction.y
		+ aabbDiag.x*aabbDiag.y*light.direction.z;
	return light.radiance * surface;
}

// Gets the lights intensity, independent of type
template < class LightType >
ei::Vec3 get_light_intensity(const LightType& light, const ei::Vec3& aabbDiag) {
	static_assert((is_light_type<LightType>() && !is_envmap_light_type<LightType>())
				  || std::is_same_v<LightType, PositionalLights>,
				  "Must be non-envmap light");

	if constexpr(std::is_same_v<LightType, DirectionalLight>)
		return get_dir_light_intensity(light, aabbDiag);
	else
		return std::visit(get_pos_light_intensity, light);
}

// Computes the cluster centers for the light tree for directional lights/nodes
template < class DirLeft, class DirRight >
inline ei::Vec3 compute_cluster_center(const DirLeft& left, const DirRight& right) {
	return (left.direction + right.direction) / 2.f;
}
// Cluster centers for positional lights/nodes need some more finesse
inline ei::Vec3 compute_cluster_center(const LightTree::PosNode& left, const LightTree::PosNode& right) {
	return (left.position + right.position) / 2.f;
}
inline ei::Vec3 compute_cluster_center(const LightTree::PosNode& left, const PositionalLights& right) {
	return (left.position + std::visit(get_light_center, right)) / 2.f;
}
inline ei::Vec3 compute_cluster_center(const PositionalLights& left, const LightTree::PosNode& right) {
	return (std::visit(get_light_center, left) + right.position) / 2.f;
}
inline ei::Vec3 compute_cluster_center(const PositionalLights& left, const PositionalLights& right) {
	return (std::visit(get_light_center, left) + std::visit(get_light_center, right)) / 2.f;
}

// Returns the intensity depending on its type (light or node)
template < class T >
inline ei::Vec3 get_cluster_intensity(const T& cluster, const ei::Vec3& aabbDiag) {
	if constexpr(is_light_type<T>() || std::is_same_v<T, PositionalLights>)
		return get_light_intensity(cluster, aabbDiag);
	else
		return cluster.intensity;
}

// Computes the light intensities
template < class ClusterLeft, class ClusterRight >
inline ei::Vec3 compute_cluster_intensity(const ClusterLeft& left, const ClusterRight& right,
										  const ei::Vec3& aabbDiag) {
	return get_cluster_intensity(left, aabbDiag) + get_cluster_intensity(right, aabbDiag);
}

template < class NodeType, class LightType, class TreeType >
void create_light_tree(const std::vector<LightType>& lights, TreeType& tree,
					   const ei::Vec3& aabbDiag) {
	if(lights.size() == 0u)
		return;

	// Only one light -> no actual tree, only light
	// TODO: we still need the type...
	if(lights.size() == 1u)
		return;

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
			NodeType& interiorNode = tree.nodes[startNode + i];

			interiorNode = NodeType{
				compute_cluster_intensity(left, right, aabbDiag),
				LightTree::Child(),
				compute_cluster_center(left, right),
				LightTree::Child()
			};
			interiorNode.left.set_type(get_light_type(left));
			interiorNode.right.set_type(get_light_type(right));
		}

		// Also merge together inner nodes for last level!
		// Two cases: Merge two interior nodes each or (for last one) merge interior with light
		std::size_t startInnerNode = static_cast<std::size_t>(std::pow(2u, height - 1u)) - 1u;
		for(std::size_t i = 0u; i < extraNodes / 2u; ++i) {
			mAssert(startNode + 2u*i + 1u < get_num_internal_nodes(lights.size()));
			const NodeType& left = tree.nodes[startNode + 2u * i];
			const NodeType& right = tree.nodes[startNode + 2u * i + 1u];
			NodeType& node = tree.nodes[startInnerNode + i];

			node =  NodeType{
				compute_cluster_intensity(left, right, aabbDiag),
				LightTree::Child(),
				compute_cluster_center(left, right),
				LightTree::Child()
			};
			node.left.set_index(0u);
			node.right.set_index(0u);
		}
		if(extraNodes % 2 != 0u) {
			// One interior leftover; must be very first light
			mAssert(startNode + extraNodes - 1u < get_num_internal_nodes(lights.size()));
			const NodeType& left = tree.nodes[startNode + extraNodes - 1u];
			const LightType& right = lights.front();
			NodeType& node = tree.nodes[startInnerNode + extraNodes / 2u];

			node = NodeType{
				compute_cluster_intensity(left, right, aabbDiag),
				LightTree::Child(),
				compute_cluster_center(left, right),
				LightTree::Child()
			};
			node.left.set_index(0u);
			node.right.set_type(get_light_type(right));
		}
	}

	// Now the nodes from the next higher (incomplete) level
	// Take into account that up to one light has been marged already at the beginning
	std::size_t startLight = (extraNodes % 2 == 0u) ? 0u : 1u;
	std::size_t nodeCount = (lights.size() - 2u * extraNodes - startLight) / 2u;
	// Start node for completely filled tree, but we may need an offset
	std::size_t lightStartNode = static_cast<std::size_t>(std::pow(2u, height)) - 1u + extraNodes;
	for(std::size_t i = 0u; i < nodeCount; ++i) {
		mAssert(2u * i + startLight + 1u < lights.size());
		mAssert(lightStartNode + i < get_num_internal_nodes(lights.size()));
		const LightType& left = lights[2u * i + startLight];
		const LightType& right = lights[2u * i + startLight + 1u];
		NodeType& node = tree.nodes[lightStartNode + i];

		node = NodeType{
				compute_cluster_intensity(left, right, aabbDiag),
				LightTree::Child(),
				compute_cluster_center(left, right),
				LightTree::Child()
		};
		node.left.set_type(get_light_type(left));
		node.right.set_type(get_light_type(right));
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
			const NodeType& left = tree.nodes[nodes - 1u + 2u * i];
			const NodeType& right = tree.nodes[nodes - 1u + 2u * i + 1u];
			NodeType& node = tree.nodes[innerNode + i];

			node = NodeType{
				compute_cluster_intensity(left, right, aabbDiag),
				LightTree::Child(),
				compute_cluster_center(left, right),
				LightTree::Child()
			};
			node.left.set_index(0u);
			node.right.set_index(0u);
			// TODO: do we even need index?
		}
	}
}

} // namespace


LightTree::LightTree() :
	m_envMapTexture(nullptr),
	m_flags(),
	m_trees{
		Tree<Device::CPU>{
			LightType::NUM_LIGHTS,
			EnvMapLight<Device::CPU>{},
			{ 0u, nullptr, nullptr },
			{ 0u, nullptr, nullptr },
			0u,
			nullptr
		},
		Tree<Device::CUDA>{
			LightType::NUM_LIGHTS,
			EnvMapLight<Device::CUDA>{},
			{ 0u, nullptr, nullptr },
			{ 0u, nullptr, nullptr },
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
		ei::UVec3 x = ei::UVec3(std::visit(get_light_center, a) * scale);
		ei::UVec3 y = ei::UVec3(std::visit(get_light_center, b) * scale);
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
	std::size_t NodeTypes = get_num_internal_nodes(posLights.size());
	// For light memory we need to iterate the positional lights
	std::size_t dirLightSize = sizeof(DirectionalLight) * dirLights.size();
	std::size_t posLightSize = 0u;
	for(const auto& light : posLights)
		posLightSize += std::visit([](const auto& posLight) { return sizeof(posLight); }, light);

	tree.length = sizeof(DirNode) * dirNodes + sizeof(PosNode) * NodeTypes
		+ dirLightSize + posLightSize;
	tree.memory.handle = Allocator<Device::CPU>::alloc_array<char>(tree.length);
	// Set up the node pointers
	tree.dirLights.nodes = reinterpret_cast<DirNode*>(tree.memory.handle);
	tree.dirLights.lights = &tree.memory.handle[sizeof(DirNode) * dirNodes];
	tree.posLights.nodes = reinterpret_cast<PosNode*>(&tree.memory.handle[sizeof(DirNode) * dirNodes
																		  + dirLightSize]);
	tree.posLights.lights = &tree.memory.handle[sizeof(DirNode) * dirNodes + dirLightSize
												+ sizeof(PosNode) * NodeTypes];

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
	create_light_tree<DirNode>(dirLights, tree.dirLights, scale);
	create_light_tree<PosNode>(posLights, tree.posLights, scale);

	// Special casing for singular lights
	if(posLights.size() == 1u) {
		// It's kind of ugly having to keep this around, but hey...
		tree.singlePosLightType = get_light_type(posLights.front());
	}
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