#include "light_tree.hpp"
#include "core/scene/allocator.hpp"
#include "util/assert.hpp"
#include "core/cuda/error.hpp"
#include "ei/3dtypes.hpp"
#include <cuda_runtime.h>
#include <cmath>

namespace mufflon { namespace scene { namespace lights {

namespace {

// Computes the number of nodes of a balanced tree with elems leaves
std::size_t get_num_internal_nodes(std::size_t elems) {
	if(elems == 0u)
		return 0u;
	// Height of a balanced binary tree is log2(N) + 1
	std::size_t height = static_cast<std::size_t>(std::log2(elems)) + 1u;
	// Interior nodes are then 2^(height-1) - 1u
	std::size_t nodes = static_cast<std::size_t>(std::pow(2u, height - 1u)) - 1u;
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
	// Compute the center of the light source
	if constexpr(std::is_same_v<PointLight, Type> || std::is_same_v<SpotLight, Type>
				 || std::is_same_v<AreaLightSphere, Type>) {
		return light.position;
	} else if constexpr(std::is_same_v<AreaLightTriangle, Type>) {
		return ei::center(ei::Triangle(light.points[0u], light.points[1u], light.points[2u]));
	} else if constexpr(std::is_same_v<AreaLightQuad, Type>) {
		return ei::center(ei::Tetrahedron(light.points[0u], light.points[1u],
										  light.points[2u], light.points[3u]));
	} else {
		static_assert(false, "Unknown positional light source!");
	}
};

// Extracts the intensity from a positional light source
auto get_pos_light_intensity = [](const auto& light) -> ei::Vec3 {
	using Type = std::decay_t<decltype(light)>;

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
	} else {
		static_assert(false, "Unknown positional light source!");
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

auto get_light_type = [](const auto& light) constexpr -> LightType {
	using Type = std::decay_t<decltype(light)>;

	if constexpr(std::is_same_v<Type, PointLight>)
		return LightType::POINT_LIGHT;
	else if constexpr(std::is_same_v<Type, SpotLight>)
		return LightType::SPOT_LIGHT;
	else if constexpr(std::is_same_v<Type, AreaLightTriangle>)
		return LightType::AREA_LIGHT_TRIANGLE;
	else if constexpr(std::is_same_v<Type, AreaLightQuad>)
		return LightType::AREA_LIGHT_QUAD;
	else if constexpr(std::is_same_v<Type, AreaLightSphere>)
		return LightType::AREA_LIGHT_SPHERE;
	else if constexpr(std::is_same_v<Type, DirectionalLight>)
		return LightType::DIRECTIONAL_LIGHT;
	else if constexpr(std::is_same_v<Type, EnvMapLight>)
		return LightType::ENVMAP_LIGHT;
	else
		static_assert(false, "Unknown light type");
};

void create_positional_light_tree(const std::vector<PositionalLights>& posLights,
								  char* memory) {
	using PosNode = LightTree::PosNode;
	if(posLights.size() == 0u)
		return;

	// Only one light -> no actual tree, only light
	// TODO: we still need the type...
	if(posLights.size() == 1u)
		return;

	// Correctly set the internal nodes
	// Start with nodes that form the last (incomplete!) level
	std::size_t height = static_cast<std::size_t>(std::log2(posLights.size()));
	mAssert(height > 0u);
	std::size_t extraNodes = posLights.size() - height;
	if(extraNodes > 0u) {
		// Compute starting positions for internal nodes and lights
		std::size_t startNode = static_cast<std::size_t>(std::pow(2u, height)) - 1u;
		std::size_t startLight = posLights.size() - 2u * extraNodes;
		// "Merge" together two nodes
		for(std::size_t i = 0u; i < extraNodes; ++i) {
			mAssert(startLight + 2u * i < posLights.size());
			const PositionalLights& left = posLights[startLight + 2u * i];
			const PositionalLights& right = posLights[startLight + 2u * i + 1u];

			PosNode* interiorNode = new (&memory[sizeof(PosNode) * (startNode + i)]) PosNode{
				std::visit(get_pos_light_intensity, left) + std::visit(get_pos_light_intensity, right),
				PosNode::Child(),
				(std::visit(get_light_center, left) + std::visit(get_light_center, right)) / 2.f,
				PosNode::Child()
			};
			interiorNode->left.set_type(std::visit(get_light_type, left));
			interiorNode->right.set_type(std::visit(get_light_type, right));
		}

		// Also merge together inner nodes for last level!
		// Two cases: Merge two interior nodes each or (for last one) merge interior with light
		std::size_t startInnerNode = static_cast<std::size_t>(std::pow(2u, height - 1u)) - 1u;
		for(std::size_t i = 0u; i < extraNodes / 2u; ++i) {
			const PosNode* left = reinterpret_cast<const PosNode*>(&memory[sizeof(PosNode) * (startNode + 2u * i)]);
			const PosNode* right = reinterpret_cast<const PosNode*>(&memory[sizeof(PosNode) * (startNode + 2u * i + 1u)]);
			mAssert(left != nullptr);
			mAssert(right != nullptr);

			PosNode* node = new (&memory[sizeof(PosNode) * (startInnerNode + i)]) PosNode{
				left->intensity + right->intensity,
				PosNode::Child(),
				(left->position + right->position) / 2.f,
				PosNode::Child()
			};
			node->left.set_index(0u);
			node->right.set_index(0u);
		}
		if(extraNodes % 2 != 0u) {
			// One interior leftover; must be very first light
			const PosNode* left = reinterpret_cast<const PosNode*>(&memory[sizeof(PosNode) * (startNode + (extraNodes + 1u) / 2u)]);
			const PositionalLights& right = posLights.front();
			mAssert(left != nullptr);

			PosNode* node = new (&memory[sizeof(PosNode) * (startInnerNode + extraNodes / 2u)]) PosNode{
				left->intensity + std::visit(get_pos_light_intensity, right),
				PosNode::Child(),
				(left->position + std::visit(get_light_center, right)) / 2.f,
				PosNode::Child()
			};
			node->left.set_index(0u);
			node->right.set_type(std::visit(get_light_type, right));
		}
	}

	// Now the nodes from the next higher (incomplete) level
	// Take into account that up to one light has been marged already at the beginning
	std::size_t startLight = (extraNodes % 2 == 0u) ? 0u : 1u;
	std::size_t nodeCount = (posLights.size() - 2u * extraNodes - startLight) / 2u;
	// Start node for completely filled tree, but we may need an offset
	std::size_t lightStartNode = static_cast<std::size_t>(std::pow(2u, height)) - 1u + extraNodes;
	for(std::size_t i = 0u; i < nodeCount; ++i) {
		const PositionalLights& left = posLights[2u * i + startLight];
		const PositionalLights& right = posLights[2u * i + startLight + 1u];
		PosNode* node = new (&memory[sizeof(PosNode) * (lightStartNode + i)]) PosNode{
				std::visit(get_pos_light_intensity, left) + std::visit(get_pos_light_intensity, right),
				PosNode::Child(),
				(std::visit(get_light_center, left) + std::visit(get_light_center, right)) / 2.f,
				PosNode::Child()
		};
		node->left.set_type(std::visit(get_light_type, left));
		node->right.set_type(std::visit(get_light_type, right));
	}

	// Now for the rest of the levels (ie. inner nodes, no more lights nowhere)
	height -= 1u;
	for(std::size_t level = height; level >= 1u; --level) {
		std::size_t nodes = static_cast<std::size_t>(std::pow(2u, level));
		std::size_t innerNode = static_cast<std::size_t>(std::pow(2u, level - 1u)) - 1u;
		// Accumulate for higher-up node
		for(std::size_t i = 0u; i < nodes / 2u; ++i) {
			const PosNode* left = reinterpret_cast<const PosNode*>(&memory[sizeof(PosNode) * (nodes - 1u + 2 * i)]);
			const PosNode* right = reinterpret_cast<const PosNode*>(&memory[sizeof(PosNode) * (nodes - 1u + 2 * i + 1u)]);
			mAssert(left != nullptr);
			mAssert(right != nullptr);

			PosNode* node = new (&memory[sizeof(PosNode) * (innerNode + i)]) PosNode{
				left->intensity + right->intensity,
				PosNode::Child(),
				(left->position + right->position) / 2.f,
				PosNode::Child()
			};
			node->left.set_index(0u);
			node->right.set_index(0u);
			// TODO: do we even need index?
		}
	}
}

void create_directional_light_tree(const std::vector<DirectionalLight>& dirLights,
								   char* memory, const ei::Vec3& aabbDiag) {
	using DirNode = LightTree::DirNode;
	if(dirLights.size() == 0u)
		return;

	// Only one light -> no actual tree, only light
	// TODO: we still need the type...
	if(dirLights.size() == 1u)
		return;

	// Correctly set the internal nodes
	// Start with nodes that form the last (incomplete!) level
	std::size_t height = static_cast<std::size_t>(std::log2(dirLights.size()));
	mAssert(height > 0u);
	std::size_t extraNodes = dirLights.size() - height;
	if(extraNodes > 0u) {
		// Compute starting positions for internal nodes and lights
		std::size_t startNode = static_cast<std::size_t>(std::pow(2u, height)) - 1u;
		std::size_t startLight = dirLights.size() - 2u * extraNodes;
		// "Merge" together two nodes
		for(std::size_t i = 0u; i < extraNodes; ++i) {
			mAssert(startLight + 2u * i < dirLights.size());
			const DirectionalLight& left = dirLights[startLight + 2u * i];
			const DirectionalLight& right = dirLights[startLight + 2u * i + 1u];

			DirNode* interiorNode = new (&memory[sizeof(DirNode) * (startNode + i)]) DirNode{
				get_dir_light_intensity(left, aabbDiag) + get_dir_light_intensity(right, aabbDiag),
				DirNode::Child(),
				(left.direction + right.direction) / 2.f,
				DirNode::Child()
			};
			interiorNode->left.make_leaf();
			interiorNode->right.make_leaf();
		}

		// Also merge together inner nodes for last level!
		// Two cases: Merge two interior nodes each or (for last one) merge interior with light
		std::size_t startInnerNode = static_cast<std::size_t>(std::pow(2u, height - 1u)) - 1u;
		for(std::size_t i = 0u; i < extraNodes / 2u; ++i) {
			const DirNode* left = reinterpret_cast<const DirNode*>(&memory[sizeof(DirNode) * (startNode + 2u * i)]);
			const DirNode* right = reinterpret_cast<const DirNode*>(&memory[sizeof(DirNode) * (startNode + 2u * i + 1u)]);
			mAssert(left != nullptr);
			mAssert(right != nullptr);

			DirNode* node = new (&memory[sizeof(DirNode) * (startInnerNode + i)]) DirNode{
				left->intensity + right->intensity,
				DirNode::Child(),
				(left->direction + right->direction) / 2.f,
				DirNode::Child()
			};
			node->left.set_index(0u);
			node->right.set_index(0u);
		}
		if(extraNodes % 2 != 0u) {
			// One interior leftover; must be very first light
			const DirNode* left = reinterpret_cast<const DirNode*>(&memory[sizeof(DirNode) * (startNode + (extraNodes + 1u) / 2u)]);
			const DirectionalLight& right = dirLights.front();
			mAssert(left != nullptr);

			DirNode* node = new (&memory[sizeof(DirNode) * (startInnerNode + extraNodes / 2u)]) DirNode{
				left->intensity + get_dir_light_intensity(right, aabbDiag),
				DirNode::Child(),
				(left->direction + right.direction) / 2.f,
				DirNode::Child()
			};
			node->left.set_index(0u);
			node->right.make_leaf();
		}
	}

	// Now the nodes from the next higher (incomplete) level
	// Take into account that up to one light has been marged already at the beginning
	std::size_t startLight = (extraNodes % 2 == 0u) ? 0u : 1u;
	std::size_t nodeCount = (dirLights.size() - 2u * extraNodes - startLight) / 2u;
	// Start node for completely filled tree, but we may need an offset
	std::size_t lightStartNode = static_cast<std::size_t>(std::pow(2u, height)) - 1u + extraNodes;
	for(std::size_t i = 0u; i < nodeCount; ++i) {
		const DirectionalLight& left = dirLights[2u * i + startLight];
		const DirectionalLight& right = dirLights[2u * i + startLight + 1u];
		DirNode* node = new (&memory[sizeof(DirNode) * (lightStartNode + i)]) DirNode{
				get_dir_light_intensity(left, aabbDiag) + get_dir_light_intensity(right, aabbDiag),
				DirNode::Child(),
				(left.direction + right.direction) / 2.f,
				DirNode::Child()
		};
		node->left.make_leaf();
		node->right.make_leaf();
	}

	// Now for the rest of the levels (ie. inner nodes, no more lights nowhere)
	height -= 1u;
	for(std::size_t level = height; level >= 1u; --level) {
		std::size_t nodes = static_cast<std::size_t>(std::pow(2u, level));
		std::size_t innerNode = static_cast<std::size_t>(std::pow(2u, level - 1u)) - 1u;
		// Accumulate for higher-up node
		for(std::size_t i = 0u; i < nodes / 2u; ++i) {
			const DirNode* left = reinterpret_cast<const DirNode*>(&memory[sizeof(DirNode) * (nodes - 1u + 2 * i)]);
			const DirNode* right = reinterpret_cast<const DirNode*>(&memory[sizeof(DirNode) * (nodes - 1u + 2 * i + 1u)]);
			mAssert(left != nullptr);
			mAssert(right != nullptr);

			DirNode* node = new (&memory[sizeof(DirNode) * (innerNode + i)]) DirNode{
				left->intensity + right->intensity,
				DirNode::Child(),
				(left->direction + right->direction) / 2.f,
				DirNode::Child()
			};
			node->left.set_index(0u);
			node->right.set_index(0u);
			// TODO: do we even need index?
		}
	}
}

}


void LightTree::build(std::vector<PositionalLights>&& posLights,
					  std::vector<DirectionalLight>&& dirLights,
					  const ei::Box& boundingBox,
					  std::optional<textures::TextureHandle> envLight) {
	Tree<Device::CPU>& tree = m_trees.get<Tree<Device::CPU>>();
	tree.dirLights.handle = Allocator<Device::CPU>::free(tree.dirLights.handle, tree.numDirLights);
	tree.posLights.handle = Allocator<Device::CPU>::free(tree.posLights.handle, tree.numPosLights);
	if(envLight.has_value())
		tree.envLight = EnvMapLight<Device::CPU>{ *envLight.value()->aquireConst<Device::CPU>() };
	else
		tree.envLight = EnvMapLight<Device::CPU>{ textures::ConstDeviceTextureHandle<Device::CPU>{} };
	// TODO: accumulate flux

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
	std::size_t dirNodes = get_num_internal_nodes(dirLights.size());
	std::size_t posNodes = get_num_internal_nodes(posLights.size());
	if(dirLights.size() > 0u)
		tree.dirLights.handle = Allocator<Device::CPU>::alloc<char>(sizeof(DirNode) * dirNodes + sizeof(DirectionalLight) * dirLights.size());
	if(posLights.size() > 0u)
		tree.posLights.handle = Allocator<Device::CPU>::alloc<char>(sizeof(PosNode) * posNodes + sizeof(PositionalLight) * dirLights.size());

	// Get the sum of intensities (added up); might be better to use average?
	float dirIntensities = 0.f;
	float posIntensities = 0.f;
	for(const auto& light : dirLights)
		dirIntensities += ei::sum(get_dir_light_intensity(light, scale));
	for(const auto& light : posLights)
		posIntensities += ei::sum(std::visit(get_pos_light_intensity, light));

	// Copy the lights into the tree
	// TODO: how to place this best into memory?
	// Directional lights are easier, because they have a fixed size
	{
		std::memcpy(&tree.dirLights.handle[sizeof(DirNode) * dirNodes], dirLights.data(),
					sizeof(DirectionalLight) * dirLights.size());
	}
	// Positional lights are more difficult since we don't know the concrete size
	{
		std::size_t offset = sizeof(PosNode) * posNodes;
		char* mem = tree.posLights.handle;
		for(const auto& light : posLights) {
			std::visit([&mem, &offset](const auto& posLight) {
				std::memcpy(&mem[offset], &posLight, sizeof(posLight));
				offset += sizeof(posLight);
			}, light);
		}
	}

	// Directional lights can simply be copied
	{
		char* memory = tree.dirLights.handle;
		std::memcpy(&memory[sizeof(LightTree::DirNode) * dirNodes], dirLights.data(),
					sizeof(DirectionalLight) * dirLights.size());
	}
	// Positional lights can't just be copide since we don't know the concrete size
	{
		char* memory = tree.posLights.handle;
		std::size_t offset = sizeof(PosNode) * posNodes;
		for(const auto& light : posLights) {
			std::visit([&memory, &offset](const auto& posLight) {
				std::memcpy(&memory[offset], &posLight, sizeof(posLight));
				offset += sizeof(posLight);
			}, light);
		}
	}

	// Now we gotta construct the proper nodes by recursively merging them together
	create_directional_light_tree(dirLights, tree.dirLights.handle, scale);
	create_positional_light_tree(posLights, tree.posLights.handle);

	// Special casing for singular lights
	if(posLights.size() == 1u) {
		// It's kind of ugly having to keep this around, but hey...
		tree.singlePosLightType = std::visit(get_light_type, posLights.front());
	}
}

// TODO
/*void synchronize(const LightTree::Tree<Device::CPU>& changed, LightTree::Tree<Device::CUDA>& sync,
				 std::optional<textures::TextureHandle> hdl) {
	// Ensure that the node arrays are in sync (by alloc/realloc if necessary)
	// First directional lights...
	if(changed.numDirLights == 0u) {
		// Remove all data since there are no lights
		sync.dirLights.handle = Allocator<Device::CUDA>::free(sync.dirLights.handle, sync.numDirLights);
	} else {
		// Still have data, (re)alloc and copy
		if(sync.numDirLights == 0u || sync.dirLights.handle == nullptr) {
			mAssert(sync.dirLights.handle == nullptr);
			sync.dirLights.handle = Allocator<Device::CUDA>::alloc_array<LightTree::DirNode>(changed.numDirLights);
		} else {
			sync.dirLights.handle = Allocator<Device::CUDA>::realloc(sync.dirLights.handle, sync.numDirLights,
																	 changed.numDirLights);
		}
		// Copy over the data
		cudaMemcpy(sync.dirLights.handle, changed.dirLights.handle,
				   changed.numDirLights * sizeof(LightTree::DirNode),
				   cudaMemcpyHostToDevice);
	}
	// ...then positional lights
	if(changed.numPosLights == 0u) {
		// Remove all data since there are no lights
		sync.posLights.handle = Allocator<Device::CUDA>::free(sync.posLights.handle, sync.numPosLights);
	} else {
		// Still have data, (re)alloc and copy
		if(sync.numPosLights == 0u || sync.posLights.handle == nullptr) {
			mAssert(sync.posLights.handle == nullptr);
			sync.posLights.handle = Allocator<Device::CUDA>::alloc_array<LightTree::PosNode>(changed.numPosLights);
		} else {
			sync.posLights.handle = Allocator<Device::CUDA>::realloc(sync.posLights.handle, sync.numPosLights,
																	 changed.numPosLights);
		}
		// Copy over the data
		cudaMemcpy(sync.posLights.handle, changed.posLights.handle,
				   changed.numPosLights * sizeof(LightTree::PosNode),
				   cudaMemcpyHostToDevice);
	}

	sync.numDirLights = changed.numDirLights;
	sync.numPosLights = changed.numPosLights;
	// Also copy the environment light
	// TODO: the handle will be invalid!
	sync.envLight.flux = changed.envLight.flux;
	if(hdl.has_value())
		sync.envLight.texHandle = textures::ConstDeviceTextureHandle<Device::CUDA>{ *hdl.value()->aquireConst<Device::CUDA>() };
	else
		sync.envLight.texHandle = textures::ConstDeviceTextureHandle<Device::CUDA>{};
}

void synchronize(const LightTree::Tree<Device::CUDA>& changed, LightTree::Tree<Device::CPU>& sync,
				 std::optional<textures::TextureHandle> hdl) {
	// Ensure that the node arrays are in sync (by alloc/realloc if necessary)
	// First directional lights...
	if(changed.numDirLights == 0u) {
		// Remove all data since there are no lights
		sync.dirLights.handle = Allocator<Device::CPU>::free(sync.dirLights.handle, sync.numDirLights);
	} else {
		// Still have data, (re)alloc and copy
		if(sync.numDirLights == 0u || sync.dirLights.handle == nullptr) {
			mAssert(sync.dirLights.handle == nullptr);
			sync.dirLights.handle = Allocator<Device::CPU>::alloc_array<LightTree::DirNode>(changed.numDirLights);
		} else {
			sync.dirLights.handle = Allocator<Device::CPU>::realloc(sync.dirLights.handle, sync.numDirLights,
																	 changed.numDirLights);
		}
		// Copy over the data
		cudaMemcpy(sync.dirLights.handle, changed.dirLights.handle,
				   changed.numDirLights * sizeof(LightTree::DirNode),
				   cudaMemcpyDeviceToHost);
	}
	// ...then positional lights
	if(changed.numPosLights == 0u) {
		// Remove all data since there are no lights
		sync.posLights.handle = Allocator<Device::CPU>::free(sync.posLights.handle, sync.numPosLights);
	} else {
		// Still have data, (re)alloc and copy
		if(sync.numPosLights == 0u || sync.posLights.handle == nullptr) {
			mAssert(sync.posLights.handle == nullptr);
			sync.posLights.handle = Allocator<Device::CPU>::alloc_array<LightTree::PosNode>(changed.numPosLights);
		} else {
			sync.posLights.handle = Allocator<Device::CPU>::realloc(sync.posLights.handle, sync.numPosLights,
																	 changed.numPosLights);
		}
		// Copy over the data
		cudaMemcpy(sync.posLights.handle, changed.posLights.handle,
				   changed.numPosLights * sizeof(LightTree::PosNode),
				   cudaMemcpyDeviceToHost);
	}

	sync.numDirLights = changed.numDirLights;
	sync.numPosLights = changed.numPosLights;
	// Also copy the environment light and refresh the handle
	sync.envLight.flux = changed.envLight.flux;
	if(hdl.has_value())
		sync.envLight.texHandle = textures::ConstDeviceTextureHandle<Device::CPU>{ *hdl.value()->aquireConst<Device::CPU>() };
	else
		sync.envLight.texHandle = textures::ConstDeviceTextureHandle<Device::CPU>{};
}

void unload(LightTree::Tree<Device::CPU>& tree) {
	tree.dirLights.handle =	Allocator<Device::CPU>::free(tree.dirLights.handle, tree.numDirLights);
	tree.posLights.handle = Allocator<Device::CPU>::free(tree.posLights.handle, tree.numPosLights);
	// TODO: unload envmap handle
}
void unload(LightTree::Tree<Device::CUDA>& tree) {
	tree.dirLights.handle = Allocator<Device::CUDA>::free(tree.dirLights.handle, tree.numDirLights);
	tree.posLights.handle = Allocator<Device::CUDA>::free(tree.posLights.handle, tree.numPosLights);
	// TODO: unload envmap handle
}*/

}}} // namespace mufflon::scene::lights