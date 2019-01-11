#pragma once

#include "lights.hpp"
#include "core/export/api.h"
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

namespace mufflon { namespace scene {

template < Device dev >
struct SceneDescriptor;

namespace lights {

	class Background;

#ifndef __CUDACC__
// Kind of code duplication, but for type-safety use this when constructing a light tree
struct PositionalLights {
	std::variant<PointLight, SpotLight, AreaLightTriangleDesc,
				 AreaLightQuadDesc, AreaLightSphereDesc> light;
	PrimitiveHandle primitive { ~0u };
};

// Gets the light type as an enum value
inline LightType get_light_type(const PositionalLights& light) {
	return std::visit([](const auto& posLight) constexpr -> LightType {
		using Type = std::decay_t<decltype(posLight)>;
		if constexpr(std::is_same_v<Type, PointLight>)
			return LightType::POINT_LIGHT;
		else if constexpr(std::is_same_v<Type, SpotLight>)
			return LightType::SPOT_LIGHT;
		else if constexpr(std::is_same_v<Type, AreaLightTriangleDesc>)
			return LightType::AREA_LIGHT_TRIANGLE;
		else if constexpr(std::is_same_v<Type, AreaLightQuadDesc>)
			return LightType::AREA_LIGHT_QUAD;
		else if constexpr(std::is_same_v<Type, AreaLightSphereDesc>)
			return LightType::AREA_LIGHT_SPHERE;
		else
			return LightType::NUM_LIGHTS;
	}, light.light);
}

#endif // __CUDACC__

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

	BackgroundDesc<dev> background;
	// Pointer to the tree elements
	LightSubTree dirLights;
	LightSubTree posLights;
	// A map to find the node of a given primitve.
	// The map stores an encoded path to the node. Storing its pointer/offset
	// would be useless in terms of finding its probability. Therefore,
	// the tree must be traversed
	HashMap<dev, PrimitiveHandle, u32> primToNodePath;
};

#ifndef __CUDACC__
class LightTreeBuilder {
public:
	LightTreeBuilder();
	~LightTreeBuilder();
	LightTreeBuilder(LightTreeBuilder&&) = default;
	LightTreeBuilder& operator=(LightTreeBuilder&&) = default;

	// Builds the light tree from lists of positional and directional lights and
	// resets the envmap light to black
	void build(std::vector<PositionalLights>&& posLights,
			   std::vector<DirectionalLight>&& dirLights,
			   const ei::Box& boundingBox);
	// Updates (read replaces) the envmap light only
	void set_envLight(Background& envLight) {
		m_envLight = &envLight;
	}

	const Background* get_envLight() const noexcept {
		return m_envLight;
	}

	// Determines for each point- and spotlight in what medium it is
	template < Device dev >
	void update_media(const SceneDescriptor<dev>& descriptor);

	template < Device dev >
	const LightTree<dev>& acquire_const(const ei::Box& sceneBounds) {
		this->synchronize<dev>(sceneBounds);
		if constexpr(dev == Device::CPU) return *m_treeCpu;
		else return *m_treeCuda;
	}

	template < Device dev >
	void synchronize(const ei::Box& sceneBounds);

	template < Device dev >
	void unload();

	u32 get_light_count() const noexcept {
		return m_lightCount;
	}

private:
	void update_media_cpu(const SceneDescriptor<Device::CPU>& scene);
	void remap_textures(const char* cpuMem, u32 offset, u16 type, char* cudaMem);

	// Keep track of the light count (positional and directional combined)
	u32 m_lightCount = 0u;

	util::DirtyFlags<Device> m_dirty;
	std::unique_ptr<LightTree<Device::CPU>> m_treeCpu;
	std::unique_ptr<LightTree<Device::CUDA>> m_treeCuda;
	HashMapManager<PrimitiveHandle, u32> m_primToNodePath;
	GenericResource m_treeMemory;
	// The tree is build on CPU side. For synchronization we need a possiblity to
	// find the CUDA textures.
	std::unordered_map<textures::ConstTextureDevHandle_t<Device::CPU>, TextureHandle> m_textureMap;
	// Environment light model, may be black, a texture or an analytic model
	lights::Background* m_envLight { nullptr };
};

template DeviceManagerConcept<LightTreeBuilder>;

#endif // __CUDACC__


namespace lighttree_detail {
	void update_media_cuda(const SceneDescriptor<Device::CUDA>& scene, const LightSubTree& posLights);

	std::size_t get_num_internal_nodes(std::size_t elems);

	CUDA_FUNCTION __forceinline__ ei::Vec3 get_center(const void* node, u16 type) {
		switch(type) {
			case u16(LightType::POINT_LIGHT): return get_center(*as<PointLight>(node));
			case u16(LightType::SPOT_LIGHT): return get_center(*as<SpotLight>(node));
			case u16(LightType::AREA_LIGHT_TRIANGLE): return get_center(*as<AreaLightTriangle<CURRENT_DEV>>(node));
			case u16(LightType::AREA_LIGHT_QUAD): return get_center(*as<AreaLightQuad<CURRENT_DEV>>(node));
			case u16(LightType::AREA_LIGHT_SPHERE): return get_center(*as<AreaLightSphere<CURRENT_DEV>>(node));
			case u16(LightType::DIRECTIONAL_LIGHT): return get_center(*as<DirectionalLight>(node));
			case u16(LightType::ENVMAP_LIGHT): return get_center(*as<BackgroundDesc<CURRENT_DEV>>(node));
			case LightSubTree::Node::INTERNAL_NODE_TYPE: return as<LightSubTree::Node>(node)->center;
		}
		mAssert(false);
		return ei::Vec3{0.0f};
	}
}

}}} // namespace mufflon::scene::lights
