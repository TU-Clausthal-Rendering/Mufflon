#pragma once

#include "lights.hpp"
#include "core/export/core_api.h"
#include "core/memory/allocator.hpp"
#include "core/memory/residency.hpp"
#include "core/memory/synchronize.hpp"
#include "core/memory/dyntype_memory.hpp"
#include "core/memory/hashmap.hpp"
#include "core/scene/handles.hpp"
#include "core/scene/types.hpp"
#include "core/scene/textures/texture.hpp"
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
	PrimitiveHandle primitive;
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

		Node(const char* base,
			 u32 leftOffset, u16 leftType,
			 u32 rightOffset, u16 rightType,
			 const ei::Vec3& bounds,
			 const int* materials);

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
		float flux{ 0.0f };
		u16 type;
	} root;
	u32 internalNodeCount{ 0u };
	std::size_t lightCount{ 0u };
	char* memory{ nullptr };

	CUDA_FUNCTION __forceinline__ Node* get_node(u32 offset) { return as<Node>(memory + offset); }
	CUDA_FUNCTION __forceinline__ const Node* get_node(u32 offset) const { return as<Node>(memory + offset); }

	CUDA_FUNCTION std::pair<const char*, LightType> get_light_info(const u32 index) const {
		if(lightCount == 1u)
			return { memory, static_cast<LightType>(root.type) };
		const u32 nodeIndex = index + internalNodeCount;
		const u32 parentIndex = (nodeIndex - 1u) >> 1u; // Account for root node
		const Node& parent = *get_node(parentIndex * sizeof(Node));
		if(nodeIndex & 1u) // Left/right are toggled due to root node
			return { memory + parent.left.offset, static_cast<LightType>(parent.left.type) };
		else
			return { memory + parent.right.offset, static_cast<LightType>(parent.right.type) };
	}
};

//using GuideFunction = float (*)(const scene::Point&, const scene::Point&, const scene::Point&, float, float);

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
	bool posGuide;

	// Get the total flux of all lights
	inline CUDA_FUNCTION float get_flux() const {
		return dirLights.root.flux + posLights.root.flux + ei::sum(background.flux);
	}
};

template<>
struct LightTree<Device::OPENGL> {
	static constexpr Device DEVICE = Device::OPENGL;
    
    struct SmallLight {
        // for point light and spot
		ei::Vec3 intensity;
		u32 type;

		ei::Vec3 position;
		union { float radius; float cosFalloffStart; };
		
        ei::Vec3 direction;
		union { u32 material; float cosThetaMax; };
    };

	static_assert(sizeof(SmallLight) == 4 * 4 * 3, "SmallLight invalid alignment");

    struct BigLight {
        // for area lights
		ei::Vec3 pos;
		u32 material;

		ei::Vec3 v1; // to v1
		u32 numPoints; // 3 or 4

		ei::Vec3 v2; // to v2
		float dummy0;

		ei::Vec3 v3; // to v3
		float dummy1;
    };

	LightTree(const std::vector<SmallLight>& smallLights, const std::vector<BigLight>& bigLights);
	LightTree() noexcept { smallLights.id = 0; bigLights.id = 0; }
	//~LightTree();

	ArrayDevHandle_t<DEVICE, SmallLight> smallLights;
	ArrayDevHandle_t<DEVICE, BigLight> bigLights;
	u32 numSmallLights;
	u32 numBigLights;
};

#ifndef __CUDACC__
class LightTreeBuilder {
public:
	LightTreeBuilder();
	~LightTreeBuilder();

	// Builds the light tree from lists of positional and directional lights and
	// resets the envmap light to black
	void build(std::vector<PositionalLights>&& posLights,
			   std::vector<DirectionalLight>&& dirLights,
			   const ei::Box& boundingBox,
			   const int* materials);
	// Updates (read replaces) the envmap light only
	void set_envLight(Background& envLight) {
		m_envLight = &envLight;
		m_backgroundDirty = true;
	}

	const Background* get_envLight() const noexcept {
		return m_envLight;
	}

	// Marks the background as "dirty" (which will determine whether it needs to be replaced by
	// the world container, not whether e.g. the texture needs to be synchronized); this gets
	// reset upon synchronize
	bool is_background_dirty() const noexcept {
		return m_backgroundDirty;
	}

	// Determines for each point- and spotlight in what medium it is
	template < Device dev >
	void update_media(const SceneDescriptor<dev>& descriptor);

	template < Device dev >
	const LightTree<dev>& acquire_const(const ei::Box& sceneBounds) {
		this->synchronize<dev>(sceneBounds);
		if constexpr(dev == Device::CPU) return *m_treeCpu;
		else if constexpr(dev == Device::OPENGL) return *m_treeOpengl;
		else return *m_treeCuda;
	}

	std::size_t descriptor_size() const noexcept;

	template < Device dev >
	void synchronize(const ei::Box& sceneBounds);

	template < Device dev >
	void unload();

	u32 get_light_count() const noexcept {
		return m_lightCount;
	}

	template < Device dev >
	bool is_resident() const {
		if constexpr(dev == Device::CPU) return m_treeCpu != nullptr;
		if constexpr(dev == Device::CUDA) return m_treeCuda != nullptr;
		if constexpr(dev == Device::OPENGL) return m_treeOpengl != nullptr;
		return false;
	}

	bool is_resident_anywhere() const {
		return is_resident<Device::CPU>() || is_resident<Device::CUDA>() || is_resident<Device::OPENGL>();
	}

	// Get the function pointer for flux or fluxPos guide on the target device.
//	template < Device dev >
//	static GuideFunction get_guide_fptr(bool posGuide);
private:
	void update_media_cpu(const SceneDescriptor<Device::CPU>& scene);

	// Keep track of the light count (positional and directional combined)
	u32 m_lightCount = 0u;

	std::unique_ptr<LightTree<Device::CPU>> m_treeCpu;
	std::unique_ptr<LightTree<Device::CUDA>> m_treeCuda;
	std::unique_ptr<LightTree<Device::OPENGL>> m_treeOpengl;
	HashMapManager<PrimitiveHandle, u32> m_primToNodePath;
	GenericResource m_treeMemory;
	// Environment light model, may be black, a texture or an analytic model
	lights::Background* m_envLight{ nullptr };

	bool m_backgroundDirty = true;
};

}} // namespace scene::lights

namespace scene { namespace lights {

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
