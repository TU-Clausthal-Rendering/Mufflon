#pragma once

#include "core/concepts.hpp"
#include "core/scene/descriptors.hpp"
#include "core/memory/generic_resource.hpp"
#include "core/memory/residency.hpp"
#include "accel_struct_info.hpp"

namespace mufflon { namespace scene { namespace accel_struct {

// Structure specific descriptor
// Necessary because CUDA and MSVC disagreed about the size
#pragma pack(push, 1)
struct BvhNode {
	ei::Box bb;
	i32 index;
	i32 primCount;
};
template < Device dev >
struct LBVH {
	ConstArrayDevHandle_t<NotGl<dev>(), BvhNode> bvh;
	ConstArrayDevHandle_t<dev, i32> primIds;
	i32 numInternalNodes;
};
#pragma pack(pop)

/*
 * At first, a normal LBVH is build. See Karras "TODO"
 * Then the lower nodes are collapsed based on SAH values. 
 * It ensures that there is at least one internal node after collapsing if
 * there are at least two instances.
 * If there is only one instance, no BVH is built.
 *
 * Layout of collapsedBVH:
 * Internal node (64 bytes)
 *  1. Vec4: L.bbmin.x, L.bbmin.y, L.bbmin.z, cL
 *  2. Vec4: L.bbmax.x, L.bbmax.y, L.bbmax.z, primCountL
 *  3. Vec4: R.bbmin.x, R.bbmin.y, R.bbmin.z, cR
 *  4. Vec4: R.bbmax.x, R.bbmax.y, R.bbmax.z, primCountR
 * If cX > numInternalNodes it points to a 'leaf', otherwise index of internal node
 *  'leaf' means that cX-numInternalNodes is the index in the primIds array with
 *  primCountX consecutive elements. (given by the last two values
 *  in the internal node.
 */
class LBVHBuilder {
public:
	LBVHBuilder() = default;

	LBVHBuilder(LBVHBuilder& lbvh) {
		// Warning: the copy implicitly syncs!
		m_primIds.resize(lbvh.m_primIds.size());
		m_bvhNodes.resize(lbvh.m_bvhNodes.size());

		if(lbvh.m_primIds.is_resident<Device::CPU>()) {
			const char* primMem = lbvh.m_primIds.template acquire_const<Device::CPU>();
			copy(m_primIds.template acquire<Device::CPU>(), primMem, lbvh.m_primIds.size());
		}
		if(lbvh.m_bvhNodes.is_resident<Device::CPU>()) {
			const char* bvhMem = lbvh.m_bvhNodes.template acquire_const<Device::CPU>();
			copy(m_bvhNodes.template acquire<Device::CPU>(), bvhMem, lbvh.m_bvhNodes.size());
		}
	}

	template < Device dev >
	void build(LodDescriptor<dev>& obj, const ei::Box& currentBB, const bool parallelize);

	template < Device dev >
	void build(const SceneDescriptor<dev>& scene, const u32 actualInstCount = 0u);

	template < Device dev >
	AccelDescriptor acquire_const() {
		if (dev == Device::OPENGL)
			return AccelDescriptor{}; // no valid descriptor for opengl yet

		if(needs_rebuild())
			throw std::runtime_error("[LBVHBuilder::acquire_const] the BVH must be created with build() before a descriptor can be returned.");
		synchronize<dev>();

		AccelDescriptor desc;
		desc.type = AccelType::LBVH;
		LBVH<dev>& lbvhDesc = *as<LBVH<dev>>(desc.accelParameters);
		lbvhDesc.bvh = as<ConstArrayDevHandle_t<NotGl<dev>(), BvhNode>, ConstArrayDevHandle_t<dev, char>>( m_bvhNodes.acquire_const<dev>() );
		lbvhDesc.primIds = as<ConstArrayDevHandle_t<dev, i32>, ConstArrayDevHandle_t<dev, char>>( m_primIds.acquire_const<dev>() );
		lbvhDesc.numInternalNodes = int(m_bvhNodes.size() / (4 * sizeof(ei::Vec4)));
		return desc;
	}

	template < Device dev >
	void unload() {
		m_primIds.unload<dev>();
		m_bvhNodes.unload<dev>();
	}

	template < Device dev >
	void synchronize() {
		m_primIds.synchronize<dev>();
		m_bvhNodes.synchronize<dev>();
	}

	bool needs_rebuild() const noexcept {
		return (!m_primIds.template is_resident<Device::CPU>() || !m_bvhNodes.template is_resident<Device::CPU>())
			&& (!m_primIds.template is_resident<Device::CUDA>() || !m_bvhNodes.template is_resident<Device::CUDA>());
	}

	void mark_invalid() noexcept {
		unload<Device::CPU>();
		unload<Device::CUDA>();
	}

	// Gets the size of the final descriptor
	std::size_t desciptor_size() const noexcept {
		return m_primIds.size() + m_bvhNodes.size();
	}

private:
	GenericResource m_primIds;
	GenericResource m_bvhNodes;

	// The internal build kernel for both kinds of hierarchies
	template < typename DescType >
	void build_lbvh(const DescType& desc,
					const ei::Box& sceneBB,
					const i32 numPrimitives,
					const bool parallelize,
					const u32 actualPrimCount);
};

}} // namespace scene::accel_struct

template struct DeviceManagerConcept<scene::accel_struct::LBVHBuilder>;

} // namespace mufflon
