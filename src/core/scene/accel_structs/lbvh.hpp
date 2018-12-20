#pragma once

#include "core/concepts.hpp"
#include "core/scene/descriptors.hpp"
#include "core/memory/generic_resource.hpp"
#include "util/flag.hpp"
#include "accel_struct_info.hpp"

namespace mufflon { namespace scene { namespace accel_struct {

// Structure specific descriptor
struct LBVH {
	const ei::Vec4* bvh;
	const i32* primIds;
	i32 bvhSize; // TODO: still required?
	i32 numInternalNodes;
};

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
	template < Device dev >
	void build(ObjectDescriptor<dev>& obj, const ei::Box& sceneBB);

	template < Device dev >
	void build(const SceneDescriptor<dev>& scene);

	template < Device dev >
	AccelDescriptor acquire_const() {
		if(needs_rebuild<dev>())
			throw std::runtime_error("[LBVHBuilder::acquire_const] the BVH must be created with build() before a descriptor can be returned.");
		synchronize<dev>();

		AccelDescriptor desc;
		desc.type = AccelType::LBVH;
		LBVH& lbvhDesc = *as<LBVH>(desc.accelParameters);
		lbvhDesc.bvh = as<ei::Vec4>( m_bvhNodes.acquire_const<dev>() );
		lbvhDesc.primIds = as<i32>( m_primIds.acquire_const<dev>() );
		lbvhDesc.bvhSize = int(m_bvhNodes.size() / sizeof(ei::Vec4));
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

	template < Device dev >
	bool needs_rebuild() const {
		return (!m_primIds.is_resident<Device::CPU>() || !m_bvhNodes.is_resident<Device::CPU>())
			&& (!m_primIds.is_resident<Device::CUDA>() || !m_bvhNodes.is_resident<Device::CUDA>());
	}

	void mark_invalid() noexcept {
		unload<Device::CPU>();
		unload<Device::CUDA>();
	}
private:
	GenericResource m_primIds;
	GenericResource m_bvhNodes;

	// The internal build kernel for both kinds of hierarchies
	template < typename DescType >
	void build_lbvh(const DescType& desc,
					const ei::Box& sceneBB,
					const i32 numPrimitives);
};

template DeviceManagerConcept<LBVHBuilder>;

}}} // namespace mufflon::scene::accel_struct
