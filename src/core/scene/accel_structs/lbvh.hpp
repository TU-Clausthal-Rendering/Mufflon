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
	i32 bvhSize;
};

//
class LBVHBuilder {
public:
	template < Device dev >
	void build(ObjectDescriptor<dev>& obj, const ei::Box& aabb);

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
		lbvhDesc.primIds = as<i32>( m_bvhNodes.acquire_const<dev>() );
		lbvhDesc.bvhSize = int(m_bvhNodes.size() / sizeof(ei::Vec4));
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

	template < Device dev >
	void build_lbvh32(ei::Mat3x4* matrixs,//matrices
		i32* objIds,
		ei::Box* aabbs,
		const ei::Box& sceneBB,
		ei::Vec2 traverseCosts, i32 numInstances);
};

template DeviceManagerConcept<LBVHBuilder>;


/**
 * LBVH class.
 * Linear Bounding Volume Hiearchie.
 */
/*class LBVH:
	public IAccelerationStructure {
public:
	LBVH();
	~LBVH();

	// Checks whether the structure is currently available on the given system.
	bool is_resident(Device res) const final;
	// TODO: Makes the structure's data available on the desired system.
	// TODO: hybrid Device allowed?
	void make_resident(Device res) final;
	// Removes the structure from the given system, if present.
	void unload_resident(Device res) final;
	// TODO: Builds or rebuilds the structure.
	void build(const std::vector<InstanceHandle>&) final;
	// TODO: should this be put into a different class?
	void build(ObjectData data) final;
	// TODO: Checks whether the data on a given system has been modified and is out of sync.
	virtual bool is_dirty(Device res) const final;
private:

	// Indicator for the location of the structure.
	Device m_device;

	// Layout of collapsedBVH:
	// - Notation: c0: left child; c1: right child.
	// Internal node (64 bytes):
	// 1. ei::Vec4: c0.lo.x, c0.hi.x, c0.lo.y, c0.hi.y
	// 2. ei::Vec4: c1.lo.x, c1.hi.x, c1.lo.y, c1.hi.y
	// 3. ei::Vec4: c0.lo.z, c0.hi.z, c1.lo.z, c1.hi.z
	// 4. ei::Vec4: c0, c1
	// If cx < 0, then it points to a leaf.
	// - Leaf with more than 1 primitives (16 bytes):
	// 1. 32 bits: 2bits + 10bits (numTriangles) + 10bits (numQuads) + 10bits (numSpheres).
	// 2. 32 bits: offset for triangles.
	// 3. 32 bits: offset for quads.
	// 4. 32 bits: offset for spheres.
	// - Leaf with only one primitive will not be stored. cx takes primId as its value.
	AccelStructInfo::Size m_sizes{};
	AccelStructInfo::InputArrays m_inputCUDA{};
	AccelStructInfo::InputArrays m_inputCPU{};
	AccelStructInfo::OutputArrays m_outputCUDA{};
	AccelStructInfo::OutputArrays m_outputCPU{};
};*/

}}}

