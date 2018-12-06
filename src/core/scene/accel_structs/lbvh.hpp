#pragma once

#include "core/scene/accel_structs/accel_struct.hpp"
#include "accel_structs_commons.hpp"

namespace mufflon { namespace scene { namespace accel_struct {


/**
 * LBVH class.
 * Linear Bounding Volume Hiearchie.
 */
class LBVH:
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
};

}}}

