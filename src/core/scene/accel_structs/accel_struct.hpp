#pragma once

//#include "core/scene/instance.hpp"
//#include "core/scene/handles.hpp"
//#include "core/memory/residency.hpp"
//#include <OpenMesh/Core/Mesh/PolyConnectivity.hh>
#include <cstdlib>

// Forward declarations
namespace ei {
struct Ray;
} // namespace ei

namespace mufflon { namespace scene {

// Forward declaration
struct ObjectData;

namespace accel_struct {

// All supported types of acceleration structures
enum class AccelType {
	NONE,
	LBVH
};

constexpr std::size_t MAX_ACCEL_STRUCT_PARAMETER_SIZE = 24;

/**
 * Interface for generic accelleration structure.
 */
/*class IAccelerationStructure {
public:
	IAccelerationStructure() = default;
	IAccelerationStructure(const IAccelerationStructure&) = default;
	IAccelerationStructure(IAccelerationStructure&&) = default;
	IAccelerationStructure& operator=(const IAccelerationStructure&) = default;
	IAccelerationStructure& operator=(IAccelerationStructure&&) = default;
	virtual ~IAccelerationStructure() = default;

	// Checks whether the structure is currently available on the given system.
	virtual bool is_resident(Device res) const = 0;
	// Makes the structure's data available on the desired system.
	virtual void make_resident(Device res) = 0;
	// Removes the structure from the given system, if present.
	virtual void unload_resident(Device res) = 0;
	// Builds or rebuilds the structure.
	virtual void build(const std::vector<InstanceHandle>&) = 0;
	// TODO: should this be put into a different class?
	virtual void build(ObjectData data) = 0;
	// Checks whether the data on a given system has been modified and is out of sync.
	virtual bool is_dirty(Device res) const = 0;

	// TODO: intersections for Rays
};*/

}}} // namespace mufflon::scene::accel_struct