#pragma once

#include "residency.hpp"

// forward declaration
namespace ei {
struct Ray;
} // namespace ei

namespace mufflon::scene {

/**
 * Interface for generic accelleration structure.
 */
class IAccelerationStructure {
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
	virtual void build() = 0;
	// Checks whether the data on a given system has been modified and is out of sync.
	virtual bool is_dirty(Device res) const = 0;

	// TODO: intersections for Rays
};

} // namespace mufflon::scene