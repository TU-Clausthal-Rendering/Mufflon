#pragma once

#include "residency.hpp"

namespace mufflon::scene {

/**
 * Interface for generic accelleration structure.
 */
class IAccellerationStructure {
public:
	IAccellerationStructure() = default;
	IAccellerationStructure(const IAccellerationStructure&) = default;
	IAccellerationStructure(IAccellerationStructure&&) = default;
	IAccellerationStructure& operator=(const IAccellerationStructure&) = default;
	IAccellerationStructure& operator=(IAccellerationStructure&&) = default;
	virtual ~IAccellerationStructure() = default;

	/// Checks whether the structure is currently available on the given system.
	virtual bool is_resident(Residency res) const = 0;
	/// Makes the structure's data available on the desired system.
	virtual void make_resident(Residency res) = 0;
	/// Removes the structure from the given system, if present.
	virtual void unload_resident(Residency res) = 0;
	/// Builds or rebuilds the structure.
	virtual void build() = 0;
	/// Checks whether the data on a given system has been modified and is out of sync.
	virtual bool is_dirty(Residency res) const = 0;
};

} // namespace mufflon::scene