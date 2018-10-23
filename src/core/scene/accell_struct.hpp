#pragma once

#include "residency.hpp"

namespace mufflon::scene {

class IAccellerationStructure {
public:
	virtual bool is_resident(Residency res) const = 0;
	virtual void make_resident(Residency res) = 0;
	virtual void unload_resident(Residency res) = 0;
	virtual void build() = 0;
	virtual bool is_dirty(Residency res) const = 0;
	virtual void mark_dirty(Residency res) = 0;
};

} // namespace mufflon::scene