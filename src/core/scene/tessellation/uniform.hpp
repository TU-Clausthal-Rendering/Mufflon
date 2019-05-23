#pragma once

#include "tessellater.hpp"

namespace mufflon::scene::tessellation {

class Uniform : public TessLevelOracle {
public:
	Uniform() = default;
	Uniform(const u32 level) :
		Uniform(level, level) {}
	Uniform(const u32 innerLevel, const u32 outerLevel) :
		m_innerLevel(innerLevel),
		m_outerLevel(outerLevel) {}
	Uniform(const Uniform&) = delete;
	Uniform(Uniform&&) = delete;
	Uniform& operator=(const Uniform&) = delete;
	Uniform& operator=(Uniform&&) = delete;
	virtual ~Uniform() = default;

	void set_inner_level(u32 level) {
		m_innerLevel = level;
	}

	void set_outer_level(u32 level) {
		m_outerLevel = level;
	}

protected:
	u32 get_edge_tessellation_level(const OpenMesh::EdgeHandle edge) const override {
		return m_outerLevel;
	}

	u32 get_inner_tessellation_level(const OpenMesh::FaceHandle face) const override {
		return m_innerLevel;
	}

private:
	u32 m_innerLevel = 0u;
	u32 m_outerLevel = 0u;
};

} // namespace mufflon::scene::tessellation 