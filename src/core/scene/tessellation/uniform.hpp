#pragma once

#include "tessellater.hpp"

namespace mufflon::scene::tessellation {

class Uniform : public Tessellater {
public:
	Uniform() = default;
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
	virtual u32 get_edge_tessellation_level(const OpenMesh::EdgeHandle edge) const override {
		return m_outerLevel;
	}

	virtual u32 get_inner_tessellation_level(const OpenMesh::FaceHandle face) const override {
		return m_innerLevel;
	}

private:
	u32 m_innerLevel;
	u32 m_outerLevel;
};

} // namespace mufflon::scene::tessellation 