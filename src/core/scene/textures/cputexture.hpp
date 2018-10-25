#pragma once

#include "util/types.hpp"

namespace mufflon { namespace scene { namespace textures {

class CPUTexture {
public:
	// TODO: creating texture and submitting data
	CPUTexture(u16 width, u16 height, u16 numLayers) {}
	CPUTexture(const CPUTexture&) = delete;
	CPUTexture(CPUTexture&&) = default;
	CPUTexture& operator=(const CPUTexture&) = delete;
	CPUTexture& operator=(CPUTexture&&) = default;
	~CPUTexture() = default;

	// TODO: sample(uv)

	i32 get_width() const noexcept { return m_width; }
	i32 get_height() const noexcept { return m_height; }
	i32 get_num_layers() const noexcept { return m_numLayers; }
private:
	u16 m_width;
	u16 m_height;
	u16 m_numLayers;
	// TODO: format, data, ...
};

}}} // namespace mufflon::scene::textures