#pragma once

#include "util/types.hpp"
#include "core/scene/types.hpp"

namespace mufflon { namespace scene { namespace textures {

class CpuTexture {
public:
	// TODO: creating texture and submitting data
	CpuTexture(u16 width, u16 height, u16 numLayers) {}
	CpuTexture(const CpuTexture&) = delete;
	CpuTexture(CpuTexture&&) = default;
	CpuTexture& operator=(const CpuTexture&) = delete;
	CpuTexture& operator=(CpuTexture&&) = default;
	~CpuTexture() = default;

	/*
	 * Get an (interpolated) texture sample at the given coordinate.
	 * The border handling mode is always wrap. I.e. modulu operation is performed on the
	 * texture coordinates.
	 */
	ei::Vec4 sample(const UvCoordinate& uv);

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