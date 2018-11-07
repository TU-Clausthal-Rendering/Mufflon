#pragma once

#include "util/types.hpp"
#include "core/scene/types.hpp"
#include "texture.hpp"

namespace mufflon { namespace scene { namespace textures {

class CpuTexture {
public:
	// Allocate the texture, data must be filled by using the data() pointer (or writes).
	CpuTexture(u16 width, u16 height, u16 numLayers, Format format, SamplingMode mode);
	CpuTexture(const CpuTexture&) = delete;
	CpuTexture(CpuTexture&&) = default;
	CpuTexture& operator=(const CpuTexture&) = delete;
	CpuTexture& operator=(CpuTexture&&) = default;
	~CpuTexture() = default;

	/*
	 * Get the value of a texel converted to normalized float.
	 * If the format does not define all output channels RGBA those are defaulted
	 * to (0,0,0,1). The border handling mode is always wrap. I.e. modulu operation
	 * is performed on the texture coordinates.
	 */
	ei::Vec4 read(const Pixel& texel, int layer = 0) const;

	/*
	 * Get an (interpolated) texture sample at the given coordinate.
	 * The border handling mode is always wrap. I.e. modulu operation is performed on the
	 * texture coordinates.
	 */
	ei::Vec4 sample(const UvCoordinate& uv, int layer = 0) const {
		return (this->*m_sample)(uv, layer);
	}

	i32 get_width() const noexcept { return m_size.x; }
	i32 get_height() const noexcept { return m_size.y; }
	i32 get_num_layers() const noexcept { return m_size.z; }

	u8* data() noexcept { return m_imageData.data(); }
	const u8* data() const noexcept { return m_imageData.data(); }
private:
	std::vector<u8> m_imageData;
	Format m_format;
	ei::IVec3 m_size;		// width, height, numLayers
	int m_numComponents;	// Number of components/channels per pixel for the format. Special formats like RGB9E5 have only one component per pixel but multiple channels.

	ei::Vec4 (CpuTexture::* m_fetch)(int componentIdx) const;
	ei::Vec4 fetch_R8U(int componentIdx) const;
	ei::Vec4 fetch_RG8U(int componentIdx) const;
	ei::Vec4 fetch_RGB8U(int componentIdx) const;
	ei::Vec4 fetch_RGBA8U(int componentIdx) const;
	ei::Vec4 fetch_R16U(int componentIdx) const;
	ei::Vec4 fetch_RG16U(int componentIdx) const;
	ei::Vec4 fetch_RGB16U(int componentIdx) const;
	ei::Vec4 fetch_RGBA16U(int componentIdx) const;
	ei::Vec4 fetch_R32F(int componentIdx) const;
	ei::Vec4 fetch_RG32F(int componentIdx) const;
	ei::Vec4 fetch_RGB32F(int componentIdx) const;
	ei::Vec4 fetch_RGBA32F(int componentIdx) const;
	ei::Vec4 fetch_RGB9E5(int componentIdx) const;

	ei::Vec4 (CpuTexture::* m_sample)(const UvCoordinate& uv, int layer) const;
	ei::Vec4 sample_nearest(const UvCoordinate& uv, int layer) const;
	ei::Vec4 sample_linear(const UvCoordinate& uv, int layer) const;
	// Faster methods for 1x1x1 textures
	ei::Vec4 sample111_nearest(const UvCoordinate& uv, int layer) const;
	ei::Vec4 sample111_linear(const UvCoordinate& uv, int layer) const;

	inline int get_index(const ei::IVec3 & texel) const { return (texel.x + (texel.y + texel.z * m_size.y) * m_size.x) * m_numComponents; }
};

}}} // namespace mufflon::scene::textures