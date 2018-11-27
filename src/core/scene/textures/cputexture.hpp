#pragma once

#include "util/types.hpp"
#include "core/scene/types.hpp"
#include "texture.hpp"
#include <memory>

namespace mufflon { namespace scene { namespace textures {

// TODO: cubemap access?
// TODO: Mipmapping?
class CpuTexture {
public:
	// Allocate the texture, data must be filled by using the data() pointer (or writes).
	// Takes ownership if pointer is given
	CpuTexture(u16 width, u16 height, u16 numLayers, Format format, SamplingMode mode,
			   bool sRgb, u8* data = nullptr);
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
	 * Overwrite a texel value.
	 * The input is converted to the internal format. If necessary this includes clamping
	 * of the value range.
	 * The border handling mode is always wrap. I.e. modulu operation is performed on the
	 * texture coordinates.
	 */
	void write(const ei::Vec4& value, const Pixel& texel, int layer = 0);

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

	u8* data() noexcept { return m_imageData.get(); }
	const u8* data() const noexcept { return m_imageData.get(); }
private:
	std::unique_ptr<u8[]> m_imageData;
	Format m_format;
	ei::IVec3 m_size;		// width, height, numLayers

	ei::Vec4 (CpuTexture::* m_fetch)(int texelIdx) const;
	ei::Vec4 fetch_R8U(int texelIdx) const;
	ei::Vec4 fetch_RG8U(int texelIdx) const;
	ei::Vec4 fetch_RGB8U(int texelIdx) const;
	ei::Vec4 fetch_RGBA8U(int texelIdx) const;
	ei::Vec4 fetch_sR8U(int texelIdx) const;
	ei::Vec4 fetch_sRG8U(int texelIdx) const;
	ei::Vec4 fetch_sRGB8U(int texelIdx) const;
	ei::Vec4 fetch_sRGBA8U(int texelIdx) const;
	ei::Vec4 fetch_R16U(int texelIdx) const;
	ei::Vec4 fetch_RG16U(int texelIdx) const;
	ei::Vec4 fetch_RGB16U(int texelIdx) const;
	ei::Vec4 fetch_RGBA16U(int texelIdx) const;
	ei::Vec4 fetch_R32F(int texelIdx) const;
	ei::Vec4 fetch_RG32F(int texelIdx) const;
	ei::Vec4 fetch_RGB32F(int texelIdx) const;
	ei::Vec4 fetch_RGBA32F(int texelIdx) const;
	ei::Vec4 fetch_RGB9E5(int texelIdx) const;

	void (CpuTexture::* m_write)(int texelIdx, const ei::Vec4& value);
	void write_R8U(int texelIdx, const ei::Vec4& value);
	void write_RG8U(int texelIdx, const ei::Vec4& value);
	void write_RGB8U(int texelIdx, const ei::Vec4& value);
	void write_RGBA8U(int texelIdx, const ei::Vec4& value);
	void write_sR8U(int texelIdx, const ei::Vec4& value);
	void write_sRG8U(int texelIdx, const ei::Vec4& value);
	void write_sRGB8U(int texelIdx, const ei::Vec4& value);
	void write_sRGBA8U(int texelIdx, const ei::Vec4& value);
	void write_R16U(int texelIdx, const ei::Vec4& value);
	void write_RG16U(int texelIdx, const ei::Vec4& value);
	void write_RGB16U(int texelIdx, const ei::Vec4& value);
	void write_RGBA16U(int texelIdx, const ei::Vec4& value);
	void write_R32F(int texelIdx, const ei::Vec4& value);
	void write_RG32F(int texelIdx, const ei::Vec4& value);
	void write_RGB32F(int texelIdx, const ei::Vec4& value);
	void write_RGBA32F(int texelIdx, const ei::Vec4& value);
	void write_RGB9E5(int texelIdx, const ei::Vec4& value);

	ei::Vec4 (CpuTexture::* m_sample)(const UvCoordinate& uv, int layer) const;
	ei::Vec4 sample_nearest(const UvCoordinate& uv, int layer) const;
	ei::Vec4 sample_linear(const UvCoordinate& uv, int layer) const;
	// Faster methods for 1x1x1 textures
	ei::Vec4 sample111_nearest(const UvCoordinate& uv, int layer) const;
	ei::Vec4 sample111_linear(const UvCoordinate& uv, int layer) const;

	inline int get_index(const ei::IVec3 & texel) const { return texel.x + (texel.y + texel.z * m_size.y) * m_size.x; }
};

}}} // namespace mufflon::scene::textures