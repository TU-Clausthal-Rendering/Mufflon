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
			   MipmapType type, bool sRgb, bool dataHasMipmaps = false, std::unique_ptr<u8[]> data = nullptr);
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
	ei::Vec4 read(const Pixel& texel, int layer = 0, int level = 0) const;

	/*
	 * Overwrite a texel value.
	 * The input is converted to the internal format. If necessary this includes clamping
	 * of the value range.
	 * The border handling mode is always wrap. I.e. modulu operation is performed on the
	 * texture coordinates.
	 */
	void write(const ei::Vec4& value, const Pixel& texel, int layer = 0, int level = 0);

	/*
	 * Get an (interpolated) texture sample at the given coordinate.
	 * The border handling mode is always wrap. I.e. modulu operation is performed on the
	 * texture coordinates.
	 */
	ei::Vec4 sample(const UvCoordinate& uv, int layer = 0, float level = 0.f) const {
		return (this->*m_sample)(uv, layer, level);
	}

	i32 get_width() const noexcept { return m_size.x; }
	i32 get_height() const noexcept { return m_size.y; }
	i32 get_num_layers() const noexcept { return m_size.z; }
	Format get_format() const noexcept { return m_format; }

	// Get the data pointer to the given mipmap level, clamped to [0, MaxMipmapLevel - 1]
	u8* data(int level = 0) noexcept { return m_imageData.get() + PIXEL_SIZE(m_format) * m_mipmapOffsets[std::min(level, m_size.w - 1)].z; }
	const u8* data(int level = 0) const noexcept { return m_imageData.get() + PIXEL_SIZE(m_format) * m_mipmapOffsets[std::min(level, m_size.w - 1)].z; }

	// Computes the mipmaps for the levels given at creation
	void recompute_mipmaps(MipmapType type);

private:
	std::unique_ptr<u8[]> m_imageData;
	std::unique_ptr<ei::UVec3[]> m_mipmapOffsets;
	Format m_format;
	ei::IVec4 m_size;		// width, height, numLayers, mipmapLevels

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
	ei::Vec4 fetch_R16F(int texelIdx) const;
	ei::Vec4 fetch_RG16F(int texelIdx) const;
	ei::Vec4 fetch_RGB16F(int texelIdx) const;
	ei::Vec4 fetch_RGBA16F(int texelIdx) const;
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
	void write_R16F(int texelIdx, const ei::Vec4& value);
	void write_RG16F(int texelIdx, const ei::Vec4& value);
	void write_RGB16F(int texelIdx, const ei::Vec4& value);
	void write_RGBA16F(int texelIdx, const ei::Vec4& value);
	void write_R32F(int texelIdx, const ei::Vec4& value);
	void write_RG32F(int texelIdx, const ei::Vec4& value);
	void write_RGB32F(int texelIdx, const ei::Vec4& value);
	void write_RGBA32F(int texelIdx, const ei::Vec4& value);
	void write_RGB9E5(int texelIdx, const ei::Vec4& value);

	ei::Vec4 (CpuTexture::* m_sample)(const UvCoordinate& uv, int layer, float level) const;
	ei::Vec4 sample_nearest(const UvCoordinate& uv, int layer, float level) const;
	ei::Vec4 sample_linear(const UvCoordinate& uv, int layer, float level) const;
	// Faster methods for 1x1x1 textures
	ei::Vec4 sample111_nearest(const UvCoordinate& uv, int layer, float level) const;
	ei::Vec4 sample111_linear(const UvCoordinate& uv, int layer, float level) const;

	inline int get_index(const ei::IVec4& texel) const {
		// Both layers and mipmap-levels clamp instead of wrap
		const int layer = ei::clamp(texel.z, 0, m_size.z - 1);
		const int level = ei::clamp(texel.w, 0, m_size.w - 1);
		const i32 mipmapWidth = m_mipmapOffsets[level].x;
		const i32 mipmapHeight = m_mipmapOffsets[level].y;
		// We need to clamp for mipmap-levels below 0 since we support textures that are not a power of two,
		// but the original texel coordinates wrap
		const i32 wrappedX = ei::clamp(ei::mod(texel.x, m_size.x) >> level, 0, mipmapWidth - 1);
		const i32 wrappedY = ei::clamp(ei::mod(texel.y, m_size.y) >> level, 0, mipmapHeight - 1);

		return m_mipmapOffsets[level].z + wrappedX + (wrappedY + layer * mipmapHeight) * mipmapWidth;
	}
};

// Returns the texture size (see texture.hpp)
__host__ __forceinline__ Pixel get_texture_size(const textures::ConstTextureDevHandle_t<Device::CPU>& texture) noexcept {
	return { texture->get_width(), texture->get_height() };
}

// Returns the texture layer count (see texture.hpp)
__host__ __forceinline__ u16 get_texture_layers(const textures::ConstTextureDevHandle_t<Device::CPU>& texture) noexcept {
	return texture->get_num_layers();
}

inline u16 get_texture_channel_count(const textures::ConstTextureDevHandle_t<Device::CPU>& texture) noexcept {
	constexpr char CHANNEL_COUNT[int(Format::NUM)] = {
		1, 2, 4, 1, 2, 4, 1, 2, 4, 1, 2, 4
	};
	return CHANNEL_COUNT[int(texture->get_format())];
}

}}} // namespace mufflon::scene::textures