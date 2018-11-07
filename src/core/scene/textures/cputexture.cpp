#include "cputexture.hpp"
#include "core/memory/dyntype_memory.hpp"

namespace mufflon::scene::textures {

static constexpr u8 NUM_COMPONENTS[int(Format::NUM)] = {
	1, 2, 3, 4, // ...8U formats
	1, 2, 3, 4, // ...16U formats
	//1, 2, 3, 4, // ...32U formats
	1, 2, 3, 4, // ...32F formats
	1			// RGB9E5
};

static constexpr u8 PIXEL_SIZE[int(Format::NUM)] = {
	1, 2, 3, 4, // ...8U formats
	2, 4, 6, 8, // ...16U formats
	//4, 8, 12, 16, // ...32U formats
	4, 8, 12, 16, // ...32F formats
	4			// RGB9E5
};

CpuTexture::CpuTexture(u16 width, u16 height, u16 numLayers, Format format, SamplingMode mode) :
	m_imageData(width * height * PIXEL_SIZE[int(format)]),
	m_format(format),
	m_size(width, height, numLayers),
	m_numComponents(NUM_COMPONENTS[int(format)])
{
	// Choose an optimized sampling routine
	if(width * height * numLayers == 1)
		m_sample = mode == SamplingMode::NEAREST ? &CpuTexture::sample111_nearest : &CpuTexture::sample111_linear;
	else
		m_sample = mode == SamplingMode::NEAREST ? &CpuTexture::sample_nearest : &CpuTexture::sample_linear;

	// Select the proper access function for the format
	switch(format) {
		case Format::R8U: m_fetch = &CpuTexture::fetch_R8U; break;
		case Format::RG8U: m_fetch = &CpuTexture::fetch_RG8U; break;
		case Format::RGB8U: m_fetch = &CpuTexture::fetch_RGB8U; break;
		case Format::RGBA8U: m_fetch = &CpuTexture::fetch_RGBA8U; break;
		case Format::R16U: m_fetch = &CpuTexture::fetch_R16U; break;
		case Format::RG16U: m_fetch = &CpuTexture::fetch_RG16U; break;
		case Format::RGB16U: m_fetch = &CpuTexture::fetch_RGB16U; break;
		case Format::RGBA16U: m_fetch = &CpuTexture::fetch_RGBA16U; break;
		case Format::R32F: m_fetch = &CpuTexture::fetch_R32F; break;
		case Format::RG32F: m_fetch = &CpuTexture::fetch_RG32F; break;
		case Format::RGB32F: m_fetch = &CpuTexture::fetch_RGB32F; break;
		case Format::RGBA32F: m_fetch = &CpuTexture::fetch_RGBA32F; break;
		case Format::RGB9E5: m_fetch = &CpuTexture::fetch_RGB9E5; break;
	};
}



ei::Vec4 CpuTexture::read(const Pixel& texel, int layer) const {
	ei::IVec3 wrappedPixel = mod(ei::IVec3{texel, layer}, m_size);
	int idx = get_index(wrappedPixel);
	return (this->*m_fetch)(idx);
}




ei::Vec4 CpuTexture::fetch_R8U(int componentIdx) const {
	const u8* data = m_imageData.data();
	return {data[componentIdx] / 255.0f, 0.0f, 0.0f, 1.0f};
}

ei::Vec4 CpuTexture::fetch_RG8U(int componentIdx) const {
	const u8* data = m_imageData.data();
	return {data[componentIdx] / 255.0f, data[componentIdx+1] / 255.0f, 0.0f, 1.0f};
}

ei::Vec4 CpuTexture::fetch_RGB8U(int componentIdx) const {
	const u8* data = m_imageData.data();
	return {data[componentIdx] / 255.0f, data[componentIdx+1] / 255.0f, data[componentIdx+2] / 255.0f, 1.0f};
}

ei::Vec4 CpuTexture::fetch_RGBA8U(int componentIdx) const {
	const u8* data = m_imageData.data();
	return ei::Vec4{data[componentIdx], data[componentIdx+1], data[componentIdx+2], data[componentIdx+3]} / 255.0f;
}

ei::Vec4 CpuTexture::fetch_R16U(int componentIdx) const {
	const u16* data = as<u16>(m_imageData.data());
	return {data[componentIdx] / 65535.0f, 0.0f, 0.0f, 1.0f};
}

ei::Vec4 CpuTexture::fetch_RG16U(int componentIdx) const {
	const u16* data = as<u16>(m_imageData.data());
	return {data[componentIdx] / 65535.0f, data[componentIdx+1] / 65535.0f, 0.0f, 1.0f};
}

ei::Vec4 CpuTexture::fetch_RGB16U(int componentIdx) const {
	const u16* data = as<u16>(m_imageData.data());
	return {data[componentIdx] / 65535.0f, data[componentIdx+1] / 65535.0f, data[componentIdx+2] / 65535.0f, 1.0f};
}

ei::Vec4 CpuTexture::fetch_RGBA16U(int componentIdx) const {
	const u16* data = as<u16>(m_imageData.data());
	return ei::Vec4{data[componentIdx], data[componentIdx+1], data[componentIdx+2], data[componentIdx+3]} / 65535.0f;
}

ei::Vec4 CpuTexture::fetch_R32F(int componentIdx) const {
	const float* data = as<float>(m_imageData.data());
	return {data[componentIdx], data[componentIdx], data[componentIdx], 1.0f};
}

ei::Vec4 CpuTexture::fetch_RG32F(int componentIdx) const {
	const float* data = as<float>(m_imageData.data());
	return {data[componentIdx], data[componentIdx+1], 0.0f, 1.0f};
}

ei::Vec4 CpuTexture::fetch_RGB32F(int componentIdx) const {
	const float* data = as<float>(m_imageData.data());
	return {data[componentIdx], data[componentIdx+1], data[componentIdx+2], 1.0f};
}

ei::Vec4 CpuTexture::fetch_RGBA32F(int componentIdx) const {
	const float* data = as<float>(m_imageData.data());
	return {data[componentIdx], data[componentIdx+1], data[componentIdx+2], data[componentIdx+3]};
}

ei::Vec4 CpuTexture::fetch_RGB9E5(int componentIdx) const
{
	u32 data = as<u32>(m_imageData.data())[componentIdx];
	float e = pow(2.0f, (data>>27) - 15.0f - 9.0f);
	return {(data & 0x1ff) * e, ((data>>9) & 0x1ff) * e, ((data>>18) & 0x1ff) * e, 1.0f};
}


ei::Vec4 CpuTexture::sample_nearest(const UvCoordinate& uv, int layer) const {
	ei::IVec3 baseCoord {ei::floor(uv.x * m_size.x), ei::floor(uv.y * m_size.y), layer};
	return (this->*m_fetch)(get_index(mod(baseCoord, m_size)));
}

ei::Vec4 CpuTexture::sample_linear(const UvCoordinate& uv, int layer) const {
	ei::Vec2 frac {uv.x * m_size.x, uv.y * m_size.y};
	ei::IVec3 baseCoord {ei::floor(frac.x), ei::floor(frac.y), layer};
	frac.x -= baseCoord.x; frac.y -= baseCoord.y;
	// Get all 4 texel in the layer and sum them with the interpolation weights (frac)
	ei::Vec4 sample = (this->*m_fetch)(get_index(mod(baseCoord, m_size))) * (1.0f - frac.x) * (1.0f - frac.y);
	sample += (this->*m_fetch)(get_index(mod(baseCoord + ei::IVec3(1,0,0), m_size))) * (frac.x) * (1.0f - frac.y);
	sample += (this->*m_fetch)(get_index(mod(baseCoord + ei::IVec3(0,1,0), m_size))) * (1.0f - frac.x) * (frac.y);
	sample += (this->*m_fetch)(get_index(mod(baseCoord + ei::IVec3(1,1,0), m_size))) * (frac.x) * (frac.y);
	// TODO: benchmark if replacing all the mod() calls with a single one and conditional
	// additions for the three new vectors gives some advantage.
	return sample;
}

ei::Vec4 CpuTexture::sample111_nearest(const UvCoordinate& uv, int layer) const {
	return (this->*m_fetch)(0);
}

ei::Vec4 CpuTexture::sample111_linear(const UvCoordinate& uv, int layer) const {
	return (this->*m_fetch)(0);
}

} // namespace mufflon::scene::textures
