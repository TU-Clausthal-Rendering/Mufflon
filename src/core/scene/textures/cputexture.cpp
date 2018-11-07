#include "cputexture.hpp"
#include "core/memory/dyntype_memory.hpp"
#include <ei/conversions.hpp>

using namespace ei;

namespace mufflon::scene::textures {

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
	m_size(width, height, numLayers)
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



Vec4 CpuTexture::read(const Pixel& texel, int layer) const {
	IVec3 wrappedPixel = mod(IVec3{texel, layer}, m_size);
	int idx = get_index(wrappedPixel);
	return (this->*m_fetch)(idx);
}

void CpuTexture::write(const Vec4& value, const Pixel& texel, int layer) {
	IVec3 wrappedPixel = mod(IVec3{texel, layer}, m_size);
	int idx = get_index(wrappedPixel);
	(this->*m_write)(idx, value);
}




Vec4 CpuTexture::fetch_R8U(int texelIdx) const {
	const u8* data = m_imageData.data();
	return {data[texelIdx] / 255.0f, 0.0f, 0.0f, 1.0f};
}

Vec4 CpuTexture::fetch_RG8U(int texelIdx) const {
	const Vec<u8,2>* data = as<Vec<u8,2>>(m_imageData.data());
	return {data[texelIdx] / 255.0f, 0.0f, 1.0f};
}

Vec4 CpuTexture::fetch_RGB8U(int texelIdx) const {
	const Vec<u8,3>* data = as<Vec<u8,3>>(m_imageData.data());
	return {data[texelIdx] / 255.0f, 1.0f};
}

Vec4 CpuTexture::fetch_RGBA8U(int texelIdx) const {
	const Vec<u8,4>* data = as<Vec<u8,4>>(m_imageData.data());
	return data[texelIdx] / 255.0f;
}

Vec4 CpuTexture::fetch_R16U(int texelIdx) const {
	const u16* data = as<u16>(m_imageData.data());
	return {data[texelIdx] / 65535.0f, 0.0f, 0.0f, 1.0f};
}

Vec4 CpuTexture::fetch_RG16U(int texelIdx) const {
	const Vec<u16,2>* data = as<Vec<u16,2>>(m_imageData.data());
	return {data[texelIdx] / 65535.0f, 0.0f, 1.0f};
}

Vec4 CpuTexture::fetch_RGB16U(int texelIdx) const {
	const Vec<u16,3>* data = as<Vec<u16,3>>(m_imageData.data());
	return {data[texelIdx] / 65535.0f, 1.0f};
}

Vec4 CpuTexture::fetch_RGBA16U(int texelIdx) const {
	const Vec<u16,4>* data = as<Vec<u16,4>>(m_imageData.data());
	return data[texelIdx] / 65535.0f;
}

Vec4 CpuTexture::fetch_R32F(int texelIdx) const {
	return {as<float>(m_imageData.data())[texelIdx], 0.0f, 0.0f, 1.0f};
}

Vec4 CpuTexture::fetch_RG32F(int texelIdx) const {
	return {as<Vec2>(m_imageData.data()[texelIdx]), 0.0f, 1.0f};
}

Vec4 CpuTexture::fetch_RGB32F(int texelIdx) const {
	return {as<Vec3>(m_imageData.data()[texelIdx]), 1.0f};
}

Vec4 CpuTexture::fetch_RGBA32F(int texelIdx) const {
	return as<Vec4>(m_imageData.data()[texelIdx]);
}

Vec4 CpuTexture::fetch_RGB9E5(int texelIdx) const
{
	u32 data = as<u32>(m_imageData.data())[texelIdx];
	return {unpackRGB9E5(data), 1.0f};
}


void CpuTexture::write_R8U(int texelIdx, const Vec4& value) {
	u8* data = m_imageData.data();
	data[texelIdx] = static_cast<u8>(clamp(value.x, 0.0f, 1.0f) * 255.0f);
}

void CpuTexture::write_RG8U(int texelIdx, const Vec4& value) {
	Vec<u8,2>* data = as<Vec<u8,2>>(m_imageData.data());
	data[texelIdx] = Vec<u8,2>{clamp(Vec2{value.x}, 0.0f, 1.0f) * 255.0f};
}

void CpuTexture::write_RGB8U(int texelIdx, const Vec4& value) {
	Vec<u8,3>* data = as<Vec<u8,3>>(m_imageData.data());
	data[texelIdx] = Vec<u8,3>{clamp(Vec3{value.x}, 0.0f, 1.0f) * 255.0f};
}

void CpuTexture::write_RGBA8U(int texelIdx, const Vec4& value) {
	Vec<u8,4>* data = as<Vec<u8,4>>(m_imageData.data());
	data[texelIdx] = Vec<u8,4>{clamp(value.x, 0.0f, 1.0f) * 255.0f};
}

void CpuTexture::write_R16U(int texelIdx, const Vec4& value) {
	u16* data = as<u16>(m_imageData.data());
	data[texelIdx] = static_cast<u16>(clamp(value.x, 0.0f, 1.0f) * 65535.0f);
}

void CpuTexture::write_RG16U(int texelIdx, const Vec4& value) {
	Vec<u16,2>* data = as<Vec<u16,2>>(m_imageData.data());
	data[texelIdx] = Vec<u16,2>{clamp(Vec2{value}, 0.0f, 1.0f) * 65535.0f};
}

void CpuTexture::write_RGB16U(int texelIdx, const Vec4& value) {
	Vec<u16,3>* data = as<Vec<u16,3>>(m_imageData.data());
	data[texelIdx] = Vec<u16,3>{clamp(Vec3{value}, 0.0f, 1.0f) * 65535.0f};
}

void CpuTexture::write_RGBA16U(int texelIdx, const Vec4& value) {
	Vec<u16,4>* data = as<Vec<u16,4>>(m_imageData.data());
	data[texelIdx] = Vec<u16,4>{clamp(value, 0.0f, 1.0f) * 65535.0f};
}

void CpuTexture::write_R32F(int texelIdx, const Vec4& value) {
	float* data = as<float>(m_imageData.data());
	data[texelIdx]   = value.x;
}

void CpuTexture::write_RG32F(int texelIdx, const Vec4& value) {
	Vec2* data = as<Vec2>(m_imageData.data());
	data[texelIdx]   = Vec2{value};
}

void CpuTexture::write_RGB32F(int texelIdx, const Vec4& value) {
	Vec3* data = as<Vec3>(m_imageData.data());
	data[texelIdx] = Vec3{value};
}

void CpuTexture::write_RGBA32F(int texelIdx, const Vec4& value) {
	Vec4* data = as<Vec4>(m_imageData.data());
	data[texelIdx] = value;
}

void CpuTexture::write_RGB9E5(int texelIdx, const Vec4& value) {
	as<u32>(m_imageData.data())[texelIdx] = ei::packRGB9E5(Vec3{value});
}


Vec4 CpuTexture::sample_nearest(const UvCoordinate& uv, int layer) const {
	IVec3 baseCoord {ei::floor(uv.x * m_size.x), ei::floor(uv.y * m_size.y), layer};
	return (this->*m_fetch)(get_index(mod(baseCoord, m_size)));
}

Vec4 CpuTexture::sample_linear(const UvCoordinate& uv, int layer) const {
	Vec2 frac {uv.x * m_size.x, uv.y * m_size.y};
	IVec3 baseCoord {floor(frac), layer};
	frac.x -= baseCoord.x; frac.y -= baseCoord.y;
	// Get all 4 texel in the layer and sum them with the interpolation weights (frac)
	Vec4 sample = (this->*m_fetch)(get_index(mod(baseCoord, m_size))) * (1.0f - frac.x) * (1.0f - frac.y);
	sample += (this->*m_fetch)(get_index(mod(baseCoord + IVec3(1,0,0), m_size))) * (frac.x) * (1.0f - frac.y);
	sample += (this->*m_fetch)(get_index(mod(baseCoord + IVec3(0,1,0), m_size))) * (1.0f - frac.x) * (frac.y);
	sample += (this->*m_fetch)(get_index(mod(baseCoord + IVec3(1,1,0), m_size))) * (frac.x) * (frac.y);
	// TODO: benchmark if replacing all the mod() calls with a single one and conditional
	// additions for the three new vectors gives some advantage.
	return sample;
}

Vec4 CpuTexture::sample111_nearest(const UvCoordinate& uv, int layer) const {
	return (this->*m_fetch)(0);
}

Vec4 CpuTexture::sample111_linear(const UvCoordinate& uv, int layer) const {
	return (this->*m_fetch)(0);
}

} // namespace mufflon::scene::textures
