#include "cputexture.hpp"
#include "core/memory/dyntype_memory.hpp"
#include <ei/conversions.hpp>

using namespace ei;

namespace mufflon::scene::textures {

CpuTexture::CpuTexture(u16 width, u16 height, u16 numLayers, Format format, SamplingMode mode, bool sRgb) :
	m_imageData(width * height * PIXEL_SIZE(format)),
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
		case Format::R8U: m_fetch = sRgb ? &CpuTexture::fetch_sR8U : &CpuTexture::fetch_R8U;
						  m_write = sRgb ? &CpuTexture::write_sR8U : &CpuTexture::write_R8U; break;
		case Format::RG8U: m_fetch = sRgb ? &CpuTexture::fetch_sRG8U : &CpuTexture::fetch_RG8U;
						   m_write = sRgb ? &CpuTexture::write_sRG8U : &CpuTexture::write_RG8U; break;
		case Format::RGB8U: m_fetch = sRgb ? &CpuTexture::fetch_sRGB8U : &CpuTexture::fetch_RGB8U;
							m_write = sRgb ? &CpuTexture::write_sRGB8U : &CpuTexture::write_RGB8U; break;
		case Format::RGBA8U: m_fetch = sRgb ? &CpuTexture::fetch_sRGBA8U : &CpuTexture::fetch_RGBA8U;
							 m_write = sRgb ? &CpuTexture::write_sRGBA8U : &CpuTexture::write_RGBA8U; break;
		case Format::R16U: m_fetch = &CpuTexture::fetch_R16U;
						   m_write = &CpuTexture::write_R16U; break;
		case Format::RG16U: m_fetch = &CpuTexture::fetch_RG16U;
							m_write = &CpuTexture::write_RG16U; break;
		case Format::RGB16U: m_fetch = &CpuTexture::fetch_RGB16U;
							 m_write = &CpuTexture::write_RGB16U; break;
		case Format::RGBA16U: m_fetch = &CpuTexture::fetch_RGBA16U;
							  m_write = &CpuTexture::write_RGBA16U; break;
		case Format::R32F: m_fetch = &CpuTexture::fetch_R32F;
						   m_write = &CpuTexture::write_R32F; break;
		case Format::RG32F: m_fetch = &CpuTexture::fetch_RG32F;
							m_write = &CpuTexture::write_RG32F; break;
		case Format::RGB32F: m_fetch = &CpuTexture::fetch_RGB32F;
							 m_write = &CpuTexture::write_RGB32F; break;
		case Format::RGBA32F: m_fetch = &CpuTexture::fetch_RGBA32F;
							  m_write = &CpuTexture::write_RGBA32F; break;
		case Format::RGB9E5: m_fetch = &CpuTexture::fetch_RGB9E5;
							 m_write = &CpuTexture::write_RGB9E5; break;
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




constexpr float SRGB_TO_RGB[256] = {
	0.0f, 0.0498401f, 0.0849447f, 0.110702f, 0.131804f, 0.150005f, 0.166186f, 0.180859f, 0.194353f, 0.206896f, 0.218649f, 0.229735f, 0.240247f, 0.25026f, 0.259833f, 0.269015f,
	0.277847f, 0.286361f, 0.294589f, 0.302554f, 0.310278f, 0.31778f, 0.325076f, 0.332181f, 0.339108f, 0.345869f, 0.352474f, 0.358932f, 0.365252f, 0.371442f, 0.377508f, 0.383458f,
	0.389297f, 0.39503f, 0.400663f, 0.4062f, 0.411645f, 0.417003f, 0.422277f, 0.427471f, 0.432587f, 0.43763f, 0.442601f, 0.447504f, 0.45234f, 0.457113f, 0.461825f, 0.466477f,
	0.471071f, 0.47561f, 0.480096f, 0.484529f, 0.488912f, 0.493246f, 0.497533f, 0.501773f, 0.505969f, 0.510122f, 0.514232f, 0.518301f, 0.522329f, 0.526319f, 0.530271f, 0.534186f,
	0.538065f, 0.541909f, 0.545718f, 0.549494f, 0.553237f, 0.556948f, 0.560628f, 0.564277f, 0.567897f, 0.571487f, 0.575048f, 0.578582f, 0.582088f, 0.585568f, 0.589021f, 0.592449f,
	0.595851f, 0.599229f, 0.602582f, 0.605911f, 0.609218f, 0.612501f, 0.615762f, 0.619001f, 0.622218f, 0.625414f, 0.628589f, 0.631744f, 0.634878f, 0.637993f, 0.641088f, 0.644164f,
	0.647221f, 0.650259f, 0.65328f, 0.656282f, 0.659267f, 0.662235f, 0.665185f, 0.668119f, 0.671036f, 0.673936f, 0.676821f, 0.67969f, 0.682543f, 0.685381f, 0.688203f, 0.691011f,
	0.693804f, 0.696583f, 0.699347f, 0.702097f, 0.704833f, 0.707556f, 0.710264f, 0.71296f, 0.715642f, 0.718312f, 0.720968f, 0.723612f, 0.726243f, 0.728862f, 0.731469f, 0.734064f,
	0.736647f, 0.739218f, 0.741778f, 0.744326f, 0.746862f, 0.749388f, 0.751902f, 0.754406f, 0.756899f, 0.759381f, 0.761853f, 0.764314f, 0.766765f, 0.769205f, 0.771636f, 0.774056f,
	0.776467f, 0.778868f, 0.78126f, 0.783641f, 0.786014f, 0.788377f, 0.790731f, 0.793075f, 0.795411f, 0.797738f, 0.800056f, 0.802365f, 0.804665f, 0.806957f, 0.80924f, 0.811515f,
	0.813782f, 0.81604f, 0.81829f, 0.820532f, 0.822766f, 0.824993f, 0.827211f, 0.829421f, 0.831624f, 0.833819f, 0.836007f, 0.838187f, 0.84036f, 0.842525f, 0.844683f, 0.846834f,
	0.848978f, 0.851114f, 0.853244f, 0.855366f, 0.857482f, 0.859591f, 0.861693f, 0.863788f, 0.865877f, 0.867959f, 0.870034f, 0.872103f, 0.874166f, 0.876222f, 0.878272f, 0.880315f,
	0.882352f, 0.884383f, 0.886408f, 0.888427f, 0.89044f, 0.892447f, 0.894448f, 0.896443f, 0.898432f, 0.900416f, 0.902393f, 0.904365f, 0.906332f, 0.908292f, 0.910248f, 0.912197f,
	0.914141f, 0.91608f, 0.918013f, 0.919941f, 0.921864f, 0.923781f, 0.925693f, 0.9276f, 0.929502f, 0.931398f, 0.93329f, 0.935176f, 0.937057f, 0.938934f, 0.940805f, 0.942672f,
	0.944534f, 0.94639f, 0.948242f, 0.95009f, 0.951932f, 0.95377f, 0.955603f, 0.957432f, 0.959256f, 0.961075f, 0.962889f, 0.9647f, 0.966505f, 0.968307f, 0.970104f, 0.971896f,
	0.973684f, 0.975468f, 0.977247f, 0.979022f, 0.980793f, 0.98256f, 0.984322f, 0.986081f, 0.987835f, 0.989585f, 0.991331f, 0.993073f, 0.994811f, 0.996544f, 0.998274f, 1.0f
};

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

Vec4 CpuTexture::fetch_sR8U(int texelIdx) const {
	const u8* data = m_imageData.data();
	return {SRGB_TO_RGB[data[texelIdx]], 0.0f, 0.0f, 1.0f};
}

Vec4 CpuTexture::fetch_sRG8U(int texelIdx) const {
	const Vec<u8,2>* data = as<Vec<u8,2>>(m_imageData.data());
	return {SRGB_TO_RGB[data[texelIdx].r], SRGB_TO_RGB[data[texelIdx].g], 0.0f, 1.0f};
}

Vec4 CpuTexture::fetch_sRGB8U(int texelIdx) const {
	const Vec<u8,3>* data = as<Vec<u8,3>>(m_imageData.data());
	return {SRGB_TO_RGB[data[texelIdx].r], SRGB_TO_RGB[data[texelIdx].g], SRGB_TO_RGB[data[texelIdx].b], 1.0f};
}

Vec4 CpuTexture::fetch_sRGBA8U(int texelIdx) const {
	const Vec<u8,4>* data = as<Vec<u8,4>>(m_imageData.data());
	return {SRGB_TO_RGB[data[texelIdx].r], SRGB_TO_RGB[data[texelIdx].g], SRGB_TO_RGB[data[texelIdx].b], data[texelIdx].a / 255.0f};
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
	data[texelIdx] = Vec<u8,2>{clamp(Vec2{value}, 0.0f, 1.0f) * 255.0f};
}

void CpuTexture::write_RGB8U(int texelIdx, const Vec4& value) {
	Vec<u8,3>* data = as<Vec<u8,3>>(m_imageData.data());
	data[texelIdx] = Vec<u8,3>{clamp(Vec3{value}, 0.0f, 1.0f) * 255.0f};
}

void CpuTexture::write_RGBA8U(int texelIdx, const Vec4& value) {
	Vec<u8,4>* data = as<Vec<u8,4>>(m_imageData.data());
	data[texelIdx] = Vec<u8,4>{clamp(value, 0.0f, 1.0f) * 255.0f};
}

void CpuTexture::write_sR8U(int texelIdx, const Vec4& value) {
	u8* data = m_imageData.data();
	data[texelIdx] = static_cast<u8>(rgbToSRgb(clamp(value.x, 0.0f, 1.0f)) * 255.0f);
}

void CpuTexture::write_sRG8U(int texelIdx, const Vec4& value) {
	Vec<u8,2>* data = as<Vec<u8,2>>(m_imageData.data());
	data[texelIdx] = Vec<u8,2>{rgbToSRgb(clamp(Vec2{value}, 0.0f, 1.0f)) * 255.0f};
}

void CpuTexture::write_sRGB8U(int texelIdx, const Vec4& value) {
	Vec<u8,3>* data = as<Vec<u8,3>>(m_imageData.data());
	data[texelIdx] = Vec<u8,3>{rgbToSRgb(clamp(Vec3{value}, 0.0f, 1.0f)) * 255.0f};
}

void CpuTexture::write_sRGBA8U(int texelIdx, const Vec4& value) {
	Vec<u8,4>* data = as<Vec<u8,4>>(m_imageData.data());
	data[texelIdx] = Vec<u8,4>{rgbToSRgb(clamp(Vec3{value}, 0.0f, 1.0f)) * 255.0f, u8(clamp(value.a, 0.0f, 1.0f) * 255.0f)};
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
