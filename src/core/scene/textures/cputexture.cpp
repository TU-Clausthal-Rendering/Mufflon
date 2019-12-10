#include "cputexture.hpp"
#include "util/parallel.hpp"
#include "core/memory/dyntype_memory.hpp"
#include <ei/conversions.hpp>
#include <cuda_fp16.h>

using namespace ei;

namespace mufflon::scene::textures {

CpuTexture::CpuTexture(u16 width, u16 height, u16 numLayers, Format format, SamplingMode mode,
					   MipmapType type, bool sRgb, bool dataHasMipmaps, std::unique_ptr<u8[]> data) :
	m_imageData((data == nullptr) ? nullptr : move(data)),
	m_format(format),
	m_size(width, height, numLayers, type == MipmapType::NONE ? 1u : (1 + ei::ilog2(std::max(width, height))))
{
	// Compute the mipmap offsets
	m_mipmapOffsets = std::make_unique<ei::UVec3[]>(m_size.w);
	u32 currOffset = 0u;
	u32 currWidth = width;
	u32 currHeight = height;
	for(i16 level = 0; level < m_size.w; ++level) {
		m_mipmapOffsets[level] = ei::UVec3{ currWidth, currHeight, currOffset };
		currOffset += static_cast<u32>(currWidth * currHeight * m_size.z);
		currWidth = std::max(1u, currWidth / 2u);
		currHeight = std::max(1u, currHeight / 2u);
	}
	const bool computeMipmaps = (m_imageData == nullptr) || (m_size.w > 1 && !dataHasMipmaps);
	if(m_imageData == nullptr) {
		m_imageData = std::make_unique<u8[]>(currOffset * PIXEL_SIZE(format));
	} else if(m_size.w > 1 && !dataHasMipmaps) {
		// If the old array only had space for the highest mipmap we gotta realloc
		u8* ptr = static_cast<u8*>(std::realloc(m_imageData.release(), currOffset * PIXEL_SIZE(format)));
		if(ptr == nullptr)
			throw std::bad_alloc();
		m_imageData.reset(ptr);
	}

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
		case Format::RGBA8U: m_fetch = sRgb ? &CpuTexture::fetch_sRGBA8U : &CpuTexture::fetch_RGBA8U;
			m_write = sRgb ? &CpuTexture::write_sRGBA8U : &CpuTexture::write_RGBA8U; break;
		case Format::R16U: m_fetch = &CpuTexture::fetch_R16U;
			m_write = &CpuTexture::write_R16U; break;
		case Format::RG16U: m_fetch = &CpuTexture::fetch_RG16U;
			m_write = &CpuTexture::write_RG16U; break;
		case Format::RGBA16U: m_fetch = &CpuTexture::fetch_RGBA16U;
			m_write = &CpuTexture::write_RGBA16U; break;
		case Format::R16F: m_fetch = &CpuTexture::fetch_R16F;
			m_write = &CpuTexture::write_R16F; break;
		case Format::RG16F: m_fetch = &CpuTexture::fetch_RG16F;
			m_write = &CpuTexture::write_RG16F; break;
		case Format::RGBA16F: m_fetch = &CpuTexture::fetch_RGBA16F;
			m_write = &CpuTexture::write_RGBA16F; break;
		case Format::R32F: m_fetch = &CpuTexture::fetch_R32F;
			m_write = &CpuTexture::write_R32F; break;
		case Format::RG32F: m_fetch = &CpuTexture::fetch_RG32F;
			m_write = &CpuTexture::write_RG32F; break;
		case Format::RGBA32F: m_fetch = &CpuTexture::fetch_RGBA32F;
			m_write = &CpuTexture::write_RGBA32F; break;
		default: mAssert(false);
	};

	// Compute the actual mipmap
	if(computeMipmaps)
		recompute_mipmaps(type);
}

void CpuTexture::recompute_mipmaps(MipmapType type) {
	if(type == MipmapType::NONE)
		return;

	for(i32 level = 1; level < m_size.w; ++level) {
		for(i32 layer = 0; layer < m_size.z; ++layer) {
			// Take four texels to form the mipmap texel
			const i32 mipWidth = m_size.x >> level;
			const i32 mipHeight = m_size.y >> level;
			const float stepX = m_size.x / static_cast<float>(mipWidth);
			const float stepY = m_size.y / static_cast<float>(mipHeight);

#pragma PARALLEL_FOR
			for(i32 y = 0u; y < mipHeight; ++y) {
				for(i32 x = 0u; x < mipWidth; ++x) {
					const i32 lowerX = ei::floor(x * stepX);
					const i32 upperX = ei::ceil((x + 1) * stepX);
					const i32 lowerY = ei::floor(y * stepY);
					const i32 upperY = ei::ceil((y + 1) * stepY);
					ei::Vec4 val;
					switch(type) {
						case MipmapType::AVG: {
							val = ei::Vec4{ 0.f };
							for(i32 cy = lowerY; cy <= upperY; ++cy)
								for(i32 cx = lowerX; cx <= upperX; ++cx)
									val += read(Pixel{ cx, cy }, layer, level - 1);
							val /= static_cast<float>((upperX - lowerX + 1) * (upperY - lowerY + 1));
						}	break;
						case MipmapType::MIN: {
							val = ei::Vec4{ std::numeric_limits<float>::max() };
							for(i32 cy = lowerY; cy <= upperY; ++cy)
								for(i32 cx = lowerX; cx <= upperX; ++cx)
									val = ei::min(val, read(Pixel{ cx, cy }, layer, level - 1));
						}	break;
						case MipmapType::MAX: {
							val = ei::Vec4{ -std::numeric_limits<float>::max() };
							for(i32 cy = lowerY; cy <= upperY; ++cy)
								for(i32 cx = lowerX; cx <= upperX; ++cx)
									val = ei::max(val, read(Pixel{ cx, cy }, layer, level - 1));
						}	break;
						default: mAssert(false); break;
					}
					write(val, Pixel{ x << level, y << level }, layer, level);
				}
			}
		}
	}
}

Vec4 CpuTexture::read(const Pixel& texel, int layer, int level) const {
	int idx = get_index(ei::IVec4{ texel, layer, level });
	return (this->*m_fetch)(idx);
}

void CpuTexture::write(const Vec4& value, const Pixel& texel, int layer, int level) {
	int idx = get_index(ei::IVec4{ texel, layer, level });
	(this->*m_write)(idx, value);
}




constexpr float SRGB_TO_RGB[256] = {
	0.0f, 0.000303527f, 0.000607054f, 0.000910581f, 0.0012141079f, 0.0015176349f, 0.0018211619f, 0.0021246889f, 0.0024282159f, 0.0027317429f, 0.0030352698f, 0.0033465358f, 0.0036765073f, 0.004024717f, 0.004391442f, 0.0047769535f,
	0.0051815167f, 0.0056053916f, 0.006048833f, 0.0065120908f, 0.0069954102f, 0.007499032f, 0.008023193f, 0.0085681256f, 0.0091340587f, 0.0097212173f, 0.010329823f, 0.010960094f, 0.0116122452f, 0.0122864884f, 0.0129830323f, 0.013702083f,
	0.0144438436f, 0.0152085144f, 0.0159962934f, 0.0168073758f, 0.0176419545f, 0.0185002201f, 0.019382361f, 0.0202885631f, 0.0212190104f, 0.0221738848f, 0.0231533662f, 0.0241576324f, 0.0251868596f, 0.0262412219f, 0.0273208916f, 0.0284260395f,
	0.0295568344f, 0.0307134437f, 0.0318960331f, 0.0331047666f, 0.0343398068f, 0.0356013149f, 0.0368894504f, 0.0382043716f, 0.0395462353f, 0.0409151969f, 0.0423114106f, 0.0437350293f, 0.0451862044f, 0.0466650863f, 0.0481718242f, 0.049706566f,
	0.0512694584f, 0.052860647f, 0.0544802764f, 0.05612849f, 0.0578054302f, 0.0595112382f, 0.0612460542f, 0.0630100177f, 0.0648032667f, 0.0666259386f, 0.0684781698f, 0.0703600957f, 0.0722718507f, 0.0742135684f, 0.0761853815f, 0.0781874218f,
	0.0802198203f, 0.0822827071f, 0.0843762115f, 0.086500462f, 0.0886555863f, 0.0908417112f, 0.0930589628f, 0.0953074666f, 0.0975873471f, 0.0998987282f, 0.1022417331f, 0.1046164841f, 0.107023103f, 0.1094617108f, 0.1119324278f, 0.1144353738f,
	0.1169706678f, 0.119538428f, 0.1221387722f, 0.1247718176f, 0.1274376804f, 0.1301364767f, 0.1328683216f, 0.1356333297f, 0.138431615f, 0.1412632911f, 0.1441284709f, 0.1470272665f, 0.1499597898f, 0.152926152f, 0.1559264637f, 0.1589608351f,
	0.1620293756f, 0.1651321945f, 0.1682694002f, 0.1714411007f, 0.1746474037f, 0.177888416f, 0.1811642442f, 0.1844749945f, 0.1878207723f, 0.1912016827f, 0.1946178304f, 0.1980693196f, 0.2015562538f, 0.2050787364f, 0.2086368701f, 0.2122307574f,
	0.2158605001f, 0.2195261997f, 0.2232279573f, 0.2269658735f, 0.2307400485f, 0.2345505822f, 0.2383975738f, 0.2422811225f, 0.2462013267f, 0.2501582847f, 0.2541520943f, 0.2581828529f, 0.2622506575f, 0.2663556048f, 0.270497791f, 0.2746773121f,
	0.2788942635f, 0.2831487404f, 0.2874408377f, 0.2917706498f, 0.2961382708f, 0.3005437944f, 0.3049873141f, 0.3094689228f, 0.3139887134f, 0.3185467781f, 0.3231432091f, 0.3277780981f, 0.3324515363f, 0.337163615f, 0.3419144249f, 0.3467040564f,
	0.3515325995f, 0.3564001441f, 0.3613067798f, 0.3662525956f, 0.3712376805f, 0.376262123f, 0.3813260114f, 0.3864294338f, 0.3915724777f, 0.3967552307f, 0.4019777798f, 0.4072402119f, 0.4125426135f, 0.4178850708f, 0.42326767f, 0.4286904966f,
	0.4341536362f, 0.4396571738f, 0.4452011945f, 0.4507857828f, 0.4564110232f, 0.4620769997f, 0.4677837961f, 0.4735314961f, 0.4793201831f, 0.4851499401f, 0.4910208498f, 0.4969329951f, 0.502886458f, 0.5088813209f, 0.5149176654f, 0.5209955732f,
	0.5271151257f, 0.533276404f, 0.539479489f, 0.5457244614f, 0.5520114015f, 0.5583403896f, 0.5647115057f, 0.5711248295f, 0.5775804404f, 0.5840784179f, 0.5906188409f, 0.5972017884f, 0.6038273389f, 0.6104955708f, 0.6172065624f, 0.6239603917f,
	0.6307571363f, 0.637596874f, 0.644479682f, 0.6514056374f, 0.6583748173f, 0.6653872983f, 0.672443157f, 0.6795424696f, 0.6866853124f, 0.6938717613f, 0.7011018919f, 0.7083757799f, 0.7156935005f, 0.7230551289f, 0.7304607401f, 0.7379104088f,
	0.7454042095f, 0.7529422168f, 0.7605245047f, 0.7681511472f, 0.7758222183f, 0.7835377915f, 0.7912979403f, 0.799102738f, 0.8069522577f, 0.8148465722f, 0.8227857544f, 0.8307698768f, 0.8387990117f, 0.8468732315f, 0.8549926081f, 0.8631572135f,
	0.8713671192f, 0.8796223969f, 0.8879231179f, 0.8962693534f, 0.9046611744f, 0.9130986518f, 0.9215818563f, 0.9301108584f, 0.9386857285f, 0.9473065367f, 0.9559733532f, 0.9646862479f, 0.9734452904f, 0.9822505503f, 0.9911020971f, 1.0f
};

Vec4 CpuTexture::fetch_R8U(int texelIdx) const {
	const u8* data = m_imageData.get();
	return {data[texelIdx] / 255.0f, 0.0f, 0.0f, 1.0f};
}

Vec4 CpuTexture::fetch_RG8U(int texelIdx) const {
	const Vec<u8,2>* data = as<Vec<u8,2>>(m_imageData.get());
	return {data[texelIdx] / 255.0f, 0.0f, 1.0f};
}

Vec4 CpuTexture::fetch_RGB8U(int texelIdx) const {
	const Vec<u8,3>* data = as<Vec<u8,3>>(m_imageData.get());
	return {data[texelIdx] / 255.0f, 1.0f};
}

Vec4 CpuTexture::fetch_RGBA8U(int texelIdx) const {
	const Vec<u8,4>* data = as<Vec<u8,4>>(m_imageData.get());
	return data[texelIdx] / 255.0f;
}

Vec4 CpuTexture::fetch_sR8U(int texelIdx) const {
	const u8* data = m_imageData.get();
	return {SRGB_TO_RGB[data[texelIdx]], 0.0f, 0.0f, 1.0f};
}

Vec4 CpuTexture::fetch_sRG8U(int texelIdx) const {
	const Vec<u8,2>* data = as<Vec<u8,2>>(m_imageData.get());
	return {SRGB_TO_RGB[data[texelIdx].r], SRGB_TO_RGB[data[texelIdx].g], 0.0f, 1.0f};
}

Vec4 CpuTexture::fetch_sRGB8U(int texelIdx) const {
	const Vec<u8,3>* data = as<Vec<u8,3>>(m_imageData.get());
	return {SRGB_TO_RGB[data[texelIdx].r], SRGB_TO_RGB[data[texelIdx].g], SRGB_TO_RGB[data[texelIdx].b], 1.0f};
}

Vec4 CpuTexture::fetch_sRGBA8U(int texelIdx) const {
	const Vec<u8,4>* data = as<Vec<u8,4>>(m_imageData.get());
	return {SRGB_TO_RGB[data[texelIdx].r], SRGB_TO_RGB[data[texelIdx].g], SRGB_TO_RGB[data[texelIdx].b], data[texelIdx].a / 255.0f};
}

Vec4 CpuTexture::fetch_R16U(int texelIdx) const {
	const u16* data = as<u16>(m_imageData.get());
	return {data[texelIdx] / 65535.0f, 0.0f, 0.0f, 1.0f};
}

Vec4 CpuTexture::fetch_RG16U(int texelIdx) const {
	const Vec<u16,2>* data = as<Vec<u16,2>>(m_imageData.get());
	return {data[texelIdx] / 65535.0f, 0.0f, 1.0f};
}

Vec4 CpuTexture::fetch_RGB16U(int texelIdx) const {
	const Vec<u16,3>* data = as<Vec<u16,3>>(m_imageData.get());
	return {data[texelIdx] / 65535.0f, 1.0f};
}

Vec4 CpuTexture::fetch_RGBA16U(int texelIdx) const {
	const Vec<u16,4>* data = as<Vec<u16,4>>(m_imageData.get());
	return data[texelIdx] / 65535.0f;
}

Vec4 CpuTexture::fetch_R16F(int texelIdx) const {
	const __half* data = as<__half>(m_imageData.get());
	return { __half2float(data[texelIdx]), 0.0f, 0.0f, 1.0f };
}

Vec4 CpuTexture::fetch_RG16F(int texelIdx) const {
	const Vec<__half, 2>* data = as<Vec<__half, 2>>(m_imageData.get());
	return { __half2float(data[texelIdx].x), __half2float(data[texelIdx].y), 0.0f, 1.0f };
}

Vec4 CpuTexture::fetch_RGB16F(int texelIdx) const {
	const Vec<__half, 3>* data = as<Vec<__half, 3>>(m_imageData.get());
	return { __half2float(data[texelIdx].x), __half2float(data[texelIdx].y),
		__half2float(data[texelIdx].z), 1.0f };
}

Vec4 CpuTexture::fetch_RGBA16F(int texelIdx) const {
	const Vec<__half, 4>* data = as<Vec<__half, 4>>(m_imageData.get());
	return { __half2float(data[texelIdx].x), __half2float(data[texelIdx].y),
		__half2float(data[texelIdx].z), __half2float(data[texelIdx].w) };
}

Vec4 CpuTexture::fetch_R32F(int texelIdx) const {
	return {as<float>(m_imageData.get())[texelIdx], 0.0f, 0.0f, 1.0f};
}

Vec4 CpuTexture::fetch_RG32F(int texelIdx) const {
	return {as<Vec2>(m_imageData.get())[texelIdx], 0.0f, 1.0f};
}

Vec4 CpuTexture::fetch_RGB32F(int texelIdx) const {
	return {as<Vec3>(m_imageData.get())[texelIdx], 1.0f};
}

Vec4 CpuTexture::fetch_RGBA32F(int texelIdx) const {
	return as<Vec4>(m_imageData.get())[texelIdx];
}

Vec4 CpuTexture::fetch_RGB9E5(int texelIdx) const
{
	u32 data = as<u32>(m_imageData.get())[texelIdx];
	return {unpackRGB9E5(data), 1.0f};
}


void CpuTexture::write_R8U(int texelIdx, const Vec4& value) {
	u8* data = m_imageData.get();
	data[texelIdx] = static_cast<u8>(clamp(value.x, 0.0f, 1.0f) * 255.0f);
}

void CpuTexture::write_RG8U(int texelIdx, const Vec4& value) {
	Vec<u8,2>* data = as<Vec<u8,2>>(m_imageData.get());
	data[texelIdx] = Vec<u8,2>{clamp(Vec2{value}, 0.0f, 1.0f) * 255.0f};
}

void CpuTexture::write_RGB8U(int texelIdx, const Vec4& value) {
	Vec<u8,3>* data = as<Vec<u8,3>>(m_imageData.get());
	data[texelIdx] = Vec<u8,3>{clamp(Vec3{value}, 0.0f, 1.0f) * 255.0f};
}

void CpuTexture::write_RGBA8U(int texelIdx, const Vec4& value) {
	Vec<u8,4>* data = as<Vec<u8,4>>(m_imageData.get());
	data[texelIdx] = Vec<u8,4>{clamp(value, 0.0f, 1.0f) * 255.0f};
}

void CpuTexture::write_sR8U(int texelIdx, const Vec4& value) {
	u8* data = m_imageData.get();
	data[texelIdx] = static_cast<u8>(rgbToSRgb(clamp(value.x, 0.0f, 1.0f)) * 255.0f);
}

void CpuTexture::write_sRG8U(int texelIdx, const Vec4& value) {
	Vec<u8,2>* data = as<Vec<u8,2>>(m_imageData.get());
	data[texelIdx] = Vec<u8,2>{rgbToSRgb(clamp(Vec2{value}, 0.0f, 1.0f)) * 255.0f};
}

void CpuTexture::write_sRGB8U(int texelIdx, const Vec4& value) {
	Vec<u8,3>* data = as<Vec<u8,3>>(m_imageData.get());
	data[texelIdx] = Vec<u8,3>{rgbToSRgb(clamp(Vec3{value}, 0.0f, 1.0f)) * 255.0f};
}

void CpuTexture::write_sRGBA8U(int texelIdx, const Vec4& value) {
	Vec<u8,4>* data = as<Vec<u8,4>>(m_imageData.get());
	data[texelIdx] = Vec<u8,4>{rgbToSRgb(clamp(Vec3{value}, 0.0f, 1.0f)) * 255.0f, u8(clamp(value.a, 0.0f, 1.0f) * 255.0f)};
}

void CpuTexture::write_R16U(int texelIdx, const Vec4& value) {
	u16* data = as<u16>(m_imageData.get());
	data[texelIdx] = static_cast<u16>(clamp(value.x, 0.0f, 1.0f) * 65535.0f);
}

void CpuTexture::write_RG16U(int texelIdx, const Vec4& value) {
	Vec<u16,2>* data = as<Vec<u16,2>>(m_imageData.get());
	data[texelIdx] = Vec<u16,2>{clamp(Vec2{value}, 0.0f, 1.0f) * 65535.0f};
}

void CpuTexture::write_RGB16U(int texelIdx, const Vec4& value) {
	Vec<u16,3>* data = as<Vec<u16,3>>(m_imageData.get());
	data[texelIdx] = Vec<u16,3>{clamp(Vec3{value}, 0.0f, 1.0f) * 65535.0f};
}

void CpuTexture::write_RGBA16U(int texelIdx, const Vec4& value) {
	Vec<u16,4>* data = as<Vec<u16,4>>(m_imageData.get());
	data[texelIdx] = Vec<u16,4>{clamp(value, 0.0f, 1.0f) * 65535.0f};
}

void CpuTexture::write_R16F(int texelIdx, const Vec4& value) {
	__half* data = as<__half>(m_imageData.get());
	data[texelIdx] = __float2half(value.x);
}

void CpuTexture::write_RG16F(int texelIdx, const Vec4& value) {
	Vec<__half, 2>* data = as<Vec<__half, 2>>(m_imageData.get());
	data[texelIdx] = { __float2half(value.x), __float2half(value.y) };
}

void CpuTexture::write_RGB16F(int texelIdx, const Vec4& value) {
	Vec<__half, 3>* data = as<Vec<__half, 3>>(m_imageData.get());
	data[texelIdx] = { __float2half(value.x), __float2half(value.y),
						__float2half(value.z) };
}

void CpuTexture::write_RGBA16F(int texelIdx, const Vec4& value) {
	Vec<__half, 4>* data = as<Vec<__half, 4>>(m_imageData.get());
	data[texelIdx] = { __float2half(value.x), __float2half(value.y),
						__float2half(value.z), __float2half(value.w) };
}

void CpuTexture::write_R32F(int texelIdx, const Vec4& value) {
	float* data = as<float>(m_imageData.get());
	data[texelIdx]   = value.x;
}

void CpuTexture::write_RG32F(int texelIdx, const Vec4& value) {
	Vec2* data = as<Vec2>(m_imageData.get());
	data[texelIdx]   = Vec2{value};
}

void CpuTexture::write_RGB32F(int texelIdx, const Vec4& value) {
	Vec3* data = as<Vec3>(m_imageData.get());
	data[texelIdx] = Vec3{value};
}

void CpuTexture::write_RGBA32F(int texelIdx, const Vec4& value) {
	Vec4* data = as<Vec4>(m_imageData.get());
	data[texelIdx] = value;
}

void CpuTexture::write_RGB9E5(int texelIdx, const Vec4& value) {
	as<u32>(m_imageData.get())[texelIdx] = ei::packRGB9E5(Vec3{value});
}


Vec4 CpuTexture::sample_nearest(const UvCoordinate& uv, int layer, float level) const {
	IVec4 baseCoord {ei::floor(uv.x * m_size.x), ei::floor(uv.y * m_size.y), layer,
					 static_cast<u16>(ei::round(level))};
	return (this->*m_fetch)(get_index(baseCoord));
}

Vec4 CpuTexture::sample_linear(const UvCoordinate& uv, int layer, float level) const {
	Vec2 frac {uv.x * m_size.x, uv.y * m_size.y};
	// Higher mipmap
	const float upperMipmapLevel = std::floor(level);
	IVec4 upperCoord {floor(frac), layer, static_cast<u16>(upperMipmapLevel)};
	frac.x -= upperCoord.x; frac.y -= upperCoord.y;
	// Get all 4 texel in the layer and sum them with the interpolation weights (frac)
	Vec4 sample = (this->*m_fetch)(get_index(upperCoord)) * (1.0f - frac.x) * (1.0f - frac.y);
	sample += (this->*m_fetch)(get_index(upperCoord + IVec4(1,0,0,0))) * (frac.x) * (1.0f - frac.y);
	sample += (this->*m_fetch)(get_index(upperCoord + IVec4(0,1,0,0))) * (1.0f - frac.x) * (frac.y);
	sample += (this->*m_fetch)(get_index(upperCoord + IVec4(1,1,0,0))) * (frac.x) * (frac.y);
	// Check if we need to compute another sample
	if(ei::ceil(level) != upperMipmapLevel) {
		upperCoord.w += 1u;
		Vec4 lowerSample = (this->*m_fetch)(get_index(mod(upperCoord, m_size))) * (1.0f - frac.x) * (1.0f - frac.y);
		lowerSample += (this->*m_fetch)(get_index(mod(upperCoord + IVec4(1, 0, 0, 0), m_size))) * (frac.x) * (1.0f - frac.y);
		lowerSample += (this->*m_fetch)(get_index(mod(upperCoord + IVec4(0, 1, 0, 0), m_size))) * (1.0f - frac.x) * (frac.y);
		lowerSample += (this->*m_fetch)(get_index(mod(upperCoord + IVec4(1, 1, 0, 0), m_size))) * (frac.x) * (frac.y);
		sample = ei::lerp(sample, lowerSample, level - upperMipmapLevel);
	}
	// TODO: benchmark if replacing all the mod() calls with a single one and conditional
	// additions for the three new vectors gives some advantage.
	return sample;
}

Vec4 CpuTexture::sample111_nearest(const UvCoordinate& /*uv*/, int /*layer*/, float /*level*/) const {
	return (this->*m_fetch)(0);
}

Vec4 CpuTexture::sample111_linear(const UvCoordinate& /*uv*/, int /*layer*/, float /*level*/) const {
	return (this->*m_fetch)(0);
}

} // namespace mufflon::scene::textures
