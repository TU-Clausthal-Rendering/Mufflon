#include "texture.hpp"
#include "cputexture.hpp"
#include "util/log.hpp"
#include "core/cuda/error.hpp"
#include "core/memory/synchronize.hpp"
#include <algorithm>
#include <vector>

namespace mufflon::scene::textures {

Texture::Texture(std::string name, u16 width, u16 height, u16 numLayers, MipmapType mipmapType, Format format,
				 SamplingMode mode, bool sRgb, bool dataHasMipmaps, std::unique_ptr<u8[]> data) :
	m_width(width),
	m_height(height),
	m_numLayers(std::max<u16>(1u, numLayers)),
	m_mipmapType(mipmapType),
	m_mipmapLevels(mipmapType == MipmapType::NONE ? 1u : (1 + ei::ilog2(std::max(width, height)))),
	m_format(format),
	m_mode(mode),
	m_sRgb(sRgb),
	m_cudaTexture(nullptr),
	m_name(move(name))
{
	if(data) {
		// A file loader provides an array with pixel data. This is loaded into
		// a CPUTexture per default.
		create_texture_cpu(dataHasMipmaps, move(data));
	}
}

Texture::Texture(Texture&& tex) :
	m_width(tex.m_width),
	m_height(tex.m_height),
	m_numLayers(tex.m_numLayers),
	m_format(tex.m_format),
	m_mode(tex.m_mode),
	m_sRgb(tex.m_sRgb),
	m_cpuTexture(std::move(tex.m_cpuTexture)),
	m_cudaTexture(tex.m_cudaTexture),
	m_handles(tex.m_handles),
	m_constHandles(tex.m_constHandles),
	m_mipmapLevels(tex.m_mipmapLevels),
	m_name(std::move(tex.m_name))
{
	m_cudaTexture = nullptr;
}

Texture::~Texture() {
	m_cpuTexture = nullptr;
	if(m_cudaTexture) {
		cuda::check_error(cudaDestroyTextureObject(m_constHandles.get<ConstTextureDevHandle_t<Device::CUDA>>().handle));
		cuda::check_error(cudaDestroySurfaceObject(m_handles.get<TextureDevHandle_t<Device::CUDA>>().handle));
		cuda::check_error(cudaFreeMipmappedArray(m_cudaTexture));
		m_cudaTexture = nullptr;
		m_cudaTexture = nullptr;
	}
}

template < Device dev >
void Texture::synchronize() {
		// Create texture resources if necessary
	if(!is_valid(m_constHandles.get<ConstTextureDevHandle_t<dev>>())) {
		switch(dev) {
			case Device::CPU: create_texture_cpu(); break;
			case Device::CUDA: create_texture_cuda(); break;
			//case Device::OPENGL: create_texture_opengl(); break;
			default: mAssert(false);
		}

		// Check from which device to copy
		if(dev != Device::CPU && is_valid(m_constHandles.get<ConstTextureDevHandle_t<Device::CPU>>())) {
			if(dev == Device::CUDA) {
				u32 width = m_width;
				u32 height = m_height;
				// Copy over all mipmap levels!
				for(u32 level = 0u; level < m_mipmapLevels; ++level) {
					cudaMemcpy3DParms copyParams{ 0u };
					copyParams.srcPtr = make_cudaPitchedPtr(m_cpuTexture->data(level), width * PIXEL_SIZE(m_format), width, height);
					cuda::check_error(cudaGetMipmappedArrayLevel(&copyParams.dstArray, m_cudaTexture, level));
					copyParams.extent = make_cudaExtent(width, height, m_numLayers);
					copyParams.kind = cudaMemcpyDefault;
					cuda::check_error(cudaMemcpy3D(&copyParams));
					width = std::max(1u, width / 2u);
					height = std::max(1u, height / 2u);
				}
			} else {
				// TODO: OpenGL
			}
		} else if(dev != Device::CUDA && is_valid(m_constHandles.get<ConstTextureDevHandle_t<Device::CUDA>>())) {
			if(dev == Device::CPU) {
				u32 width = m_width;
				u32 height = m_height;
				// Copy over all mipmap levels!
				for(u32 level = 0u; level < m_mipmapLevels; ++level) {
					cudaMemcpy3DParms copyParams{ 0u };
					copyParams.dstPtr = make_cudaPitchedPtr(m_cpuTexture->data(level), width * PIXEL_SIZE(m_format), width, height);
					cuda::check_error(cudaGetMipmappedArrayLevel(&copyParams.srcArray, m_cudaTexture, level));
					copyParams.extent = make_cudaExtent(width, height, m_numLayers);
					copyParams.kind = cudaMemcpyDefault;
					cuda::check_error(cudaMemcpy3D(&copyParams));
					width = std::max(1u, width / 2u);
					height = std::max(1u, height / 2u);
				}
			} else {
				// TODO: OpenGL
			}
		} else if(dev != Device::OPENGL && is_valid(m_constHandles.get<ConstTextureDevHandle_t<Device::OPENGL>>())) {
			if(dev == Device::CPU) {
				// TODO
			} else {
				// TODO
			}
		}
	}
}
template void Texture::synchronize<Device::CPU>();
template void Texture::synchronize<Device::CUDA>();
template void Texture::synchronize<Device::OPENGL>();



template < Device dev >
void Texture::unload() {
	switch(dev) {
		case Device::CPU: {
			if(m_cpuTexture) {
				m_handles.get<TextureDevHandle_t<Device::CPU>>() = nullptr;
				m_constHandles.get<ConstTextureDevHandle_t<Device::CPU>>() = nullptr;
				m_cpuTexture = nullptr;
			}
		} break;
		case Device::CUDA: {
			if(m_cudaTexture) {
				// Free
				auto& texHdl = m_constHandles.get<ConstTextureDevHandle_t<Device::CUDA>>();
				auto& surfHdl = m_handles.get<TextureDevHandle_t<Device::CUDA>>();
				cuda::check_error(cudaDestroyTextureObject(texHdl.handle));
				cuda::check_error(cudaDestroySurfaceObject(surfHdl.handle));
				cuda::check_error(cudaFreeMipmappedArray(m_cudaTexture));
				// Reset handles
				texHdl.handle = 0;
				surfHdl.handle = 0;
				m_cudaTexture = nullptr;
			}
		} break;
		case Device::OPENGL: {
			// TODO
		} break;
	}
}

template void Texture::unload<Device::CPU>();
template void Texture::unload<Device::CUDA>();
template void Texture::unload<Device::OPENGL>();



template < Device dev >
void Texture::clear() {
	// There is no clear-call in cuda. The only possibility is to have a
	// memory with zeros and to copy it. Therefore here is a static vector
	// to avoid allocations on each clear call.
	static std::vector<u8> s_zeroMem;
	size_t texMemSize = size_t(m_width) * size_t(m_height) * size_t(m_numLayers) * PIXEL_SIZE(m_format);
	switch(dev) {
		case Device::CPU: {
			if(!m_cpuTexture) {
				logError("[Texture::zero] Trying to clear a CPU texture without memory.");
			} else {
				u32 width = m_width;
				u32 height = m_height;
				for(u32 level = 0u; level < m_mipmapLevels; ++level) {
					memset(m_cpuTexture->data(level), 0, size_t(width) * size_t(height) * size_t(m_numLayers) * PIXEL_SIZE(m_format));
					width = std::max(1u, width / 2u);
					height = std::max(1u, height / 2u);
				}
			}
		} break;
		case Device::CUDA: {
			if(!m_cudaTexture)
				logError("[Texture::zero] Trying to clear a CUDA texture without memory.");
			else {
				// Is the dummy memory to small?
				if(s_zeroMem.size() < texMemSize) {
					s_zeroMem.resize(texMemSize);
					memset(s_zeroMem.data(), 0, texMemSize);
				}

				u32 width = m_width;
				u32 height = m_height;
				for(u32 level = 0u; level < m_mipmapLevels; ++level) {
					cudaMemcpy3DParms copyParams{ 0u };
					copyParams.srcPtr = make_cudaPitchedPtr(s_zeroMem.data(), width * PIXEL_SIZE(m_format), width, height);
					cuda::check_error(cudaGetMipmappedArrayLevel(&copyParams.dstArray, m_cudaTexture, level));
					copyParams.extent = make_cudaExtent(width, height, m_numLayers);
					copyParams.kind = cudaMemcpyDefault;
					cuda::check_error(cudaMemcpy3D(&copyParams));
					width = std::max(1u, width / 2u);
					height = std::max(1u, height / 2u);
				}
			}
		} break;
		case Device::OPENGL: {
			// TODO
		} break;
	}
}

template void Texture::clear<Device::CPU>();
template void Texture::clear<Device::CUDA>();
template void Texture::clear<Device::OPENGL>();




void Texture::create_texture_cpu(bool dataHasMipmaps, std::unique_ptr<u8[]> data) {
	mAssert(m_cpuTexture == nullptr);
	m_cpuTexture = std::make_unique<CpuTexture>(m_width, m_height, m_numLayers, m_format, m_mode,
												m_mipmapType, m_sRgb, data != nullptr ? dataHasMipmaps : false, move(data));
	m_handles.get<TextureDevHandle_t<Device::CPU>>() = m_cpuTexture.get();
	m_constHandles.get<ConstTextureDevHandle_t<Device::CPU>>() = m_cpuTexture.get();
}

void Texture::create_texture_cuda() {
	mAssert(m_cudaTexture == nullptr);
	cudaChannelFormatDesc channelDesc;
	switch(m_format) {
		case Format::R8U: channelDesc = cudaCreateChannelDesc(8, 0, 0, 0, cudaChannelFormatKindUnsigned); break;
		case Format::RG8U: channelDesc = cudaCreateChannelDesc(8, 8, 0, 0, cudaChannelFormatKindUnsigned); break;
		case Format::RGBA8U: channelDesc = cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsigned); break;
		case Format::R16U: channelDesc = cudaCreateChannelDesc(16, 0, 0, 0, cudaChannelFormatKindUnsigned); break;
		case Format::RG16U: channelDesc = cudaCreateChannelDesc(16, 16, 0, 0, cudaChannelFormatKindUnsigned); break;
		case Format::RGBA16U: channelDesc = cudaCreateChannelDesc(16, 16, 16, 16, cudaChannelFormatKindUnsigned); break;
			// TODO: needs driver API
		case Format::R16F: channelDesc = cudaCreateChannelDesc(16, 0, 0, 0, cudaChannelFormatKindFloat); break;
		case Format::RG16F: channelDesc = cudaCreateChannelDesc(16, 16, 0, 0, cudaChannelFormatKindFloat); break;
		case Format::RGBA16F: channelDesc = cudaCreateChannelDesc(16, 16, 16, 16, cudaChannelFormatKindFloat); break;
		case Format::R32F: channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat); break;
		case Format::RG32F: channelDesc = cudaCreateChannelDesc(32, 32, 0, 0, cudaChannelFormatKindFloat); break;
		case Format::RGBA32F: channelDesc = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat); break;
		default:
			mAssertMsg(false, "Format not implemented.");
	}
	cuda::check_error(cudaGetLastError());

	// Allocate the CUDA texture array
	cuda::check_error(cudaMallocMipmappedArray(&m_cudaTexture, &channelDesc, make_cudaExtent(m_width, m_height, m_numLayers),
											   m_mipmapLevels, cudaArrayLayered));

	// Specify the texture view on the memory
	cudaResourceDesc resDesc{};
	resDesc.resType = cudaResourceTypeMipmappedArray;
	resDesc.res.mipmap.mipmap = m_cudaTexture;
	cudaTextureDesc texDesc{};
	texDesc.addressMode[0] = texDesc.addressMode[1] = texDesc.addressMode[2] = cudaAddressModeWrap;
	texDesc.sRGB = m_sRgb;
	texDesc.borderColor[0] = texDesc.borderColor[1] = texDesc.borderColor[2] = texDesc.borderColor[3] = 0.0f;
	texDesc.normalizedCoords = true;
	texDesc.maxAnisotropy = 0;
	// The texture read mode depends on the type and size of texels
	// Normalization is only possible for 8- and 16-bit non-floats
	if(channelDesc.f == cudaChannelFormatKindFloat) {
		texDesc.readMode = cudaReadModeElementType;
		texDesc.filterMode = m_mode == SamplingMode::NEAREST ? cudaFilterModePoint : cudaFilterModeLinear;
	} else {
		if(channelDesc.x <= 16 && channelDesc.y <= 16 && channelDesc.z <= 16 && channelDesc.w <= 16) {
			texDesc.filterMode = m_mode == SamplingMode::NEAREST ? cudaFilterModePoint : cudaFilterModeLinear;
			texDesc.readMode = cudaReadModeNormalizedFloat;
		} else {
			if(m_mode == SamplingMode::LINEAR)
				logWarning("[Texture::create_texture_cuda] Textures with integer-components > 16 do not support linear filtering");
			texDesc.filterMode = cudaFilterModePoint;
			texDesc.readMode = cudaReadModeElementType;
		}
	}
	texDesc.mipmapFilterMode = texDesc.filterMode;
	texDesc.mipmapLevelBias = 0.0f;
	texDesc.minMipmapLevelClamp = 0;
	texDesc.maxMipmapLevelClamp = static_cast<float>(m_mipmapLevels);

	// Fill the handle (texture)
	auto& texHdl = m_constHandles.get<ConstTextureDevHandle_t<Device::CUDA>>();
	cuda::check_error(cudaCreateTextureObject(&texHdl.handle, &resDesc, &texDesc, nullptr));
	texHdl.width = m_width;
	texHdl.height = m_height;
	texHdl.depth = m_numLayers;
	texHdl.format = m_format;

	/* Problem: CUDA surfaces bind to exactly one cudaArray, ie. one needs to select the mipmap level prior to binding.
	 * Since surfaces have their own set of issues (e.g. no atomic access) we chose to NOT support CUDA surface creation.
	 * For the sake of a consistent interface one may still call acquire<Device::CUDA>(), but will only ever get a null
	 * handle back.
	 */
}

} // namespace mufflon::scene::textures