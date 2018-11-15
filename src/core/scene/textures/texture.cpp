#include "texture.hpp"
#include "cputexture.hpp"
#include "util/log.hpp"
#include "core/memory/synchronize.hpp"

namespace mufflon::scene::textures {

Texture::Texture(u16 width, u16 height, u16 numLayers, Format format, SamplingMode mode, bool sRgb, void* data) :
	m_width(width),
	m_height(height),
	m_numLayers(numLayers),
	m_format(format),
	m_mode(mode),
	m_sRgb(sRgb),
	m_cudaTexture(nullptr)
{
	if(data) {
		// A file loader provides an array with pixel data. This is loaded into
		// a CPUTexture per default.
		create_texture_cpu();
		copy<Device::CPU, Device::CPU>(m_cpuTexture->data(), data, m_width * m_height * PIXEL_SIZE(m_format));
	}
}

Texture::~Texture() {
	m_cpuTexture = nullptr;
	if(m_cudaTexture) {
		cudaDestroyTextureObject(m_constHandles.get<ConstTextureDevHandle_t<Device::CUDA>>().handle);
		cudaDestroySurfaceObject(m_handles.get<TextureDevHandle_t<Device::CUDA>>().handle);
		cudaFreeArray(m_cudaTexture);
	}
}

template < Device dev >
void Texture::synchronize() {
	if(m_dirty.has_competing_changes())
		logError("[Texture::synchronize] Competing changes for this texture. Some changes will be lost.");
	if(m_dirty.needs_sync(dev)) {
		// Create texture resources if necessary
		if(!is_valid(m_handles.get<TextureDevHandle_t<dev>>())) {
			switch(dev) {
				case Device::CPU: create_texture_cpu(); break;
				case Device::CUDA: create_texture_cuda(); break;
				//case Device::OPENGL: create_texture_opengl(); break;
			}
		}
		// Copy the memory (wherever it changed)
		if((dev == Device::CUDA) && m_dirty.has_changes(Device::CPU))
			copy<dev, Device::CPU>(m_cudaTexture, m_cpuTexture->data(), m_width * m_height * PIXEL_SIZE(m_format));
		if((dev == Device::CPU) && m_dirty.has_changes(Device::CUDA))
			copy<dev, Device::CUDA>(m_cpuTexture->data(), m_cudaTexture, m_width * m_height * PIXEL_SIZE(m_format));
	}
	m_dirty.mark_synced(dev);
}
template void Texture::synchronize<Device::CPU>();
template void Texture::synchronize<Device::CUDA>();
template void Texture::synchronize<Device::OPENGL>();



template < Device dev >
void Texture::unload() {
	switch(dev) {
		case Device::CPU: {
			if(!m_cudaTexture) {
				logError("[Texture::unload] The CPU texture is the last instance. If unloaded the texture object would be invalid.");
				return;
			}
			m_handles.get<TextureDevHandle_t<Device::CPU>>() = nullptr;
			m_constHandles.get<ConstTextureDevHandle_t<Device::CPU>>() = nullptr;
			m_cpuTexture = nullptr;
		} break;
		case Device::CUDA: {
			if(!m_cpuTexture) {
				logError("[Texture::unload] The CUDA texture is the last instance. If unloaded the texture object would be invalid.");
				return;
			}
			if(m_cudaTexture) {
				// Free
				auto& texHdl = m_constHandles.get<ConstTextureDevHandle_t<Device::CUDA>>();
				auto& surfHdl = m_handles.get<TextureDevHandle_t<Device::CUDA>>();
				cudaDestroyTextureObject(texHdl.handle);
				cudaDestroySurfaceObject(surfHdl.handle);
				cudaFreeArray(m_cudaTexture);
				// Reset handles
				texHdl.handle = 0;
				surfHdl.handle = 0;
				m_cpuTexture = nullptr;
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
			if(!m_cpuTexture)
				logError("[Texture::zero] Trying to clear a CPU texture without memory.");
			else memset(m_cpuTexture->data(), 0, texMemSize);
		} break;
		case Device::CUDA: {
			if(!m_cpuTexture)
				logError("[Texture::zero] Trying to clear a CUDA texture without memory.");
			else {
				// Is the dummy memory to small?
				if(s_zeroMem.size() < texMemSize) {
					s_zeroMem.resize(texMemSize);
					memset(s_zeroMem.data(), 0, texMemSize);
				}
				cudaMemcpyToArray(m_cudaTexture, 0, 0, s_zeroMem.data(), texMemSize, cudaMemcpyHostToDevice);
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




void Texture::create_texture_cpu() {
	m_cpuTexture = std::make_unique<CpuTexture>(m_width, m_height, m_numLayers, m_format, m_mode, m_sRgb);
	m_handles.get<TextureDevHandle_t<Device::CPU>>() = m_cpuTexture.get();
	m_constHandles.get<ConstTextureDevHandle_t<Device::CPU>>() = m_cpuTexture.get();
}

void Texture::create_texture_cuda() {
	cudaChannelFormatDesc channelDesc;
	switch(m_format) {
		case Format::R8U: channelDesc = cudaCreateChannelDesc(8, 0, 0, 0, cudaChannelFormatKindUnsigned); break;
		case Format::RG8U: channelDesc = cudaCreateChannelDesc(8, 8, 0, 0, cudaChannelFormatKindUnsigned); break;
		case Format::RGB8U: channelDesc = cudaCreateChannelDesc(8, 8, 8, 0, cudaChannelFormatKindUnsigned); break;
		case Format::RGBA8U: channelDesc = cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsigned); break;
		case Format::R16U: channelDesc = cudaCreateChannelDesc(16, 0, 0, 0, cudaChannelFormatKindUnsigned); break;
		case Format::RG16U: channelDesc = cudaCreateChannelDesc(16, 16, 0, 0, cudaChannelFormatKindUnsigned); break;
		case Format::RGB16U: channelDesc = cudaCreateChannelDesc(16, 16, 16, 0, cudaChannelFormatKindUnsigned); break;
		case Format::RGBA16U: channelDesc = cudaCreateChannelDesc(16, 16, 16, 16, cudaChannelFormatKindUnsigned); break;
		case Format::R32F: channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat); break;
		case Format::RG32F: channelDesc = cudaCreateChannelDesc(32, 32, 0, 0, cudaChannelFormatKindFloat); break;
		case Format::RGB32F: channelDesc = cudaCreateChannelDesc(32, 32, 32, 0, cudaChannelFormatKindFloat); break;
		case Format::RGBA32F: channelDesc = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat); break;
		case Format::RGB9E5: channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindUnsigned); break;
		default:
			mAssertMsg(false, "Format not implemented.");
	} 
	cudaMallocArray(&m_cudaTexture, &channelDesc, m_width, m_height);

	// Specify the texture view on the memory
	cudaResourceDesc resDesc;
	resDesc.resType = cudaResourceTypeArray;
	resDesc.res.array.array = m_cudaTexture;
	cudaTextureDesc texDesc;
	texDesc.addressMode[0] = texDesc.addressMode[1] = texDesc.addressMode[2] = cudaAddressModeWrap;
	texDesc.filterMode = m_mode == SamplingMode::NEAREST ? cudaFilterModePoint : cudaFilterModeLinear;
	texDesc.readMode = cudaReadModeNormalizedFloat;
	texDesc.sRGB = m_sRgb;
	texDesc.borderColor[0] = texDesc.borderColor[1] = texDesc.borderColor[2] = texDesc.borderColor[3] = 0.0f;
	texDesc.normalizedCoords = true;
	texDesc.maxAnisotropy = 0;
	texDesc.mipmapFilterMode = cudaFilterModePoint;
	texDesc.mipmapLevelBias = 0.0f;
	texDesc.minMipmapLevelClamp = 0;
	texDesc.maxMipmapLevelClamp = 0;

	// Fill the handle (texture)
	auto& texHdl = m_constHandles.get<ConstTextureDevHandle_t<Device::CUDA>>();
	cudaCreateTextureObject(&texHdl.handle, &resDesc, &texDesc, nullptr);
	texHdl.width = m_width;
	texHdl.height = m_height;
	texHdl.depth = m_numLayers;

	// Fill the handle (surface)
	if(NUM_CHANNELS(m_format) != 3) // Allow read-only RGB formats without causing errors here
	{
		auto& surfHdl = m_handles.get<TextureDevHandle_t<Device::CUDA>>();
		cudaCreateSurfaceObject(&surfHdl.handle, &resDesc);
		surfHdl.width = m_width;
		surfHdl.height = m_height;
		surfHdl.depth = m_numLayers;
	}
}

} // namespace mufflon::scene::textures