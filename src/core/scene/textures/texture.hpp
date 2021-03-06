#pragma once

#include "format.hpp"
#include "util/types.hpp"
#include "util/tagged_tuple.hpp"
#include "util/string_view.hpp"
#include "core/export/core_api.h"
#include "core/memory/residency.hpp"
#include "core/opengl/gl_texture.hpp"
#include <array>
#include <cuda_runtime.h>
#include <memory>
#include <string>

namespace mufflon { namespace scene { namespace textures {

class CpuTexture;

// Handle type exclusively for textures
template < Device dev >
struct TextureDevHandle;

// CPU
template<>
struct TextureDevHandle<Device::CPU> : public DeviceHandle<Device::CPU> {
	using HandleType = textures::CpuTexture*;
	using ConstHandleType = const textures::CpuTexture*;
};
__host__ constexpr bool is_valid(typename TextureDevHandle<Device::CPU>::HandleType handle) noexcept {
	return handle != nullptr;
}
__host__ constexpr bool is_valid(typename TextureDevHandle<Device::CPU>::ConstHandleType handle) noexcept {
	return handle != nullptr;
}

// CUDA
/*
 * Wrapper around cudaTextureObject_t. We need this extra layer, because their is no
 * chance to get the texture size of a texture on device side.
 */
struct CudaTextureHandle {
	cudaTextureObject_t handle;
	u16 width;
	u16 height;
	u16 depth;
	Format format;

	__host__ __device__ operator cudaTextureObject_t () const noexcept { return handle; }
};
struct CudaSurfaceHandle {
	cudaSurfaceObject_t handle;
	u16 width;
	u16 height;
	u16 depth;
	Format format;

	__host__ __device__ operator cudaSurfaceObject_t () const noexcept { return handle; }
};

// Ensure handle sizes
static_assert(sizeof(CudaTextureHandle) == 2u * 8u, "CUDA texture handle doesn't match expected size");
static_assert(sizeof(CudaSurfaceHandle) == 2u * 8u, "CUDA surface handle doesn't match expected size");

template<>
struct TextureDevHandle<Device::CUDA> : public DeviceHandle<Device::CUDA> {
	using HandleType = CudaSurfaceHandle;
	using ConstHandleType = CudaTextureHandle;
};
__host__ __device__ constexpr bool is_valid(typename TextureDevHandle<Device::CUDA>::HandleType handle) noexcept {
	return handle.handle != 0u;
}
__host__ __device__ constexpr bool is_valid(typename TextureDevHandle<Device::CUDA>::ConstHandleType handle) noexcept {
	return handle.handle != 0u;
}

// OPENGL
template<>
struct TextureDevHandle<Device::OPENGL> : public DeviceHandle<Device::OPENGL> {
	using HandleType = gl::TextureHandle;
	using ConstHandleType = gl::TextureHandle;
};
__host__ constexpr bool is_valid(typename TextureDevHandle<Device::OPENGL>::HandleType handle) noexcept {
	return handle != 0u;
}


// Short type alias
template < Device dev >
using TextureDevHandle_t = typename TextureDevHandle<dev>::HandleType;
template < Device dev >
using ConstTextureDevHandle_t = typename TextureDevHandle<dev>::ConstHandleType;

// Returns the size of a texture based on its handle
// The other overload is located in cputexture.hpp
CUDA_FUNCTION __forceinline__ Pixel get_texture_size(const textures::ConstTextureDevHandle_t<Device::CUDA>& texture) noexcept {
	return { texture.width, texture.height };
}

// Returns the number of layers of a texture based on its handle
// The other specialization is located in cputexture.hpp
CUDA_FUNCTION __forceinline__ u16 get_texture_layers(const textures::ConstTextureDevHandle_t<Device::CUDA>& texture) noexcept {
	return texture.depth;
}

// Returns the number of channels per pixel.
// The other specialization is located in cputexture.hpp
CUDA_FUNCTION __forceinline__ u16 get_texture_channel_count(const textures::ConstTextureDevHandle_t<Device::CUDA>& texture) noexcept {
	constexpr char CHANNEL_COUNT[int(Format::NUM)] = {
		1, 2, 4, 1, 2, 4, 1, 2, 4, 1, 2, 4
	};
	return CHANNEL_COUNT[int(texture.format)];
}


inline constexpr size_t PIXEL_SIZE(Format format) {
	constexpr u8 PIXEL_SIZES[int(Format::NUM)] = {
		1, 2, 4, // ...8U formats
		2, 4, 8, // ...16U formats
		2, 4, 8, // ...16F formats
		//4, 8, 16, // ...32U formats
		4, 8, 16 // ...32F formats
	};
	return PIXEL_SIZES[int(format)];
}

inline constexpr int NUM_CHANNELS(Format format) {
	constexpr u8 NUM_CHANNELS[int(Format::NUM)] = {
		1, 2, 4, // ...8U formats
		1, 2, 4, // ...16U formats
		1, 2, 4, // ...16F formats
		//1, 2, 4, // ...32U formats
		1, 2, 4 // ...32F formats
	};
	return NUM_CHANNELS[int(format)];
}

inline StringView FORMAT_NAME(Format format) {
	static std::array<StringView, static_cast<std::underlying_type_t<Format>>(Format::NUM)> NAMES = { {
		StringView{"FORMAT_R8U"}, StringView{"FORMAT_RG8U"}, StringView{"FORMAT_RGBA8U"},
		StringView{"FORMAT_R16U"}, StringView{"FORMAT_RG16U"}, StringView{"FORMAT_RGBA16U"},
		StringView{"FORMAT_R16F"}, StringView{"FORMAT_RG16F"}, StringView{"FORMAT_RGBA16F"},
		StringView{"FORMAT_R32F"}, StringView{"FORMAT_RG32F"}, StringView{"FORMAT_RGBA32F"},
	} };
	return NAMES[static_cast<std::underlying_type_t<Format>>(format)];
}

enum class SamplingMode {
	NEAREST,
	LINEAR
};

enum class MipmapType {
	NONE,
	AVG,
	MIN,
	MAX
};

/*
 * The texture class handles the resource. Sampling and accessing the data
 * is up to device specific needs and is implemented in textures/interface.hpp.
 */
class Texture {
public:
	using HandleTypes = util::TaggedTuple<TextureDevHandle_t<Device::CPU>,
										  TextureDevHandle_t<Device::CUDA>,
										  TextureDevHandle_t<Device::OPENGL>>;
	using ConstHandleTypes = util::TaggedTuple<ConstTextureDevHandle_t<Device::CPU>,
											   ConstTextureDevHandle_t<Device::CUDA>,
											   ConstTextureDevHandle_t<Device::OPENGL>>;

	// Loads a texture into the CPU-RAM
	Texture(std::string name, u16 width, u16 height, u16 numLayers, MipmapType mipmapType, Format format,
			SamplingMode mode, bool sRgb, bool dataHasMipmaps = false, std::unique_ptr<u8[]> data = nullptr);
	Texture(const Texture&) = delete;
	Texture(Texture&&) noexcept;
	Texture& operator=(const Texture&) = delete;
	Texture& operator=(Texture&&) = delete;
	~Texture();

	const std::string& get_name() const noexcept { return m_name; }

	// Aquire a read-only accessor
	template < Device dev >
	ConstTextureDevHandle_t<dev> acquire_const() {
		this->synchronize<dev>();
		return m_constHandles.get<ConstTextureDevHandle_t<dev>>();
	}

	// Aquire a writing (and thus dirtying) accessor
	template < Device dev >
	TextureDevHandle_t<dev> acquire() {
		mAssertMsg(NUM_CHANNELS(m_format) != 3, "Write access to RGB formats is not possible on the GPU -> not allowed in all our code.");
		this->synchronize<dev>();
		return m_handles.get<TextureDevHandle_t<dev>>();
	}

	void mark_changed(Device changed) noexcept {
		if(changed != Device::CPU)
			this->unload<Device::CPU>();
		if(changed != Device::CUDA)
			this->unload<Device::CUDA>();
		if(changed != Device::OPENGL)
			this->unload<Device::OPENGL>();
	}

	// Explicitly synchronize the given device
	template < Device dev >
	void synchronize();

	// Remove a texture from one device.
	template < Device dev >
	void unload();

	// Set all values to 0
	template < Device dev >
	void clear();

	constexpr bool is_sRgb() const noexcept { return m_sRgb; }
	constexpr i32 get_width() const noexcept { return m_width; }
	constexpr i32 get_height() const noexcept { return m_height; }
	constexpr i32 get_num_layers() const noexcept { return m_numLayers; }
	constexpr Format get_format() const noexcept { return m_format; }
	constexpr SamplingMode get_sampling_mode() const noexcept { return m_mode; }
	constexpr std::size_t get_size() const noexcept {
		return static_cast<std::size_t>(m_width * m_height * m_numLayers * PIXEL_SIZE(m_format));
	}

	static gl::Handle get_gl_sampler(SamplingMode mode);
private:
	// Information
	u16 m_width;
	u16 m_height;
	u16 m_numLayers;
	MipmapType m_mipmapType;
	u32 m_mipmapLevels;
	Format m_format;
	SamplingMode m_mode;
	bool m_sRgb;
	// Handles and resources
	std::unique_ptr<CpuTexture> m_cpuTexture;
	cudaMipmappedArray_t m_cudaTexture;
	gl::Handle m_glHandle = 0;
	gl::TextureFormat m_glFormat = {};
	HandleTypes m_handles;
	ConstHandleTypes m_constHandles;
	std::string m_name;

	void create_texture_cpu(bool dataHasMipmaps = false, std::unique_ptr<u8[]> data = nullptr);
	void create_texture_cuda();
	void create_texture_opengl();
};

}}} // namespace mufflon::scene::textures
