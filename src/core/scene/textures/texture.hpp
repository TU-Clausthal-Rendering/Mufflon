#pragma once

#include "core/memory/residency.hpp"
#include "core/memory/accessor.hpp"
#include "util/types.hpp"
#include "util/tagged_tuple.hpp"
#include "util/flag.hpp"
#include <cuda_runtime.h>
#include <memory>
#ifndef __CUDACC__
#include <string_view>
#endif // __CUDACC__

namespace mufflon { namespace scene { namespace textures {

/*
 * A list of supported texture formats for this renderer.
 * While Hardware and texture formats may support many more, this list is rather short
 * because each format must also be implemented in the CpuTexture.
 * A loader must choose the most appropriate target format which is supported internally.
 * Also, some of the formats cannot be aquired for write mode (the RGB ones) on GPU side.
 *
 * Format semantics:
 * ...XU	Unsigned int per channel with X bits
 * ...XF	Float with X bits
 */
enum class Format : u16 {
	R8U,
	RG8U,
	RGBA8U,
	R16U,
	RG16U,
	RGBA16U,
	R16F,
	RG16F,
	RGBA16F,
	R32F,
	RG32F,
	RGBA32F,

	NUM
};

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
	using HandleType = u64;
	using ConstHandleType = u64;
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
inline __host__ __device__ __forceinline__ Pixel get_texture_size(const textures::ConstTextureDevHandle_t<Device::CUDA>& texture) noexcept {
	return { texture.width, texture.height };
}

// Returns the number of layers of a texture based on its handle
// The other specialization is located in cputexture.hpp
inline __host__ __device__ __forceinline__ u16 get_texture_layers(const textures::ConstTextureDevHandle_t<Device::CUDA>& texture) noexcept {
	return texture.depth;
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

#ifndef __CUDACC__
inline constexpr std::string_view FORMAT_NAME(Format format) {
	constexpr std::string_view NAMES[static_cast<std::underlying_type_t<Format>>(Format::NUM)] = {
		"FORMAT_R8U", "FORMAT_RG8U", "FORMAT_RGBA8U",
		"FORMAT_R16U", "FORMAT_RG16U", "FORMAT_RGBA16U",
		"FORMAT_R16F", "FORMAT_RG16F", "FORMAT_RGBA16F",
		"FORMAT_R32F", "FORMAT_RG32F", "FORMAT_RGBA32F",
	};
	return NAMES[static_cast<std::underlying_type_t<Format>>(format)];
}
#endif // __CUDACC__

enum class SamplingMode {
	NEAREST,
	LINEAR
};

/*
 * The texture class handles the resource. Sampling and accessing the data
 * is up to device specific needs and is implemented in textures/interface.hpp.
 */
class Texture {
public:
	static constexpr Device DEFAULT_DEVICE = Device::CPU;
	using HandleTypes = util::TaggedTuple<TextureDevHandle_t<Device::CPU>,
										  TextureDevHandle_t<Device::CUDA>,
										  TextureDevHandle_t<Device::OPENGL>>;
	using ConstHandleTypes = util::TaggedTuple<ConstTextureDevHandle_t<Device::CPU>,
											   ConstTextureDevHandle_t<Device::CUDA>,
											   ConstTextureDevHandle_t<Device::OPENGL>>;

	// Loads a texture into the CPU-RAM
	Texture(u16 width, u16 height, u16 numLayers, Format format, SamplingMode mode,
			bool sRgb, std::unique_ptr<u8[]> data = nullptr);
	Texture(const Texture&) = delete;
	Texture(Texture&&) = default;
	Texture& operator=(const Texture&) = delete;
	Texture& operator=(Texture&&) = default;
	~Texture();

	// Aquire a read-only accessor
	template < Device dev >
	ConstAccessor<TextureDevHandle<dev>> aquireConst() {
		this->synchronize<dev>();
		return ConstAccessor<TextureDevHandle<dev>>{m_constHandles.get<ConstTextureDevHandle_t<dev>>()};
	}

	// Aquire a writing (and thus dirtying) accessor
	template < Device dev >
	Accessor<TextureDevHandle<dev>> aquire() {
		mAssertMsg(NUM_CHANNELS(m_format) != 3, "Write access to RGB formats is not possible on the GPU -> not allowed in all our code.");
		this->synchronize<dev>();
		return Accessor<TextureDevHandle<dev>>{m_handles.get<TextureDevHandle_t<dev>>(), m_dirty};
	}

	// Explicitly synchronize the given device
	template < Device dev >
	void synchronize();

	// Remove a texture from one device.
	// If the resource is the last one, an error is issued and the ressource is not unloaded.
	template < Device dev >
	void unload();

	// Set all values to 0
	template < Device dev >
	void clear();

	constexpr i32 get_width() const noexcept { return m_width; }
	constexpr i32 get_height() const noexcept { return m_height; }
	constexpr i32 get_num_layers() const noexcept { return m_numLayers; }
	constexpr Format get_format() const noexcept { return m_format; }
	constexpr std::size_t get_size() const noexcept {
		return static_cast<std::size_t>(m_width * m_height * m_numLayers * PIXEL_SIZE(m_format));
	}

private:
	// Information
	u16 m_width;
	u16 m_height;
	u16 m_numLayers;
	Format m_format;
	SamplingMode m_mode;
	bool m_sRgb;
	// Handles and resources
	util::DirtyFlags<Device> m_dirty;
	std::unique_ptr<CpuTexture> m_cpuTexture;
	cudaArray_t m_cudaTexture;
	HandleTypes m_handles;
	ConstHandleTypes m_constHandles;

	void create_texture_cpu(std::unique_ptr<u8[]> data = nullptr);
	void create_texture_cuda();
};

}}} // namespace mufflon::scene::textures