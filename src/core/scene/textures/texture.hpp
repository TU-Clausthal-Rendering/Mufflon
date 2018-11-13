#pragma once

#include "core/memory/residency.hpp"
#include "core/memory/accessor.hpp"
#include "util/types.hpp"
#include "util/tagged_tuple.hpp"
#include <string>
#include <cuda_runtime.h>

namespace mufflon { namespace scene { namespace textures {

class CpuTexture;

// Handle type exclusively for textures
template < Device dev >
struct TextureDevHandle;

// TODO: use this one globally
template < Device dev >
struct DeviceHandle2 {
	static constexpr Device DEVICE = dev;
	DeviceHandle2() = delete; // No instanciation (pure type trait).
};

// CPU
template<>
struct TextureDevHandle<Device::CPU> : public DeviceHandle2<Device::CPU> {
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
	u16 padding_;

	__host__ __device__ operator cudaTextureObject_t () const noexcept { return handle; }
};
struct CudaSurfaceHandle {
	cudaSurfaceObject_t handle;
	u16 width;
	u16 height;
	u16 depth;
	u16 padding_;

	__host__ __device__ operator cudaSurfaceObject_t () const noexcept { return handle; }
};

template<>
struct TextureDevHandle<Device::CUDA> : public DeviceHandle2<Device::CUDA> {
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
struct TextureDevHandle<Device::OPENGL> : public DeviceHandle2<Device::OPENGL> {
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
	RGB8U,
	RGBA8U,
	R16U,
	RG16U,
	RGB16U,
	RGBA16U,
	R32F,
	RG32F,
	RGB32F,
	RGBA32F,
	RGB9E5,		// Special shared exponent format (9-bit mantissa per channel, 5-bit exponent).
	// TODO: 16F

	NUM
};

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
#ifndef __CUDACC__
	Texture(std::string_view fileName);
#endif // __CUDACC__
	Texture(const Texture&) = delete;
	Texture(Texture&&) = default;
	Texture& operator=(const Texture&) = delete;
	Texture& operator=(Texture&&) = default;
	~Texture() = default;

	// Aquire a read-only accessor
	template < Device dev >
	ConstAccessor<TextureDevHandle<dev>> aquireConst() {
		this->synchronize<dev>();
		return ConstAccessor<TextureDevHandle<dev>>{m_constHandles.get<ConstTextureDevHandle_t<dev>>()};
	}

	// Aquire a writing (and thus dirtying) accessor
	template < Device dev >
	Accessor<TextureDevHandle<dev>> aquire() {
		this->synchronize<dev>();
		return Accessor<TextureDevHandle<dev>>{m_handles.get<TextureDevHandle_t<dev>>(), m_dirty};
	}

	// Explicitly synchronize the given device
	template < Device dev >
	void synchronize() {
		if(m_dirty.has_competing_changes())
			throw std::runtime_error("Failure: competing changes for this texture");
		if(m_dirty.needs_sync(dev)) {
			// TODO create texture resources if necessary
		}
		m_dirty.mark_synced(dev);
	}

	// Remove a texture from one device. An error is issued if the target
	// is the last device in which case the resource is not released.
	template < Device dev >
	void release() {

	}

private:
	std::string m_srcFileName;
	util::DirtyFlags<Device> m_dirty;
	std::unique_ptr<CpuTexture> m_cpuTexture;
	HandleTypes m_handles;
	ConstHandleTypes m_constHandles;
};

using TextureHandle = std::shared_ptr<Texture>;

}}} // namespace mufflon::scene::textures