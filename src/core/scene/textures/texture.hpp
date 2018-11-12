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
struct DeviceTextureHandle;
template < Device dev >
struct ConstDeviceTextureHandle;

template<>
struct DeviceTextureHandle<Device::CPU> :
	public DeviceHandle<Device::CPU, textures::CpuTexture*> {
	__host__ __device__ constexpr bool is_valid() const noexcept {
		return handle != nullptr;
	}
};
template<>
struct ConstDeviceTextureHandle<Device::CPU> :
	public ConstDeviceHandle<Device::CPU, const textures::CpuTexture*> {
	__host__ __device__ constexpr bool is_valid() const noexcept {
		return handle != nullptr;
	}
};
// TODO: Const handle and stuff
template<>
struct DeviceTextureHandle<Device::CUDA> :
	public DeviceHandle<Device::CUDA, cudaSurfaceObject_t> {
	__host__ __device__ constexpr bool is_valid() const noexcept {
		return handle != 0u;
	}
};
template<>
struct ConstDeviceTextureHandle<Device::CUDA> :
	public ConstDeviceHandle<Device::CUDA, cudaTextureObject_t> {
	__host__ __device__ constexpr bool is_valid() const noexcept {
		return handle != 0u;
	}
};
template<>
struct DeviceTextureHandle<Device::OPENGL> :
	public DeviceHandle<Device::OPENGL, u64> {
	__host__ __device__ constexpr bool is_valid() const noexcept {
		return handle != 0u;
	}
};
template<>
struct ConstDeviceTextureHandle<Device::OPENGL> :
	public ConstDeviceHandle<Device::OPENGL, u64> {
	__host__ __device__ constexpr bool is_valid() const noexcept {
		return handle != 0u;
	}
};


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
 * is up to device specific needs.
 */
class Texture {
public:
	static constexpr Device DEFAULT_DEVICE = Device::CPU;
	using HandleTypes = util::TaggedTuple<DeviceTextureHandle<Device::CPU>,
								DeviceTextureHandle<Device::CUDA>,
								DeviceTextureHandle<Device::OPENGL>>;

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
	ConstAccessor<ConstDeviceTextureHandle<dev>> aquireConst() {
		this->synchronize<dev>();
		return ConstAccessor<ConstDeviceTextureHandle<dev>>{m_handles.get<DeviceTextureHandle<dev>>().handle};
	}

	// Aquire a writing (and thus dirtying) accessor
	template < Device dev >
	Accessor<DeviceTextureHandle<dev>> aquire() {
		this->synchronize<dev>();
		return Accessor<DeviceTextureHandle<dev>>{m_handles.get<DeviceTextureHandle<dev>>().handle, m_dirty};
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
	// TODO Cuda resoures
	// TODO OGL resoures
	// TODO: move into CPUTexture

};

using TextureHandle = std::shared_ptr<Texture>;

}}} // namespace mufflon::scene::textures