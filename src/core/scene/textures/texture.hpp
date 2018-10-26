#pragma once

#include "core/scene/residency.hpp"
#include "core/scene/accessor.hpp"
#include "util/types.hpp"
#include "util/tagged_tuple.hpp"
#include <string>

namespace mufflon { namespace scene { namespace textures {

class CpuTexture;

// Handle type exclusively for textures
template < Device dev >
struct DeviceTextureHandle;

template<>
struct DeviceTextureHandle<Device::CPU> :
	public DeviceHandle<Device::CPU, textures::CpuTexture*> {
};
template<>
struct DeviceTextureHandle<Device::CUDA> :
	public DeviceHandle<Device::CUDA, cudaTextureObject_t> {
};
template<>
struct DeviceTextureHandle<Device::OPENGL> :
	public DeviceHandle<Device::OPENGL, u64> {
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
	Texture(std::string_view fileName);
	Texture(const Texture&) = delete;
	Texture(Texture&&) = default;
	Texture& operator=(const Texture&) = delete;
	Texture& operator=(Texture&&) = default;
	~Texture() = default;

	// Aquire a read-only accessor
	template < Device dev >
	ConstAccessor<DeviceTextureHandle<dev>> aquireConst() {
		this->synchronize<dev>();
		return ConstAccessor<DeviceTextureHandle<dev>>{m_handles.get<DeviceTextureHandle<dev>>().handle};
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