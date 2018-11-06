#include "light_tree.hpp"
#include "core/scene/allocator.hpp"
#include "util/assert.hpp"
#include "core/cuda/error.hpp"
#include <cuda_runtime.h>

namespace mufflon { namespace scene { namespace lights {


void LightTree::build(const std::vector<PositionalLights>& posLights,
					  const std::vector<DirectionalLight>& dirLights,
					  std::optional<textures::TextureHandle> envLight) {
	/*if(m_tree.dirLights != nullptr) {

	}*/
}

void synchronize(const LightTree::Tree<Device::CPU>& changed, LightTree::Tree<Device::CUDA>& sync,
				 textures::TextureHandle hdl) {
	// Ensure that the node arrays are in sync (by alloc/realloc if necessary)
	// First directional lights...
	if(changed.numDirLights == 0u) {
		// Remove all data since there are no lights
		sync.dirLights.handle = Allocator<Device::CUDA>::free(sync.dirLights.handle, sync.numDirLights);
	} else {
		// Still have data, (re)alloc and copy
		if(sync.numDirLights == 0u || sync.dirLights.handle == nullptr) {
			mAssert(sync.dirLights.handle == nullptr);
			sync.dirLights.handle = Allocator<Device::CUDA>::alloc_array<LightTree::DirNode>(changed.numDirLights);
		} else {
			sync.dirLights.handle = Allocator<Device::CUDA>::realloc(sync.dirLights.handle, sync.numDirLights,
																	 changed.numDirLights);
		}
		// Copy over the data
		cudaMemcpy(sync.dirLights.handle, changed.dirLights.handle,
				   changed.numDirLights * sizeof(LightTree::DirNode),
				   cudaMemcpyHostToDevice);
	}
	// ...then positional lights
	if(changed.numPosLights == 0u) {
		// Remove all data since there are no lights
		sync.posLights.handle = Allocator<Device::CUDA>::free(sync.posLights.handle, sync.numPosLights);
	} else {
		// Still have data, (re)alloc and copy
		if(sync.numPosLights == 0u || sync.posLights.handle == nullptr) {
			mAssert(sync.posLights.handle == nullptr);
			sync.posLights.handle = Allocator<Device::CUDA>::alloc_array<LightTree::PosNode>(changed.numPosLights);
		} else {
			sync.posLights.handle = Allocator<Device::CUDA>::realloc(sync.posLights.handle, sync.numPosLights,
																	 changed.numPosLights);
		}
		// Copy over the data
		cudaMemcpy(sync.posLights.handle, changed.posLights.handle,
				   changed.numPosLights * sizeof(LightTree::PosNode),
				   cudaMemcpyHostToDevice);
	}

	sync.numDirLights = changed.numDirLights;
	sync.numPosLights = changed.numPosLights;
	// Also copy the environment light
	// TODO: the handle will be invalid!
	sync.envLight.flux = changed.envLight.flux;
	sync.envLight.texHandle = textures::ConstDeviceTextureHandle<Device::CUDA>{ *hdl->aquireConst<Device::CUDA>() };
}

void synchronize(const LightTree::Tree<Device::CUDA>& changed, LightTree::Tree<Device::CPU>& sync,
				 textures::TextureHandle hdl) {
	// Ensure that the node arrays are in sync (by alloc/realloc if necessary)
	// First directional lights...
	if(changed.numDirLights == 0u) {
		// Remove all data since there are no lights
		sync.dirLights.handle = Allocator<Device::CPU>::free(sync.dirLights.handle, sync.numDirLights);
	} else {
		// Still have data, (re)alloc and copy
		if(sync.numDirLights == 0u || sync.dirLights.handle == nullptr) {
			mAssert(sync.dirLights.handle == nullptr);
			sync.dirLights.handle = Allocator<Device::CPU>::alloc_array<LightTree::DirNode>(changed.numDirLights);
		} else {
			sync.dirLights.handle = Allocator<Device::CPU>::realloc(sync.dirLights.handle, sync.numDirLights,
																	 changed.numDirLights);
		}
		// Copy over the data
		cudaMemcpy(sync.dirLights.handle, changed.dirLights.handle,
				   changed.numDirLights * sizeof(LightTree::DirNode),
				   cudaMemcpyDeviceToHost);
	}
	// ...then positional lights
	if(changed.numPosLights == 0u) {
		// Remove all data since there are no lights
		sync.posLights.handle = Allocator<Device::CPU>::free(sync.posLights.handle, sync.numPosLights);
	} else {
		// Still have data, (re)alloc and copy
		if(sync.numPosLights == 0u || sync.posLights.handle == nullptr) {
			mAssert(sync.posLights.handle == nullptr);
			sync.posLights.handle = Allocator<Device::CPU>::alloc_array<LightTree::PosNode>(changed.numPosLights);
		} else {
			sync.posLights.handle = Allocator<Device::CPU>::realloc(sync.posLights.handle, sync.numPosLights,
																	 changed.numPosLights);
		}
		// Copy over the data
		cudaMemcpy(sync.posLights.handle, changed.posLights.handle,
				   changed.numPosLights * sizeof(LightTree::PosNode),
				   cudaMemcpyDeviceToHost);
	}

	sync.numDirLights = changed.numDirLights;
	sync.numPosLights = changed.numPosLights;
	// Also copy the environment light and refresh the handle
	sync.envLight.flux = changed.envLight.flux;
	sync.envLight.texHandle = textures::ConstDeviceTextureHandle<Device::CPU>{ *hdl->aquireConst<Device::CPU>() };
}

void unload(LightTree::Tree<Device::CPU>& tree) {
	tree.dirLights.handle =	Allocator<Device::CPU>::free(tree.dirLights.handle, tree.numDirLights);
	tree.posLights.handle = Allocator<Device::CPU>::free(tree.posLights.handle, tree.numPosLights);
	// TODO: unload envmap handle
}
void unload(LightTree::Tree<Device::CUDA>& tree) {
	tree.dirLights.handle = Allocator<Device::CUDA>::free(tree.dirLights.handle, tree.numDirLights);
	tree.posLights.handle = Allocator<Device::CUDA>::free(tree.posLights.handle, tree.numPosLights);
	// TODO: unload envmap handle
}

}}} // namespace mufflon::scene::lights