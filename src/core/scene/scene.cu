#include "core/cameras/pinhole.hpp"
#include "core/cameras/focus.hpp"
#include "core/scene/descriptors.hpp"
#include "core/scene/materials/point_medium.hpp"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace mufflon { namespace scene {

__global__ void set_camera_medium(SceneDescriptor<Device::CUDA>* scene) {
	// Single-thread kernel ;_;
#ifdef __CUDA_ARCH__
	if(threadIdx.x == 0) {
		cameras::CameraParams& params = scene->camera.get();
		switch(params.type) {
			case cameras::CameraModel::PINHOLE:
				params.mediumIndex = materials::get_point_medium(*scene, reinterpret_cast<cameras::PinholeParams&>(params).position);
				break;
			case cameras::CameraModel::FOCUS:
				params.mediumIndex = materials::get_point_medium(*scene, reinterpret_cast<cameras::FocusParams&>(params).position);
				break;
			default: mAssert(false);
		}
	}
#endif // __CUDA_ARCH__
}

namespace scene_detail {

void update_camera_medium_cuda(SceneDescriptor<Device::CUDA>& scene) {
	auto cudaSceneDesc = make_udevptr<Device::CUDA, mufflon::scene::SceneDescriptor<Device::CUDA>>();
	copy(cudaSceneDesc.get(), &scene, sizeof(SceneDescriptor<Device::CUDA>));
	set_camera_medium <<<1, 1>>>(cudaSceneDesc.get());
	copy(&scene.camera.get().mediumIndex, &cudaSceneDesc->camera.get().mediumIndex, sizeof(materials::MediumHandle));
	cuda::check_error(cudaStreamSynchronize(0));
	cuda::check_error(cudaGetLastError());
}

} // namespace scene_detail


}} // namespace mufflon::scene