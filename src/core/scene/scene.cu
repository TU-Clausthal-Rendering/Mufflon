#include "core/cameras/pinhole.hpp"
#include "core/cameras/focus.hpp"
#include "core/scene/descriptors.hpp"
#include "core/scene/materials/point_medium.hpp"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace mufflon { namespace scene {

__global__ void set_medium(SceneDescriptor<Device::CUDA> scene) {
	// Single-thread kernel ;_;
#ifdef __CUDA_ARCH__
	if(threadIdx.x == 0) {
		cameras::CameraParams& params = scene.camera.get();
		switch(params.type) {
			case cameras::CameraModel::PINHOLE:
				params.mediumIndex = materials::get_point_medium(scene, reinterpret_cast<cameras::PinholeParams&>(params).position);
				break;
			case cameras::CameraModel::FOCUS:
				params.mediumIndex = materials::get_point_medium(scene, reinterpret_cast<cameras::FocusParams&>(params).position);
				break;
			default: mAssert(false);
		}
	}
#endif // __CUDA_ARCH__
}

namespace scene_detail {

void update_camera_medium_cuda(SceneDescriptor<Device::CUDA>& scene) {
	set_medium<<<1, 1>>>(scene);
	cuda::check_error(cudaGetLastError());
}

} // namespace scene_detail


}} // namespace mufflon::scene