#include "cpu_pt.hpp"
#include "path_util.hpp"
#include "core/cameras/camera.hpp"
#include "core/cameras/sample.hpp"

namespace mufflon::renderer {

CpuPathTracer::CpuPathTracer(scene::SceneHandle scene) :
	m_currentScene(scene),
	m_cameraParams(scene->get_camera()->get_parameter_pack_size())
{
	// Make sure the scene is loaded completely for the use on CPU side
	scene->synchronize<scene::Device::CPU>();
	scene->get_camera()->get_parameter_pack(reinterpret_cast<cameras::CameraParams*>(m_cameraParams.data()));

	// The PT does not need additional memory resources like photon maps.
}

void CpuPathTracer::iterate(OutputHandler* outputBuffer) const {
	// TODO: call sample in a parallel way for each output pixel
}

void CpuPathTracer::sample(const Pixel coord, OutputHandler* outputBuffer) const {
	// TODO: Create a start for the path
	//RaySample camVertex = sample_ray(reinterpret_cast<const cameras::CameraParams*>(m_cameraParams.data()),
	//	coord, outputBuffer->get_resolution(), rndSet);
	// TODO setup from camera
	PathHead head{
		Interaction::CAMERA
	};

	int pathLen = 0;
	do {
		if(pathLen < 666) { // TODO: min and max pathLen bounds
			// TODO: Call NEE member function for the camera start/recursive vertices
			// TODO: Walk
			//head = walk(head);
		}
		++pathLen;
	} while(pathLen < 666 && head.type != Interaction::VOID);

	// Random walk ended because of missing the scene?
	if(pathLen < 666 && head.type == Interaction::VOID) {
		// TODO: fetch background
	}
}

} // namespace mufflon::renderer