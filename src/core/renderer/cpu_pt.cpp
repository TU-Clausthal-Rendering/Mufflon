#include "cpu_pt.hpp"
#include "path_util.hpp"
#include "output_handler.hpp"
#include "core/cameras/camera.hpp"
#include "core/cameras/camera_sampling.hpp"
#include "core/math/rng.hpp"
#include <random>

namespace mufflon::renderer {

CpuPathTracer::CpuPathTracer(scene::SceneHandle scene) :
	m_currentScene(scene)
{
	// Make sure the scene is loaded completely for the use on CPU side
	scene->synchronize<Device::CUDA>();
	// TODO: init one RNG per thread?
	m_rngs.emplace_back(static_cast<u32>(std::random_device()()));

	// The PT does not need additional memory resources like photon maps.
}

void CpuPathTracer::iterate(OutputHandler& outputBuffer) {
	// (Re) create the random number generators
	if(m_rngs.size() != outputBuffer.get_num_pixels()
		|| m_reset)
		initRngs(outputBuffer.get_num_pixels());

	RenderBuffer<Device::CPU> buffer = outputBuffer.begin_iteration<Device::CPU>(m_reset);
	m_reset = false;

	// TODO: call sample in a parallel way for each output pixel
	// TODO: better pixel order?
	// TODO: different scheduling?
//#pragma omp parallel for
	for(int pixel = 0; pixel < outputBuffer.get_num_pixels(); ++pixel) {
		this->sample(Pixel{ pixel % outputBuffer.get_width(), pixel / outputBuffer.get_width() }, buffer);
	}

	outputBuffer.end_iteration<Device::CPU>();
}

void CpuPathTracer::reset() {
	this->m_reset = true;
}

void CpuPathTracer::sample(const Pixel coord, RenderBuffer<Device::CPU>& outputBuffer) {
	int pixel = coord.x + coord.y * outputBuffer.get_width();
	// TODO: Create a start for the path
	//RaySample camVertex = sample_ray(reinterpret_cast<const cameras::CameraParams*>(m_cameraParams.data()),
	//	coord, outputBuffer->get_resolution(), rndSet);
	// TODO setup from camera
	PathHead head{
		Throughput{ei::Vec3{1.0f}, 1.0f}
	};

	int pathLen = 0;
	do {
		if(pathLen < 666) { // TODO: min and max pathLen bounds

			// TODO: Call NEE member function for the camera start/recursive vertices
			// TODO: Walk
			//head = walk(head);
			break;
		}
		++pathLen;
	} while(pathLen < 666 && head.type != Interaction::VOID);

	// Random walk ended because of missing the scene?
	if(pathLen < 666 && head.type == Interaction::VOID) {
		// TODO: fetch background
		// TODO: normals, position???
		constexpr ei::Vec3 colors[4]{
			ei::Vec3{1, 0, 0},
			ei::Vec3{0, 1, 0},
			ei::Vec3{0, 0, 1},
			ei::Vec3{1, 1, 1}
		};
		
		ei::Vec2 xy { coord.x / static_cast<float>(m_currentScene->get_resolution().x),
					  coord.y / static_cast<float>(m_currentScene->get_resolution().y) };
		xy *= math::sample_uniform( m_rngs[pixel].next() );
		ei::Vec3 testRadiance = colors[0u] * (1.f - xy.x)*(1.f - xy.y) + colors[1u] * xy.x*(1.f - xy.y)
			+ colors[2u] * (1.f - xy.x)*xy.y + colors[3u] * xy.x*xy.y;
		outputBuffer.contribute(coord, head.throughput, testRadiance,
								ei::Vec3{ 0, 0, 0 }, ei::Vec3{ 0, 0, 0 },
								ei::Vec3{ 0, 0, 0 });
	}
}

void CpuPathTracer::initRngs(int num) {
	m_rngs.resize(num);
	// TODO: incude some global seed into the initialization
	for(int i = 0; i < num; ++i)
		m_rngs[i] = math::Rng(i);
}

} // namespace mufflon::renderer