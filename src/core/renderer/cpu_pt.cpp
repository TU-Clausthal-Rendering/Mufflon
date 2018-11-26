#include "cpu_pt.hpp"
#include "path_util.hpp"
#include "random_walk.hpp"
#include "output_handler.hpp"
#include "core/cameras/camera.hpp"
#include "core/cameras/camera_sampling.hpp"
#include "core/scene/materials/medium.hpp"
#include "core/math/rng.hpp"
#include <random>

namespace mufflon::renderer {

using PtPathVertex = PathVertex<u8, 4>;

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
	// Create a start for the path
	const cameras::CameraParams& cam = *m_currentScene->get_camera_data<Device::CPU>();
	cameras::RndSet camRnd { math::sample_uniform(m_rngs[pixel].next()),
							 math::sample_uniform(m_rngs[pixel].next()) };
	cameras::RaySample camSample = camera_sample_ray(cam, coord, ei::Vec2{outputBuffer.get_resolution()}, camRnd);
	// Setup first "vertex" from camera
	//ei::Ray ray{camSample.origin, camSample.excident};
	Throughput throughput {ei::Vec3{1.0f}, 1.0f};
	PtPathVertex vertex; // TODO: larger buffer
	scene::materials::Medium medium;


	/*PathHead head {
		Throughput{ei::Vec3{camVertex.w / float(camVertex.pdf)}, 1.0f},
		camVertex.origin,
		camVertex.pdf,
		ei::Vec3{0.0f},
		AngularPdf{0.0f},
		cam.type == cameras::CameraModel::ORTHOGRAPHIC ? Interaction::CAMERA_ORTHO : Interaction::CAMERA
	};*/

	int pathLen = 0;
	do {
		if(pathLen < 666) { // TODO: min and max pathLen bounds

			// TODO: Call NEE member function for the camera start/recursive vertices
			// TODO: Walk
			scene::materials::RndSet rnd { math::sample_uniform(m_rngs[pixel].next()), m_rngs[pixel].next() >> 32 };
			scene::materials::Sample sample = vertex.sample(rnd);
			scene::accel_struct::RayIntersectionResult nextHit;
			if(!walk(ei::Ray(vertex.get_position(), sample.excident), sample.pdfF,
				sample.throughput, medium, 0.0f, throughput, nextHit))
				break;
		}
		++pathLen;
	} while(pathLen < 666);

	// Random walk ended because of missing the scene?
	if(pathLen < 666) {
		// TODO: fetch background
		// TODO: normals, position???
		constexpr ei::Vec3 colors[4]{
			ei::Vec3{1, 0, 0},
			ei::Vec3{0, 1, 0},
			ei::Vec3{0, 0, 1},
			ei::Vec3{1, 1, 1}
		};
		
		ei::Vec2 xy { coord.x / static_cast<float>(outputBuffer.get_width()),
					  coord.y / static_cast<float>(outputBuffer.get_height()) };
		xy *= math::sample_uniform( m_rngs[pixel].next() );
		ei::Vec3 testRadiance = colors[0u] * (1.f - xy.x)*(1.f - xy.y) + colors[1u] * xy.x*(1.f - xy.y)
			+ colors[2u] * (1.f - xy.x)*xy.y + colors[3u] * xy.x*xy.y;
		outputBuffer.contribute(coord, throughput, testRadiance,
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