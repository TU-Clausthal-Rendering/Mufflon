#include "cpu_pt.hpp"
#include "path_util.hpp"
#include "random_walk.hpp"
#include "output_handler.hpp"
#include "profiler/cpu_profiler.hpp"
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
	scene->synchronize<Device::CPU>();
	// TODO: init one RNG per thread?
	m_rngs.emplace_back(static_cast<u32>(std::random_device()()));

	// The PT does not need additional memory resources like photon maps.
}

void CpuPathTracer::iterate(OutputHandler& outputBuffer) {
	auto scope = Profiler::instance().start<CpuProfileState>("CPU PT iteration", ProfileLevel::LOW);
	// (Re) create the random number generators
	if(m_rngs.size() != outputBuffer.get_num_pixels()
		|| m_reset)
		init_rngs(outputBuffer.get_num_pixels());

	RenderBuffer<Device::CPU> buffer = outputBuffer.begin_iteration<Device::CPU>(m_reset);
	m_currentScene->get_camera()->get_parameter_pack(as<cameras::CameraParams>(m_camParams),
													 Device::CPU, buffer.get_resolution());
	m_reset = false;
	scene::SceneDescriptor<Device::CPU> sceneDesc = m_currentScene->get_descriptor<Device::CPU>({}, {}, {});

	// TODO: call sample in a parallel way for each output pixel
	// TODO: better pixel order?
	// TODO: different scheduling?
//#pragma omp parallel for
	for(int pixel = 0; pixel < outputBuffer.get_num_pixels(); ++pixel) {
		this->sample(Pixel{ pixel % outputBuffer.get_width(), pixel / outputBuffer.get_width() }, buffer, sceneDesc);
	}

	outputBuffer.end_iteration<Device::CPU>();

	Profiler::instance().create_snapshot_all();
}

void CpuPathTracer::reset() {
	this->m_reset = true;
}

void CpuPathTracer::sample(const Pixel coord, RenderBuffer<Device::CPU>& outputBuffer,
						   const scene::SceneDescriptor<Device::CPU>& scene) {
	int pixel = coord.x + coord.y * outputBuffer.get_width();

	Throughput throughput{ ei::Vec3{1.0f}, 1.0f };
	u8 vertexBuffer[256]; // TODO: depends on materials::MAX_MATERIAL_PARAMETER_SIZE
	PtPathVertex* vertex = as<PtPathVertex>(vertexBuffer);
	// Create a start for the path
	math::PositionSample camPos = camera_sample_position(get_cam(), coord,
														 m_rngs[pixel].next());
	int s = PtPathVertex::create_camera(&vertex, &vertex, get_cam(), camPos, coord);
	mAssertMsg(s < 256, "vertexBuffer overflow.");

	int pathLen = 0;
	do {
		if(pathLen+1 >= m_params.maxPathLength) {
			// Call NEE member function for the camera start/recursive vertices
			// TODO: test/parametrize mulievent estimation (more indices in connect) and different guides.
			u64 neeSeed = m_rngs[pixel].next();
			auto nee = connect(scene.lightTree, 0, 1, neeSeed,
				vertex->get_position(), m_currentScene->get_bounding_box(),
				math::RndSet2{ m_rngs[pixel].next() }, scene::lights::guide_flux);
			bool anyhit = false; // TODO use a real anyhit method
			if(!anyhit) {
				auto value = vertex->evaluate(nee.direction, scene.media);
				AreaPdf hitPdf = value.pdfF.to_area_pdf(nee.cosOut, nee.distSq);
				float mis = 1.0f / (1.0f + hitPdf / nee.creationPdf);
				outputBuffer.contribute(coord, throughput, { Spectrum{nee.intensity}, 1.0f },
					value.cosOut, value.value * mis);
			}
		}

		// Walk
		scene::Point lastPosition = vertex->get_position();
		math::RndSet2_1 rnd { m_rngs[pixel].next(), m_rngs[pixel].next() };
		if(!walk(scene, *vertex, rnd, 0.0f, false, throughput, vertex))
			break;
		++pathLen;

		// Evaluate direct hit of area ligths
		if(pathLen >= m_params.maxPathLength) {
			Spectrum emission = vertex->get_emission();
			if(emission != 0.0f) {
				AreaPdf backwardPdf = connect_pdf(scene.lightTree, 0,
												  lastPosition, scene::lights::guide_flux);
				float mis = 1.0f / (1.0f + backwardPdf / vertex->get_incident_pdf());
				outputBuffer.contribute(coord, throughput, emission, vertex->get_position(),
					vertex->get_normal(), vertex->get_albedo());
			}
		}
	} while(pathLen < m_params.maxPathLength);

	// Random walk ended because of missing the scene?
	if(pathLen < m_params.maxPathLength) {
		// TODO: fetch background
		// TODO: normals, position???
		const float phi = 2.f * ei::PI * coord.x  / static_cast<float>(outputBuffer.get_width());
		const float theta = ei::PI * coord.y / static_cast<float>(outputBuffer.get_height());
		ei::Vec3 testRadiance = scene.lightTree.background.get_color(ei::Vec3{
			sinf(theta) * cosf(phi),
			sinf(theta) * sinf(phi),
			cosf(theta)
		});
		outputBuffer.contribute(coord, throughput, testRadiance,
								ei::Vec3{ 0, 0, 0 }, ei::Vec3{ 0, 0, 0 },
								ei::Vec3{ 0, 0, 0 });
	}
}

void CpuPathTracer::init_rngs(int num) {
	m_rngs.resize(num);
	// TODO: incude some global seed into the initialization
	for(int i = 0; i < num; ++i)
		m_rngs[i] = math::Rng(i);
}

} // namespace mufflon::renderer