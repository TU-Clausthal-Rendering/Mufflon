#include "cpu_pt.hpp"
#include "path_util.hpp"
#include "random_walk.hpp"
#include "output_handler.hpp"
#include "profiler/cpu_profiler.hpp"
#include "core/cameras/camera.hpp"
#include "core/scene/materials/medium.hpp"
#include "core/math/rng.hpp"
#include "core/scene/accel_structs/intersection.hpp"
#include <random>

namespace mufflon::renderer {

using PtPathVertex = PathVertex<u8, 4>;

CpuPathTracer::CpuPathTracer()
{
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
	if(m_reset) {
		// TODO: reset output buffer
		// Reacquire scene descriptor (partially?)
		m_sceneDesc = m_currentScene->get_descriptor<Device::CPU>({}, {}, {}, outputBuffer.get_resolution());
	}
	m_reset = false;

	// TODO: call sample in a parallel way for each output pixel
	// TODO: better pixel order?
	// TODO: different scheduling?
//#pragma omp parallel for
	for(int pixel = 0; pixel < outputBuffer.get_num_pixels(); ++pixel) {
		this->sample(Pixel{ pixel % outputBuffer.get_width(), pixel / outputBuffer.get_width() }, buffer, m_sceneDesc);
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
	int s = PtPathVertex::create_camera(vertex, vertex, scene.camera.get(), coord, m_rngs[pixel].next());
	mAssertMsg(s < 256, "vertexBuffer overflow.");


	int pathLen = 0;
	do {
		if(false)
		if(pathLen+1 >= m_params.maxPathLength) {
			// Call NEE member function for the camera start/recursive vertices
			// TODO: test/parametrize mulievent estimation (more indices in connect) and different guides.
			u64 neeSeed = m_rngs[pixel].next();
			math::RndSet2 neeRnd = m_rngs[pixel].next();
			auto nee = connect(scene.lightTree, 0, 1, neeSeed,
				vertex->get_position(), m_currentScene->get_bounding_box(),
				neeRnd, scene::lights::guide_flux);
			// TODO: set startInsPrimId with a proper value.
			bool anyhit = mufflon::scene::accel_struct::any_intersection_scene_lbvh<Device::CPU>(
				scene, { vertex->get_position() , ei::normalize(nee.direction) }, { -1, -1 },
				nee.dist); 
			if(!anyhit) {
				auto value = vertex->evaluate(nee.direction, scene.media);
				mAssert(!isnan(value.value.x) && !isnan(value.value.y) && !isnan(value.value.z));
				AreaPdf hitPdf = value.pdfF.to_area_pdf(nee.cosOut, nee.distSq);
				float mis = 1.0f / (1.0f + hitPdf / nee.creationPdf);
				mAssert(!isnan(mis));
				outputBuffer.contribute(coord, throughput, { Spectrum{nee.intensity}, 1.0f },
					value.cosOut, value.value * mis);
			}
		}

		// Walk
		scene::Point lastPosition = vertex->get_position();
		math::RndSet2_1 rnd { m_rngs[pixel].next(), m_rngs[pixel].next() };
		scene::Direction lastDir;
		if(!walk(scene, *vertex, rnd, 0.0f, false, throughput, vertex, lastDir)) {
			if(throughput.weight != Spectrum{ 0.f }) {
				// Missed scene - sample background
				ei::Vec3 background = scene.lightTree.background.get_radiance(lastDir);
				// TODO: where do we get the normal and stuff from?
				outputBuffer.contribute(coord, throughput, background,
										ei::Vec3{ 0, 0, 0 }, ei::Vec3{ 0, 0, 0 },
										ei::Vec3{ 0, 0, 0 });
			}
			break;
		}
		++pathLen;

		// Evaluate direct hit of area ligths
		if(pathLen >= m_params.maxPathLength) {
			Spectrum emission = vertex->get_emission();
			if(emission != 0.0f) {
				AreaPdf backwardPdf = connect_pdf(scene.lightTree, 0,
												  lastPosition, scene::lights::guide_flux);
				//float mis = 1.0f / (1.0f + backwardPdf / vertex->get_incident_pdf());
				float mis = 1.0f;
				outputBuffer.contribute(coord, throughput, emission, vertex->get_position(),
					vertex->get_normal(), vertex->get_albedo());
			}
		}
	} while(pathLen < m_params.maxPathLength);
}

void CpuPathTracer::init_rngs(int num) {
	m_rngs.resize(num);
	// TODO: incude some global seed into the initialization
	for(int i = 0; i < num; ++i)
		m_rngs[i] = math::Rng(i);
}

void CpuPathTracer::load_scene(scene::SceneHandle scene, const ei::IVec2& resolution) {
	if(scene != m_currentScene) {
		m_currentScene = scene;
		// Make sure the scene is loaded completely for the use on CPU side
		m_currentScene->synchronize<Device::CPU>();
		m_sceneDesc = m_currentScene->get_descriptor<Device::CPU>({}, {}, {}, resolution);
		m_reset = true;
	}
}

} // namespace mufflon::renderer