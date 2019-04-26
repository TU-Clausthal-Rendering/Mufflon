#include "cpu_silhouette_bpm.hpp"
#include "silhouette_importance_gathering_bpm.hpp"
#include "profiler/cpu_profiler.hpp"
#include "util/parallel.hpp"
#include "core/renderer/output_handler.hpp"
#include "core/renderer/path_util.hpp"
#include "core/renderer/random_walk.hpp"
#include "core/renderer/pt/pt_common.hpp"
#include "core/scene/descriptors.hpp"
#include "core/scene/scene.hpp"
#include "core/scene/world_container.hpp"
#include "core/scene/lights/lights.hpp"
#include "core/scene/lights/light_sampling.hpp"
#include "core/scene/lights/light_tree_sampling.hpp"
#include "core/scene/world_container.hpp"
#include <cstdio>
#include <random>
#include <queue>

namespace mufflon::renderer::decimaters::silhouette {

using namespace bpm;

namespace {

inline float get_luminance(const ei::Vec3& vec) {
	constexpr ei::Vec3 LUM_WEIGHT{ 0.212671f, 0.715160f, 0.072169f };
	return ei::dot(LUM_WEIGHT, vec);
}

} // namespace

CpuShadowSilhouettesBPM::CpuShadowSilhouettesBPM()
{
	// TODO: init one RNG per thread?
	std::random_device rndDev;
	m_rngs.emplace_back(static_cast<u32>(rndDev()));
}

void CpuShadowSilhouettesBPM::on_scene_load() {
	if(m_currentDecimationIteration != 0u) {
		// At least activate the created LoDs
		for(auto& obj : m_currentScene->get_objects()) {
			const u32 newLodLevel = static_cast<u32>(obj.first->get_lod_slot_count() - 1u);
			// TODO: this reeeeally breaks instancing
			scene::WorldContainer::instance().get_current_scenario()->set_custom_lod(obj.first, newLodLevel);
		}
	}
}

void CpuShadowSilhouettesBPM::on_scene_unload() {
	m_decimaters.clear();
	m_currentDecimationIteration = 0u;
}

void CpuShadowSilhouettesBPM::post_iteration(OutputHandler& outputBuffer) {
	if((int)m_currentDecimationIteration == m_params.decimationIterations) {
		// Finalize the decimation process
		logInfo("Finished decimation process");
		++m_currentDecimationIteration;
		m_reset = true;

		// Fix up all other scenarios too (TODO: not a wise choice to do so indiscriminately...)
		for(std::size_t i = 0u; i < scene::WorldContainer::instance().get_scenario_count(); ++i) {
			auto handle = scene::WorldContainer::instance().get_scenario(i);
			for(const auto& obj : m_currentScene->get_objects()) {
				const auto newLodLevel = obj.first->get_lod_slot_count() - 1u;
				handle->set_custom_lod(obj.first, static_cast<u32>(newLodLevel));
			}
		}
	} else if((int)m_currentDecimationIteration < m_params.decimationIterations) {
		logInfo("Performing decimation iteration...");
		const auto processTime = CpuProfileState::get_process_time();
		const auto cycles = CpuProfileState::get_cpu_cycle();
		auto scope = Profiler::instance().start<CpuProfileState>("Silhouette decimation");
#pragma PARALLEL_FOR
		for(i32 i = 0; i < static_cast<i32>(m_decimaters.size()); ++i) {
			m_decimaters[i]->iterate(static_cast<std::size_t>(m_params.threshold), (float)(1.0 - m_remainingVertexFactor[i]));
		}
		logInfo("Finished decimation iteration (", std::chrono::duration_cast<std::chrono::milliseconds>(CpuProfileState::get_process_time() - processTime).count(),
				"ms, ", (CpuProfileState::get_cpu_cycle() - cycles) / 1'000'000, " MCycles)");

		m_currentScene->clear_accel_structure();
		m_reset = true;
		++m_currentDecimationIteration;
	}
	RendererBase<Device::CPU>::post_iteration(outputBuffer);
}

void CpuShadowSilhouettesBPM::on_reset() {
	init_rngs(m_outputBuffer.get_num_pixels());
	m_photonMapManager.resize(m_outputBuffer.get_num_pixels() * (m_params.maxPathLength - 1));
	m_photonMap = m_photonMapManager.acquire<Device::CPU>();
}

void CpuShadowSilhouettesBPM::pre_descriptor_requery() {
	// Initialize the decimaters
	// TODO: how to deal with instancing
	if(m_currentDecimationIteration == 0u)
		this->initialize_decimaters();
}

void CpuShadowSilhouettesBPM::iterate() {
	if((int)m_currentDecimationIteration < m_params.decimationIterations) {
		logInfo("Starting decimation iteration (", m_currentDecimationIteration + 1, " of ", m_params.decimationIterations, ")...");

		const auto processTime = CpuProfileState::get_process_time();
		const auto cycles = CpuProfileState::get_cpu_cycle();
		gather_importance();

		if(m_decimaters.size() == 0u)
			return;

		// We need to update the importance density
		this->update_reduction_factors();
		compute_max_importance();

		logInfo("Finished importance gathering (",
					std::chrono::duration_cast<std::chrono::milliseconds>(processTime).count(),
					"ms, ", cycles / 1'000'000, " MCycles)");
	} else {
		if((int)m_currentDecimationIteration == m_params.decimationIterations) {
			for(auto& decimater : m_decimaters)
				decimater->copy_back_normalized_importance();
			compute_max_importance();
		}
		display_importance();
	}
}

void CpuShadowSilhouettesBPM::gather_importance() {
	// Re-upload the (possibly resized) importance buffers
	for(std::size_t i = 0u; i < m_decimaters.size(); ++i)
		m_importances[i] = m_decimaters[i]->start_iteration();

	for(int i = 0; i < m_params.importanceIterations; ++i) {
		float currentMergeRadius = m_params.mergeRadius * m_sceneDesc.diagSize;
		if(m_params.progressive)
			currentMergeRadius *= powf(float(i + 1), -1.0f / 6.0f);
		m_photonMap.clear(currentMergeRadius * 2.0001f);

		// First pass: Create one photon path per view path
		u64 photonSeed = m_rngs[0].next();
		int numPhotons = m_outputBuffer.get_num_pixels();
#pragma PARALLEL_FOR
		for(int p = 0; p < numPhotons; ++p) {
			silhouette::trace_importance_photon(m_sceneDesc, m_photonMap, m_params, p, numPhotons,
												photonSeed, currentMergeRadius, m_rngs[p]);
		}

		// Second pass: trace view paths and merge
#pragma PARALLEL_FOR
		for(int pixel = 0; pixel < m_outputBuffer.get_num_pixels(); ++pixel) {
			silhouette::sample_importance(m_sceneDesc, m_photonMap, m_params,
										  Pixel{ pixel % m_outputBuffer.get_width(), pixel / m_outputBuffer.get_width() },
										  m_rngs[pixel], numPhotons, currentMergeRadius,
										  m_importances.get(), m_importanceSums.get());
		}
	}
	// TODO: allow for this with proper reset "events"
}

void CpuShadowSilhouettesBPM::display_importance() {
	const u32 NUM_PIXELS = m_outputBuffer.get_num_pixels();
#pragma PARALLEL_FOR
	for(int pixel = 0; pixel < (int)NUM_PIXELS; ++pixel) {
		const Pixel coord{ pixel % m_outputBuffer.get_width(), pixel / m_outputBuffer.get_width() };
		silhouette::sample_vis_importance(m_outputBuffer, m_sceneDesc, coord, m_rngs[pixel],
										  m_importances.get(), m_maxImportance);
	}
}

void CpuShadowSilhouettesBPM::compute_max_importance() {
	m_maxImportance = 0.f;

	// Compute the maximum normalized importance for visualization
	for(i32 i = 0u; i < m_decimaters.size(); ++i)
		m_maxImportance = std::max(m_maxImportance, m_decimaters[i]->get_current_max_importance());
}

void CpuShadowSilhouettesBPM::initialize_decimaters() {
	auto& objects = m_currentScene->get_objects();
	m_decimaters.clear();
	m_decimaters.resize(objects.size());
	auto objIter = objects.begin();

	const auto timeBegin = CpuProfileState::get_process_time();
	m_importanceSums = make_udevptr_array<Device::CPU, DeviceImportanceSums<Device::CPU>, false>(m_decimaters.size());
	m_importances = make_udevptr_array<Device::CPU, Importances<Device::CPU>*, false>(m_decimaters.size());

#pragma PARALLEL_FOR
	for(i32 i = 0; i < static_cast<i32>(objects.size()); ++i) {
		auto objIter = objects.begin();
		for(i32 j = 0; j < i; ++j)
			++objIter;
		auto& obj = *objIter;

		auto& lod = obj.first->get_lod(0u);
		const auto& polygons = lod.template get_geometry<scene::geometry::Polygons>();

		std::size_t collapses = 0u;

		if(polygons.get_vertex_count() >= m_params.threshold && m_params.initialReduction > 0.f) {
			collapses = static_cast<std::size_t>(std::ceil(m_params.initialReduction * polygons.get_vertex_count()));
			logInfo("Reducing LoD 0 of object '", obj.first->get_name(), "' by ", collapses, " vertices");
		}
		const u32 newLodLevel = static_cast<u32>(obj.first->get_lod_slot_count());
		auto& newLod = obj.first->add_lod(newLodLevel, lod);
		m_decimaters[i] = std::make_unique<ImportanceDecimater<Device::CPU>>(lod, newLod, collapses,
																			 m_params.viewWeight, m_params.lightWeight,
																			 m_params.shadowWeight, m_params.shadowSilhouetteWeight);
		m_importanceSums[i].shadowImportance.store(0.f);
		m_importanceSums[i].shadowSilhouetteImportance.store(0.f);
		// TODO: this reeeeally breaks instancing
		scene::WorldContainer::instance().get_current_scenario()->set_custom_lod(obj.first, newLodLevel);
	}

	logInfo("Initial decimation: ", std::chrono::duration_cast<std::chrono::milliseconds>(CpuProfileState::get_process_time() - timeBegin).count(), "ms");
}

void CpuShadowSilhouettesBPM::update_reduction_factors() {
	m_remainingVertexFactor.clear();
	if(m_params.reduction == 0.f) {
		// Do not reduce anything
		for(std::size_t i = 0u; i < m_decimaters.size(); ++i) {
			ImportanceSums sums{ m_importanceSums[i].shadowImportance, m_importanceSums[i].shadowSilhouetteImportance };
			m_decimaters[i]->update_importance_density(sums);
			m_importanceSums[i].shadowImportance.store(0.f);
			m_importanceSums[i].shadowSilhouetteImportance.store(0.f);
			m_remainingVertexFactor.push_back(1.0);
		}
		return;
	}

	double expectedVertexCount = 0.0;
	for(std::size_t i = 0u; i < m_decimaters.size(); ++i) {
		auto& decimater = m_decimaters[i];
		ImportanceSums sums{ m_importanceSums[i].shadowImportance, m_importanceSums[i].shadowSilhouetteImportance };
		m_decimaters[i]->update_importance_density(sums);
		m_importanceSums[i].shadowImportance.store(0.f);
		m_importanceSums[i].shadowSilhouetteImportance.store(0.f);
		if(decimater->get_original_vertex_count() > m_params.threshold) {
			m_remainingVertexFactor.push_back(decimater->get_importance_sum());
			expectedVertexCount += (1.f - m_params.reduction) * decimater->get_original_vertex_count();
		} else {
			m_remainingVertexFactor.push_back(1.0);
			expectedVertexCount += decimater->get_original_vertex_count();
		}
	}

	// Determine the reduction parameters for each mesh
	constexpr u32 MAX_ITERATION_COUNT = 10u;
	for(u32 iteration = 0u; iteration < MAX_ITERATION_COUNT; ++iteration) {
		double vertexCountAfterDecimation = 0.0;
		for(std::size_t i = 0u; i < m_decimaters.size(); ++i)
			vertexCountAfterDecimation += m_remainingVertexFactor[i] * m_decimaters[i]->get_original_vertex_count();
		const double normalizationFactor = expectedVertexCount / vertexCountAfterDecimation;

		bool anyAboveOne = false;

		for(std::size_t i = 0u; i < m_decimaters.size(); ++i) {
			if(m_decimaters[i]->get_original_vertex_count() > m_params.threshold) {
				m_remainingVertexFactor[i] *= normalizationFactor;
				anyAboveOne |= m_remainingVertexFactor[i] > 1.0;
				m_remainingVertexFactor[i] = std::clamp(m_remainingVertexFactor[i], 0.0, 1.0);
			}
		}

		if(!anyAboveOne)
			break;
	}
}

void CpuShadowSilhouettesBPM::init_rngs(int num) {
	m_rngs.resize(num);
	// TODO: incude some global seed into the initialization
	for(int i = 0; i < num; ++i)
		m_rngs[i] = math::Rng(i);
}

} // namespace mufflon::renderer::decimaters::silhouette