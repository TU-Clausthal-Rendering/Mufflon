#include "core/renderer/decimaters/silhouette/pt/silhouette_importance_gathering_pt.hpp"
#include "cpu_animation_pt.hpp"
#include "profiler/cpu_profiler.hpp"
#include "util/parallel.hpp"
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
#include <OpenMesh/Tools/Decimater/ModQuadricT.hh>
#include <cstdio>
#include <random>
#include <queue>

namespace mufflon::renderer::decimaters::animation {

using namespace pt;

namespace {

inline float get_luminance(const ei::Vec3& vec) {
	constexpr ei::Vec3 LUM_WEIGHT{ 0.212671f, 0.715160f, 0.072169f };
	return ei::dot(LUM_WEIGHT, vec);
}

} // namespace

CpuShadowSilhouettesPT::CpuShadowSilhouettesPT()
{
	std::random_device rndDev;
	m_rngs.emplace_back(static_cast<u32>(rndDev()));
}

void CpuShadowSilhouettesPT::pre_reset() {
	if((get_reset_event() & ResetEvent::CAMERA) != ResetEvent::NONE || get_reset_event().resolution_changed())
		init_rngs(m_outputBuffer.get_num_pixels());

	if((get_reset_event() & ResetEvent::SCENARIO) != ResetEvent::NONE && m_currentDecimationIteration != 0u) {
		// At least activate the created LoDs
		for(auto& obj : m_currentScene->get_objects()) {
			const u32 newLodLevel = static_cast<u32>(obj.first->get_lod_slot_count() - 1u);
			// TODO: this reeeeally breaks instancing
			scene::WorldContainer::instance().get_current_scenario()->set_custom_lod(obj.first, newLodLevel);
		}
	}

	if(get_reset_event() & ResetEvent::PARAMETER)
		m_currentDecimationIteration = 0u;

	// Initialize the decimaters
	// TODO: how to deal with instancing
	if(m_currentDecimationIteration == 0u) {
		this->initialize_decimaters();
	}
	
	RendererBase<Device::CPU, silhouette::pt::SilhouetteTargets>::pre_reset();
}

void CpuShadowSilhouettesPT::on_world_clearing() {
	m_decimaters.clear();
	m_currentDecimationIteration = 0u;
}

void CpuShadowSilhouettesPT::on_animation_frame_changed(const u32 from, const u32 to) {
	m_currentDecimationIteration = std::numeric_limits<u32>::max();
}

void CpuShadowSilhouettesPT::post_iteration(IOutputHandler& outputBuffer) {
	if((int)m_currentDecimationIteration == m_params.decimationIterations) {
		// Finalize the decimation process
		logInfo("Finished decimation process");
		++m_currentDecimationIteration;
		this->on_manual_reset();
	} else if((int)m_currentDecimationIteration < m_params.decimationIterations) {
		logInfo("Performing decimation iteration...");
		// Update the reduction factors
		this->update_reduction_factors();

		const auto processTime = CpuProfileState::get_process_time();
		const auto cycles = CpuProfileState::get_cpu_cycle();
		auto scope = Profiler::core().start<CpuProfileState>("Silhouette decimation");
#pragma PARALLEL_FOR
		for(i32 i = 0; i < static_cast<i32>(m_decimaters.size()); ++i) {
			m_decimaters[i]->upload_importance(m_params.impWeightMethod);
			m_decimaters[i]->iterate(static_cast<std::size_t>(m_params.threshold), (float)(1.0 - m_remainingVertexFactor[i]));
		}
		m_currentScene->clear_accel_structure();
		logInfo("Finished decimation iteration (", std::chrono::duration_cast<std::chrono::milliseconds>(CpuProfileState::get_process_time() - processTime).count(),
				"ms, ", (CpuProfileState::get_cpu_cycle() - cycles) / 1'000'000, " MCycles)");

		this->on_manual_reset();
		++m_currentDecimationIteration;
	}
	RendererBase<Device::CPU, silhouette::pt::SilhouetteTargets>::post_iteration(outputBuffer);
}

void CpuShadowSilhouettesPT::iterate() {
	if((int)m_currentDecimationIteration < m_params.decimationIterations) {
		logInfo("Starting decimation iteration (", m_currentDecimationIteration + 1, " of ", m_params.decimationIterations, ")...");

		const auto processTime = CpuProfileState::get_process_time();
		const auto cycles = CpuProfileState::get_cpu_cycle();

		const auto currAnimationFrame = scene::WorldContainer::instance().get_frame_current();
		const auto startAnimationFrame = scene::WorldContainer::instance().get_frame_start();
		const auto endAnimationFrame = scene::WorldContainer::instance().get_frame_end();
		//const auto frameIndex = std::min<u32>(m_params.slidingWindowSize, currAnimationFrame - startAnimationFrame);
		//const auto windowSize = std::min<u32>(std::min(endAnimationFrame - currAnimationFrame, frameIndex + 1u), m_params.slidingWindowSize);
		
		// TODO: put this in a more conventient place / discard it from profiling
		// Gather all the importance we need
		const auto upperFrameLimit = std::min(endAnimationFrame, currAnimationFrame + m_params.slidingWindowHalfWidth) - startAnimationFrame;
		// Loop until we gathered importance for all necessary frames (half-width + 1 at the beginning, 1 normally, none towards the end)
		while(m_perFrameData.size() <= upperFrameLimit) {
			logInfo("Acquiring importance for frame ", m_perFrameData.size());
			// TODO: free data that is no longer needed
			m_perFrameData.push_back({
				make_udevptr_array<Device::CPU, silhouette::pt::DeviceImportanceSums<Device::CPU>, false>(m_decimaters.size()),
				0.f
			});
			// Set the current importance sums to zero
			for(std::size_t i = 0u; i < m_decimaters.size(); ++i) {
				m_perFrameData.back().importanceSums[i].shadowImportance.store(0.f);
				m_perFrameData.back().importanceSums[i].shadowSilhouetteImportance.store(0.f);
			}

			// Do the usual importance gathering
			gather_importance();

			// Update the importance sums
			for(std::size_t i = 0u; i < m_decimaters.size(); ++i) {
				silhouette::pt::ImportanceSums sums{ m_perFrameData.back().importanceSums[i].shadowImportance,
					m_perFrameData.back().importanceSums[i].shadowSilhouetteImportance };
				m_decimaters[i]->update_importance_density(sums);
			}
		}

		if(m_decimaters.size() == 0u)
			return;
		//compute_max_importance();

		logInfo("Finished importance gathering (",
					std::chrono::duration_cast<std::chrono::milliseconds>(processTime).count(),
					"ms, ", cycles / 1'000'000, " MCycles)");

	} else {
		if((int)m_currentDecimationIteration == m_params.decimationIterations && m_params.decimationIterations > 0) {
			if(m_params.reduction == 0) {
				for(auto& decimater : m_decimaters)
					decimater->copy_back_normalized_importance();
				compute_max_importance();
			}
		}
		if(m_params.reduction == 0 && m_params.decimationIterations > 0)
			display_importance();
	}
}

void CpuShadowSilhouettesPT::gather_importance() {
	if(m_params.maxPathLength >= 16u) {
		logError("[CpuShadowSilhouettesPT::gather_importance] Max. path length too long (max. 15 permitted)");
		return;
	}

	// Re-upload the (possibly resized) importance buffers
	for(std::size_t i = 0u; i < m_decimaters.size(); ++i)
		m_importances[i] = m_decimaters[i]->start_iteration();

	const u32 NUM_PIXELS = m_outputBuffer.get_num_pixels();
	for(int iter = 0; iter < m_params.importanceIterations; ++iter) {
#pragma PARALLEL_FOR
		for(int pixel = 0; pixel < (int)NUM_PIXELS; ++pixel) {
			const Pixel coord{ pixel % m_outputBuffer.get_width(), pixel / m_outputBuffer.get_width() };
			scene::PrimitiveHandle shadowPrim;
			silhouette::pt::sample_importance(m_outputBuffer, m_sceneDesc, reinterpret_cast<silhouette::pt::SilhouetteParameters&>(m_params),
											  coord, m_rngs[pixel], m_importances.get(),
											  m_perFrameData.back().importanceSums.get());
		}
		logPedantic("Finished importance iteration (", iter + 1, " of ", m_params.importanceIterations, ")");
	}
	// TODO: allow for this with proper reset "events"
}

void CpuShadowSilhouettesPT::display_importance() {
	const u32 NUM_PIXELS = m_outputBuffer.get_num_pixels();
#pragma PARALLEL_FOR
	for(int pixel = 0; pixel < (int)NUM_PIXELS; ++pixel) {
		const Pixel coord{ pixel % m_outputBuffer.get_width(), pixel / m_outputBuffer.get_width() };
		silhouette::pt::sample_vis_importance(m_outputBuffer, m_sceneDesc, coord, m_rngs[pixel],
											  m_importances.get(), m_perFrameData.back().importanceSums.get(),
											  m_perFrameData.back().maxImportance == 0.f ? 1.f : m_perFrameData.back().maxImportance);
	}
}

void CpuShadowSilhouettesPT::compute_max_importance() {
	m_perFrameData.back().maxImportance = 0.f;

	// Compute the maximum normalized importance for visualization
	for(i32 i = 0u; i < m_decimaters.size(); ++i)
		m_perFrameData.back().maxImportance = std::max(m_perFrameData.back().maxImportance,
													   m_decimaters[i]->get_current_max_importance());
}

void CpuShadowSilhouettesPT::initialize_decimaters() {
	auto& objects = m_currentScene->get_objects();
	const auto& instances = m_currentScene->get_instances();
	m_decimaters.clear();
	m_decimaters.resize(objects.size());

	const auto timeBegin = CpuProfileState::get_process_time();

#pragma PARALLEL_FOR
	for(i32 i = 0; i < static_cast<i32>(objects.size()); ++i) {
		auto objIter = objects.begin();
		for(i32 j = 0; j < i; ++j)
			++objIter;
		auto& obj = *objIter;

		const auto& scenario = *scene::WorldContainer::instance().get_current_scenario();
		// Find the highest-res LoD referenced by an object's instances
		u32 lowestLevel = scenario.get_custom_lod(obj.first);
		for(u32 j = 0u; j < obj.second.count; ++j) {
			scene::ConstInstanceHandle instance = instances[obj.second.offset + j];
			if(const auto level = scenario.get_effective_lod(instance); level < lowestLevel)
				lowestLevel = level;
		}

		// TODO: this only works if instances don't specify LoD levels
		if(!obj.first->has_reduced_lod_available(lowestLevel))
			obj.first->add_reduced_lod(lowestLevel);
		auto& lod = obj.first->get_or_fetch_original_lod(lowestLevel);
		auto& newLod = obj.first->get_reduced_lod(lowestLevel);
		const auto& polygons = newLod.template get_geometry<scene::geometry::Polygons>();

		std::size_t collapses = 0u;

		if(polygons.get_vertex_count() >= m_params.threshold && m_params.initialReduction > 0.f) {
			collapses = static_cast<std::size_t>(std::ceil(m_params.initialReduction * polygons.get_vertex_count()));
			logInfo("Reducing LoD 0 of object '", obj.first->get_name(), "' by ", collapses, " vertices");
		}
		m_decimaters[i] = std::make_unique<pt::ImportanceDecimater<Device::CPU>>(obj.first->get_name(), lod, newLod, collapses,
																				 1u + 2u * m_params.slidingWindowHalfWidth,
																				 m_params.viewWeight, m_params.lightWeight,
																				 m_params.shadowWeight, m_params.shadowSilhouetteWeight);
	}
	m_importances = make_udevptr_array<Device::CPU, silhouette::pt::Importances<Device::CPU>*, false>(m_decimaters.size());

	m_currentScene->clear_accel_structure();
	logInfo("Initial decimation: ", std::chrono::duration_cast<std::chrono::milliseconds>(CpuProfileState::get_process_time() - timeBegin).count(), "ms");
}

void CpuShadowSilhouettesPT::update_reduction_factors() {
	// To compute the reduction factors for each mesh, we take the total importance
	// per mesh and assign the factors proportionally to them.
	// Bringing in animations, we have different options:
	// We can determine the reduction factors for every frame from the importance
	// sum of every frame, we can take averages, or any other kind of weighting
	// function really

	m_remainingVertexFactor.clear();
	if(m_params.reduction == 0.f) {
		// Do not reduce anything
		for(std::size_t i = 0u; i < m_decimaters.size(); ++i)
			m_remainingVertexFactor.push_back(1.0);
		return;
	}

	// Compute the total expected vertex count over all meshes
	const auto currAnimationFrame = scene::WorldContainer::instance().get_frame_current();
	const auto startAnimationFrame = scene::WorldContainer::instance().get_frame_start();
	const auto endAnimationFrame = scene::WorldContainer::instance().get_frame_end();
	const auto start = std::max(startAnimationFrame, std::max<u32>(currAnimationFrame, m_params.slidingWindowHalfWidth) - m_params.slidingWindowHalfWidth);
	const auto end = std::min(currAnimationFrame + m_params.slidingWindowHalfWidth, endAnimationFrame);
	double expectedVertexCount = 0.0;

	m_remainingVertexFactor.resize(m_decimaters.size(), 0.f);
	switch(m_params.vertexDistMethod) {
		case PVertexDistMethod::Values::AVERAGE: {
			for(u32 frame = start; frame <= end; ++frame) {
				for(std::size_t i = 0u; i < m_decimaters.size(); ++i) {
					auto& decimater = m_decimaters[i];
					if(decimater->get_original_vertex_count() > m_params.threshold) {
						m_remainingVertexFactor[i] += (decimater->get_importance_sum(frame - start));
						expectedVertexCount += (1.f - m_params.reduction) * decimater->get_original_vertex_count();
					} else {
						m_remainingVertexFactor[i] += 1.f;
						expectedVertexCount += decimater->get_original_vertex_count();
					}
				}
			}
			// Compute average per frame
			for(std::size_t i = 0u; i < m_decimaters.size(); ++i)
				m_remainingVertexFactor[i] /= static_cast<float>(end - start + 1u);

			expectedVertexCount /= static_cast<float>(end - start + 1u);
		}	break;
		case PVertexDistMethod::Values::MAX: {
			for(u32 frame = start; frame <= end; ++frame) {
				for(std::size_t i = 0u; i < m_decimaters.size(); ++i) {
					auto& decimater = m_decimaters[i];
					if(decimater->get_original_vertex_count() > m_params.threshold) {
						m_remainingVertexFactor[i] = std::max(m_remainingVertexFactor[i],
							(decimater->get_importance_sum(frame - start)));
						expectedVertexCount += (1.f - m_params.reduction) * decimater->get_original_vertex_count();
					} else {
						m_remainingVertexFactor[i] = 1.f;
						expectedVertexCount += decimater->get_original_vertex_count();
					}
				}
			}

			expectedVertexCount /= static_cast<float>(end - start + 1u);
		}	break;
		default: throw std::runtime_error("Invalid vertex budget method");
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

void CpuShadowSilhouettesPT::init_rngs(int num) {
	m_rngs.resize(num);
	// TODO: incude some global seed into the initialization
	for(int i = 0; i < num; ++i)
		m_rngs[i] = math::Rng(i);
}

} // namespace mufflon::renderer::decimaters::animation