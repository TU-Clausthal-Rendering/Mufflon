#include "silhouette_importance_gathering_pt.hpp"
#include "silhouette_importance_gathering_pt_octree.hpp"
#include "cpu_silhouette_pt.hpp"
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

namespace mufflon::renderer::decimaters::silhouette {

using namespace pt;

namespace {

inline float get_luminance(const ei::Vec3& vec) {
	constexpr ei::Vec3 LUM_WEIGHT{ 0.212671f, 0.715160f, 0.072169f };
	return ei::dot(LUM_WEIGHT, vec);
}

} // namespace

CpuShadowSilhouettesPT::CpuShadowSilhouettesPT(mufflon::scene::WorldContainer& world) :
	RendererBase<Device::CPU, pt::SilhouetteTargets>{ world }
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
			m_world.get_current_scenario()->set_custom_lod(obj.first, newLodLevel);
		}
	}

	if(get_reset_event() & ResetEvent::PARAMETER)
		m_currentDecimationIteration = 0u;

	// Initialize the decimaters
	// TODO: how to deal with instancing
	if(m_currentDecimationIteration == 0u) {
		this->initialize_decimaters();
	}
	
	RendererBase<Device::CPU, pt::SilhouetteTargets>::pre_reset();
}

void CpuShadowSilhouettesPT::on_world_clearing() {
	m_decimaters.clear();
	m_currentDecimationIteration = 0u;
}

void CpuShadowSilhouettesPT::post_iteration(IOutputHandler& outputBuffer) {
	if((int)m_currentDecimationIteration == m_params.decimationIterations) {
		// Finalize the decimation process
		logInfo("Finished decimation process");
		++m_currentDecimationIteration;
		this->on_manual_reset();
	} else if((int)m_currentDecimationIteration < m_params.decimationIterations) {
		logInfo("Performing decimation iteration...");
		const auto processTime = CpuProfileState::get_process_time();
		const auto cycles = CpuProfileState::get_cpu_cycle();
		auto scope = Profiler::core().start<CpuProfileState>("Silhouette decimation");
#pragma omp parallel for schedule(dynamic)
		for(i32 i = 0; i < static_cast<i32>(m_decimaters.size()); ++i) {
			if(m_viewOctree)
				m_decimaters[i]->iterate(m_remainingVertices[i], &(*m_viewOctree)[i]);
			else
				m_decimaters[i]->iterate(m_remainingVertices[i], nullptr);
		}
		m_currentScene->clear_accel_structure();
		logInfo("Finished decimation iteration (", std::chrono::duration_cast<std::chrono::milliseconds>(CpuProfileState::get_process_time() - processTime).count(),
				"ms, ", (CpuProfileState::get_cpu_cycle() - cycles) / 1'000'000, " MCycles)");

		this->on_manual_reset();
		++m_currentDecimationIteration;
	}
	RendererBase<Device::CPU, pt::SilhouetteTargets>::post_iteration(outputBuffer);
}

void CpuShadowSilhouettesPT::iterate() {
	if((int)m_currentDecimationIteration < m_params.decimationIterations) {
		logInfo("Starting decimation iteration (", m_currentDecimationIteration + 1, " of ", m_params.decimationIterations, ")...");

		const auto processTime = CpuProfileState::get_process_time();
		const auto cycles = CpuProfileState::get_cpu_cycle();

		// First we need to reset the importance sums (which we may have kept for visualization)
		for(std::size_t i = 0u; i < m_decimaters.size(); ++i) {
			m_importanceSums[i].shadowImportance.store(0.f);
			m_importanceSums[i].shadowSilhouetteImportance.store(0.f);
		}

		m_viewOctree = nullptr;
		m_irradianceOctree = nullptr;
		m_viewGrid = nullptr;
		m_irradianceGrid = nullptr;
		m_irradianceCountGrid = nullptr;

		if(m_params.impDataStruct == PImpDataStruct::Values::OCTREE) {
			m_viewOctree = std::make_unique<data_structs::CountOctreeManager>(static_cast<u32>(m_params.impCapacity), static_cast<u32>(m_decimaters.size()), 8u);
			m_irradianceOctree = std::make_unique<data_structs::CountOctreeManager>(static_cast<u32>(m_params.impCapacity), static_cast<u32>(m_decimaters.size()), 8u);
			for(std::size_t i = 0u; i < m_decimaters.size(); ++i) {
				m_viewOctree->create(m_sceneDesc.aabbs[i]);
				m_irradianceOctree->create(m_sceneDesc.aabbs[i]);
			}
		} else if(m_params.impDataStruct == PImpDataStruct::Values::HASHGRID) {
			m_viewGrid = std::make_unique<data_structs::DmHashGrid<float>>(static_cast<u32>(m_params.impCapacity));
			m_irradianceGrid = std::make_unique<data_structs::DmHashGrid<float>>(static_cast<u32>(m_params.impCapacity));
			m_irradianceCountGrid = std::make_unique<data_structs::DmHashGrid<u32>>(static_cast<u32>(m_params.impCapacity));
			const auto cell_size = 0.00005f * m_sceneDesc.diagSize;
			m_viewGrid->set_cell_size(cell_size);
			m_irradianceGrid->set_cell_size(cell_size);
			m_irradianceCountGrid->set_cell_size(cell_size);
		}

		gather_importance();

		if(m_decimaters.size() == 0u)
			return;

		// We need to update the importance density
		this->update_reduction_factors();
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
			switch(m_params.impDataStruct) {
				case PImpDataStruct::Values::OCTREE:
					silhouette::sample_importance_octree(m_outputBuffer, m_sceneDesc, m_params, coord, m_rngs[pixel],
														 m_importanceSums.get(), *m_viewOctree, *m_irradianceOctree);
					break;
				case PImpDataStruct::Values::HASHGRID:
					silhouette::sample_importance(m_outputBuffer, m_sceneDesc, m_params, coord, m_rngs[pixel],
												  m_importanceSums.get(), *m_viewGrid, *m_irradianceGrid,
												  *m_irradianceCountGrid);
					break;
				case PImpDataStruct::Values::VERTEX:
				default:
					silhouette::sample_importance(m_outputBuffer, m_sceneDesc, m_params, coord, m_rngs[pixel],
												  m_importanceSums.get(), m_importances.get());
					break;
			}
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
		silhouette::sample_vis_importance(m_outputBuffer, m_sceneDesc, coord, m_rngs[pixel],
										  m_importances.get(), m_viewOctree.get(),
										  m_importanceSums.get(), m_maxImportance == 0.f ? 1.f : m_maxImportance);
	}
}

void CpuShadowSilhouettesPT::compute_max_importance() {
	m_maxImportance = 0.f;

	// Compute the maximum normalized importance for visualization
	for(i32 i = 0u; i < m_decimaters.size(); ++i)
		m_maxImportance = std::max(m_maxImportance, m_decimaters[i]->get_current_max_importance());
}

void CpuShadowSilhouettesPT::initialize_decimaters() {
	auto& objects = m_currentScene->get_objects();
	const auto& instances = m_currentScene->get_instances();
	m_decimaters.clear();
	m_decimaters.resize(objects.size());

	const auto timeBegin = CpuProfileState::get_process_time();
	m_importanceSums = make_udevptr_array<Device::CPU, DeviceImportanceSums<Device::CPU>, false>(m_decimaters.size());
	m_importances = make_udevptr_array<Device::CPU, Importances<Device::CPU>*, false>(m_decimaters.size());

#pragma PARALLEL_FOR
	for(i32 i = 0; i < static_cast<i32>(objects.size()); ++i) {
		auto objIter = objects.begin();
		for(i32 j = 0; j < i; ++j)
			++objIter;
		auto& obj = *objIter;

		const auto& scenario = *m_world.get_current_scenario();
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
		auto& lod = obj.first->get_or_fetch_original_lod(m_world, lowestLevel);
		auto& newLod = obj.first->get_reduced_lod(lowestLevel);
		const auto& polygons = newLod.template get_geometry<scene::geometry::Polygons>();

		std::size_t collapses = 0u;

		if(polygons.get_vertex_count() >= m_params.threshold && m_params.initialReduction > 0.f) {
			collapses = static_cast<std::size_t>(std::ceil(m_params.initialReduction * polygons.get_vertex_count()));
			logInfo("Reducing LoD 0 of object '", obj.first->get_name(), "' by ", collapses, " vertices");
		}
		m_decimaters[i] = std::make_unique<ImportanceDecimater<Device::CPU>>(obj.first->get_name(), lod, newLod,
																			 m_params.gridRes,
																			 m_params.viewWeight, m_params.lightWeight,
																			 m_params.shadowWeight, m_params.shadowSilhouetteWeight);
		m_importanceSums[i].shadowImportance.store(0.f);
		m_importanceSums[i].shadowSilhouetteImportance.store(0.f);
	}

	m_currentScene->clear_accel_structure();
	logInfo("Initial decimation: ", std::chrono::duration_cast<std::chrono::milliseconds>(CpuProfileState::get_process_time() - timeBegin).count(), "ms");
}

void CpuShadowSilhouettesPT::update_reduction_factors() {
	m_remainingVertices.clear();
	if(m_params.reduction == 0.f) {
		// Do not reduce anything
		for(std::size_t i = 0u; i < m_decimaters.size(); ++i) {
			ImportanceSums sums{ m_importanceSums[i].shadowImportance, m_importanceSums[i].shadowSilhouetteImportance };
			switch(m_params.impDataStruct) {
				case PImpDataStruct::Values::OCTREE:
					m_decimaters[i]->update_importance_density(sums, (*m_viewOctree)[i], (*m_irradianceOctree)[i]);
					break;
				case PImpDataStruct::Values::HASHGRID:
					m_decimaters[i]->update_importance_density(sums, *m_viewGrid, *m_irradianceGrid, *m_irradianceCountGrid);
					break;
				case PImpDataStruct::Values::VERTEX:
				default:
					m_decimaters[i]->update_importance_density(sums, m_params.impSumStrat == PImpSumStrat::Values::CURV_AREA);
					break;
			}
			m_remainingVertices.push_back(m_decimaters[i]->get_original_vertex_count());
		}
		return;
	}

	double importanceSum = 0.0;
	std::size_t reducibleVertices = 0u;
	std::size_t nonReducibleVertices = 0u;
	for(std::size_t i = 0u; i < m_decimaters.size(); ++i) {
		auto& decimater = m_decimaters[i];
		ImportanceSums sums{ m_importanceSums[i].shadowImportance, m_importanceSums[i].shadowSilhouetteImportance };
		switch(m_params.impDataStruct) {
			case PImpDataStruct::Values::OCTREE:
				m_decimaters[i]->update_importance_density(sums, (*m_viewOctree)[i], (*m_irradianceOctree)[i]);
				break;
			case PImpDataStruct::Values::HASHGRID:
				m_decimaters[i]->update_importance_density(sums, *m_viewGrid, *m_irradianceGrid, *m_irradianceCountGrid);
				break;
			case PImpDataStruct::Values::VERTEX:
			default:
				m_decimaters[i]->update_importance_density(sums, m_params.impSumStrat == PImpSumStrat::Values::CURV_AREA);
				break;
		}
		if(decimater->get_original_vertex_count() > m_params.threshold) {
			importanceSum += decimater->get_importance_sum();
			reducibleVertices += decimater->get_original_vertex_count();
		} else {
			nonReducibleVertices += decimater->get_original_vertex_count();
		}
	}
	const auto totalVertexCount = reducibleVertices + nonReducibleVertices;
	const std::size_t targetVertexCount = static_cast<std::size_t>((1.f - m_params.reduction) * totalVertexCount);
	std::size_t reducedVertexPool = targetVertexCount - nonReducibleVertices;

	for(auto& decimater : m_decimaters) {
		if(decimater->get_original_vertex_count() > m_params.threshold) {
			const auto targetVertexCount = std::min(decimater->get_original_vertex_count(),
													static_cast<std::size_t>(decimater->get_importance_sum()
																			 * static_cast<double>(reducedVertexPool) / importanceSum));
			reducedVertexPool -= targetVertexCount;
			importanceSum -= decimater->get_importance_sum();
			m_remainingVertices.push_back(targetVertexCount);
		} else {
			m_remainingVertices.push_back(decimater->get_original_vertex_count());
		}
	}
}

void CpuShadowSilhouettesPT::init_rngs(int num) {
	m_rngs.resize(num);
	// TODO: incude some global seed into the initialization
	for(int i = 0; i < num; ++i)
		m_rngs[i] = math::Rng(i);
}

} // namespace mufflon::renderer::decimaters::silhouette
