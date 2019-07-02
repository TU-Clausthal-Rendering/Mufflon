#include "gpu_silhouette_pt.hpp"
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
#include <OpenMesh/Tools/Decimater/ModQuadricT.hh>
#include <cstdio>
#include <random>
#include <queue>

namespace mufflon::renderer::decimaters::silhouette {

namespace pt::gpusil_details {

cudaError_t call_importance_kernel(const dim3& gridDims, const dim3& blockDims,
								   renderer::RenderBuffer<Device::CUDA>&& outputBuffer,
								   scene::SceneDescriptor<Device::CUDA>* scene,
								   const u32* seeds, const SilhouetteParameters& params,
								   Importances<Device::CUDA>** importances,
								   DeviceImportanceSums<Device::CUDA>* sums);

cudaError_t call_impvis_kernel(const dim3& gridDims, const dim3& blockDims,
							   renderer::RenderBuffer<Device::CUDA>&& outputBuffer,
							   scene::SceneDescriptor<Device::CUDA>* scene,
							   const u32* seeds, Importances<Device::CUDA>** importances,
							   const float maxImportance);

} // namespace pt::gpusil_details

using namespace pt;

GpuShadowSilhouettesPT::GpuShadowSilhouettesPT() :
	m_params{}
	//m_rng{ static_cast<u32>(std::random_device{}()) }
{}

void GpuShadowSilhouettesPT::pre_reset() {
	if(get_reset_event().resolution_changed()) {
		m_seeds = std::make_unique<u32[]>(m_outputBuffer.get_num_pixels());
		m_seedsPtr = make_udevptr_array<Device::CUDA, u32>(m_outputBuffer.get_num_pixels());
	}

	if((get_reset_event() & ResetEvent::SCENARIO) != ResetEvent::NONE && m_currentDecimationIteration != 0u) {
		// At least activate the created LoDs
		for(auto& obj : m_currentScene->get_objects()) {
			const u32 newLodLevel = static_cast<u32>(obj.first->get_lod_slot_count() - 1u);
			// TODO: this reeeeally breaks instancing
			scene::WorldContainer::instance().get_current_scenario()->set_custom_lod(obj.first, newLodLevel);
		}
	}

	// Initialize the decimaters
	// TODO: how to deal with instancing
	if(m_currentDecimationIteration == 0u)
		this->initialize_decimaters();

	RendererBase<Device::CUDA>::pre_reset();
}

void GpuShadowSilhouettesPT::on_world_clearing() {
	m_decimaters.clear();
	m_currentDecimationIteration = 0u;
}

void GpuShadowSilhouettesPT::post_iteration(OutputHandler& outputBuffer) {
	if((int)m_currentDecimationIteration == m_params.decimationIterations) {
		// Finalize the decimation process
		logInfo("Finished decimation process");
		++m_currentDecimationIteration;
		this->on_manual_reset();

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
		this->on_manual_reset();
		++m_currentDecimationIteration;
	}
	RendererBase<Device::CUDA>::post_iteration(outputBuffer);
}

void GpuShadowSilhouettesPT::iterate() {
	if((int)m_currentDecimationIteration < m_params.decimationIterations) {
		logInfo("Starting decimation iteration (", m_currentDecimationIteration + 1, " of ", m_params.decimationIterations, ")...");

		const auto processTime = CpuProfileState::get_process_time();
		const auto cycles = CpuProfileState::get_cpu_cycle();
		gather_importance();

		if(m_decimaters.size() == 0u)
			return;

		// We need to update the importance density
		this->update_reduction_factors();

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

void GpuShadowSilhouettesPT::gather_importance() {
	// Re-upload the (possibly resized) importance buffers
	std::unique_ptr<Importances<Device::CUDA>*[]> importancePtrs = std::make_unique<Importances<Device::CUDA>*[]>(m_decimaters.size());
	for(std::size_t i = 0u; i < m_decimaters.size(); ++i)
		importancePtrs[i] = m_decimaters[i]->start_iteration();
	copy(m_importances.get(), importancePtrs.get(), sizeof(ArrayDevHandle_t<Device::CUDA, Importances<Device::CUDA>>) * m_decimaters.size());

	for(int i = 0; i < m_outputBuffer.get_num_pixels(); ++i)
		m_seeds[i] = static_cast<u32>(m_rng.next());
	copy(m_seedsPtr.get(), m_seeds.get(), sizeof(u32) * m_outputBuffer.get_num_pixels());

	dim3 blockDims{ 16u, 16u, 1u };
	dim3 gridDims{
		1u + static_cast<u32>(m_outputBuffer.get_width() - 1) / blockDims.x,
		1u + static_cast<u32>(m_outputBuffer.get_height() - 1) / blockDims.y,
		static_cast<u32>(m_params.importanceIterations)
	};

	cuda::check_error(gpusil_details::call_importance_kernel(gridDims, blockDims, std::move(m_outputBuffer),
															 m_sceneDesc.get(), m_seedsPtr.get(), m_params,
															 m_importances.get(), m_importanceSums.get()));
}

void GpuShadowSilhouettesPT::display_importance() {
	for(int i = 0; i < m_outputBuffer.get_num_pixels(); ++i)
		m_seeds[i] = static_cast<u32>(m_rng.next());
	copy(m_seedsPtr.get(), m_seeds.get(), sizeof(u32) * m_outputBuffer.get_num_pixels());

	dim3 blockDims{ 16u, 16u, 1u };
	dim3 gridDims{
		1u + static_cast<u32>(m_outputBuffer.get_width() - 1) / blockDims.x,
		1u + static_cast<u32>(m_outputBuffer.get_height() - 1) / blockDims.y,
		1u
	};

	cuda::check_error(gpusil_details::call_impvis_kernel(gridDims, blockDims, std::move(m_outputBuffer),
														 m_sceneDesc.get(), m_seedsPtr.get(), m_importances.get(),
														 m_maxImportance));
}

void GpuShadowSilhouettesPT::compute_max_importance() {
	m_maxImportance = 0.f;

	// Compute the maximum normalized importance for visualization
//#pragma omp parallel for reduction(max:m_maxImportance)
	for(i32 i = 0u; i < m_decimaters.size(); ++i)
		m_maxImportance = std::max(m_maxImportance, m_decimaters[i]->get_current_max_importance());
}

void GpuShadowSilhouettesPT::initialize_decimaters() {
	auto& objects = m_currentScene->get_objects();
	m_decimaters.clear();
	m_decimaters.resize(objects.size());
	auto objIter = objects.begin();

	const auto timeBegin = CpuProfileState::get_process_time();

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
		m_decimaters[i] = std::make_unique<ImportanceDecimater<Device::CUDA>>(lod, newLod, collapses,
																			  m_params.viewWeight, m_params.lightWeight,
																			  m_params.shadowWeight, m_params.shadowSilhouetteWeight);
		// TODO: this reeeeally breaks instancing
		scene::WorldContainer::instance().get_current_scenario()->set_custom_lod(obj.first, newLodLevel);
	}

	m_importances = make_udevptr_array<Device::CUDA, Importances<Device::CUDA>*, false>(m_decimaters.size());
	m_importanceSums = make_udevptr_array<Device::CUDA, DeviceImportanceSums<Device::CUDA>, false>(m_decimaters.size());
	cuda::check_error(cudaMemset(m_importanceSums.get(), 0, sizeof(ImportanceSums) * m_decimaters.size()));

	logInfo("Initial decimation: ", std::chrono::duration_cast<std::chrono::milliseconds>(CpuProfileState::get_process_time() - timeBegin).count(), "ms");
}

void GpuShadowSilhouettesPT::update_reduction_factors() {
	// Get the sums from the GPU
	std::unique_ptr<ImportanceSums[]> sums = std::make_unique<ImportanceSums[]>(m_decimaters.size());
	copy<char>(reinterpret_cast<char*>(sums.get()), reinterpret_cast<const char*>(m_importanceSums.get()), sizeof(ImportanceSums) * m_decimaters.size());
	// ...then reset them
	cuda::check_error(cudaMemset(m_importanceSums.get(), 0, sizeof(ImportanceSums) * m_decimaters.size()));

	m_remainingVertexFactor.clear();
	if(m_params.reduction == 0.f) {
		// Do not reduce anything
		for(std::size_t i = 0u; i < m_decimaters.size(); ++i) {
			m_decimaters[i]->update_importance_density(sums[i]);
			m_remainingVertexFactor.push_back(1.0);
		}
		return;
	}

	double expectedVertexCount = 0.0;
	for(std::size_t i = 0u; i < m_decimaters.size(); ++i) {
		auto& decimater = m_decimaters[i];
		decimater->update_importance_density(sums[i]);
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

} // namespace mufflon::renderer::decimaters::silhouette