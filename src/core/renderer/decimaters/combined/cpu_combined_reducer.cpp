#include "cpu_combined_reducer.hpp"
#include "combined_gathering.hpp"
#include "profiler/cpu_profiler.hpp"
#include <random>

namespace mufflon::renderer::decimaters {

CpuCombinedReducer::CpuCombinedReducer(mufflon::scene::WorldContainer& world) :
	RendererBase<Device::CPU, combined::CombinedTargets>{ world }
{
	std::random_device rndDev;
	m_rngs.emplace_back(static_cast<u32>(rndDev()));
}

void CpuCombinedReducer::pre_reset() {
	if((get_reset_event() & ResetEvent::CAMERA) != ResetEvent::NONE || get_reset_event().resolution_changed())
		init_rngs(m_outputBuffer.get_num_pixels());

	if((get_reset_event() & ResetEvent::SCENARIO) != ResetEvent::NONE && m_stage == Stage::IMPORTANCE_GATHERED) {
		// At least activate the created LoDs
		/*for(auto& obj : m_currentScene->get_objects()) {
			const u32 newLodLevel = static_cast<u32>(obj.first->get_lod_slot_count() - 1u);
			// TODO: this reeeeally breaks instancing
			m_world.get_current_scenario()->set_custom_lod(obj.first, newLodLevel);
		}*/
	}

	if(get_reset_event() & ResetEvent::PARAMETER && !(get_reset_event() & (ResetEvent::RENDERER_ENABLE | ResetEvent::CAMERA)))
		m_stage = Stage::NONE;

	// Initialize the decimaters
	// TODO: how to deal with instancing
	if(m_stage == Stage::NONE) {
		this->initialize_decimaters();
		m_stage = Stage::INITIALIZED;
		m_hasFrameImp = std::vector<bool>(m_world.get_frame_count(), false);
		m_instanceImportanceSums.reset();
	}

	RendererBase<Device::CPU, combined::CombinedTargets>::pre_reset();
}

void CpuCombinedReducer::post_reset() {
	if(m_stage == Stage::INITIALIZED) {
		m_lightCount = 1u + m_sceneDesc.lightTree.posLights.lightCount + m_sceneDesc.lightTree.dirLights.lightCount;
		const auto statusCount = (m_params.maxPathLength - 1) * m_lightCount
			* static_cast<std::size_t>(m_outputBuffer.get_num_pixels())
			* static_cast<std::size_t>(m_world.get_frame_count());
		m_shadowStatus = make_udevptr_array<Device::CPU, combined::ShadowStatus, false>(statusCount);
		std::memset(m_shadowStatus.get(), 0, sizeof(combined::ShadowStatus) * statusCount);

		// The descriptor already accounts for all instances (over all frames);
		// unfortunately we do not know which ones are no-animated (ie. over all frames)
		if(m_instanceImportanceSums == nullptr)
			m_instanceImportanceSums = std::make_unique<std::atomic<double>[]>(m_sceneDesc.activeInstances * m_world.get_frame_count());
	}
}

void CpuCombinedReducer::on_world_clearing() {
	m_decimaters.clear();
	m_stage = Stage::NONE;
}

void CpuCombinedReducer::post_iteration(IOutputHandler& outputBuffer) {
	if(m_stage == Stage::INITIALIZED && (m_world.get_frame_current() + 1u) == m_world.get_frame_count()) {
		logInfo("Finished importance acquisition");
		m_stage = Stage::IMPORTANCE_GATHERED;
		m_isFrameReduced = std::vector<bool>(m_world.get_frame_count(), false);
	} else if(m_stage == Stage::IMPORTANCE_GATHERED && !m_isFrameReduced[m_world.get_frame_current()]) {
		logInfo("Performing decimation iteration...");

		const auto windowHalfWidth = (m_params.slidingWindowHalfWidth == 0u)
			? m_world.get_frame_count()
			: m_params.slidingWindowHalfWidth;
		const u32 startFrame = std::max<u32>(m_world.get_frame_current(), m_params.slidingWindowHalfWidth)
			- m_params.slidingWindowHalfWidth;
		const u32 endFrame = std::min<u32>(m_world.get_frame_current() + m_params.slidingWindowHalfWidth,
										   m_world.get_frame_count() - 1u);

		// Update the reduction factors
		this->update_reduction_factors(startFrame, endFrame);

		const auto processTime = CpuProfileState::get_process_time();
		const auto cycles = CpuProfileState::get_cpu_cycle();
		auto scope = Profiler::core().start<CpuProfileState>("Silhouette decimation");

#pragma omp parallel for schedule(dynamic)
		for(i32 i = 0; i < static_cast<i32>(m_decimaters.size()); ++i) {
			m_decimaters[i]->update(m_params.impWeightMethod, startFrame, endFrame);
			m_decimaters[i]->reduce(m_remainingVertices[i], m_params.maxClusterDensity, m_world.get_frame_current());
		}

		// Remove instances that are below our threshold
		// TODO: how to get access to other frames? Are they present in the scene?

		const auto& objects = m_currentScene->get_objects();
		const auto& instances = m_currentScene->get_instances();
		const auto impOffset = m_sceneDesc.numInstances * m_world.get_frame_current();
		double currRemovedImp = 0.0;
		for(auto& obj : objects) {
			u32 curr = 0u;
			// Beware: obj is a reference, and the count may be changed in 'remove_instance'!
			while(curr < obj.second.count) {
				const auto instIdx = instances[obj.second.offset + curr]->get_index();
				if(const auto currSum = m_instanceImportanceSums[impOffset + instIdx].load(std::memory_order_acquire);
				   currSum < m_params.maxInstanceDensity) {
					const auto otherIdx = m_currentScene->remove_instance(obj.first, curr);
					currRemovedImp += currSum;
				} else {
					curr += 1u;
				}
			}
		}

		m_currentScene->clear_accel_structure();
		logInfo("Finished decimation iteration (", std::chrono::duration_cast<std::chrono::milliseconds>(CpuProfileState::get_process_time() - processTime).count(),
				"ms, ", (CpuProfileState::get_cpu_cycle() - cycles) / 1'000'000, " MCycles)");

		this->on_manual_reset();
		m_isFrameReduced[m_world.get_frame_current()] = true;
	}
	RendererBase<Device::CPU, combined::CombinedTargets>::post_iteration(outputBuffer);
}

void CpuCombinedReducer::iterate() {
	if(m_stage == Stage::INITIALIZED) {
		if(m_hasFrameImp[m_world.get_frame_current()]) {
			if(m_params.reduction == 0)
				display_importance();
		} else {
			logInfo("Gathering importance for frame ", m_world.get_frame_current());

			const auto processTime = CpuProfileState::get_process_time();
			const auto cycles = CpuProfileState::get_cpu_cycle();

			// Do the usual importance gathering
			gather_importance();

			// Update the importance sums
			for(std::size_t i = 0u; i < m_decimaters.size(); ++i) {
				m_decimaters[i]->finish_gather(m_world.get_frame_current());
			}

			if(m_decimaters.size() == 0u)
				return;

			m_hasFrameImp[m_world.get_frame_current()] = true;
			logInfo("Finished importance gathering (",
					std::chrono::duration_cast<std::chrono::milliseconds>(processTime).count(),
					"ms, ", cycles / 1'000'000, " MCycles)");
			if(m_params.reduction == 0) {
				display_importance();
			}
		}
	} else if(m_stage == Stage::IMPORTANCE_GATHERED && m_params.reduction == 0) {
		std::size_t vertices = 0u;
		for(const auto& decimater : m_decimaters)
			vertices += decimater->get_decimated_vertex_count();
		// TODO: uses quite a bit of memory
		m_accumImportances = std::vector<float>(vertices);
		m_accumImpAccess.clear();
		m_accumImpAccess.reserve(m_decimaters.size());
		vertices = 0u;
		for(const auto& decimater : m_decimaters) {
			m_accumImpAccess.push_back(m_accumImportances.data() + vertices);
			decimater->get_decimated_importance(m_accumImpAccess.back());
			vertices += decimater->get_decimated_vertex_count();
		}
		display_importance(true);
	}
}

void CpuCombinedReducer::gather_importance() {
	if(m_params.maxPathLength >= 16u) {
		logError("[CpuCombinedReducer::gather_importance] Max. path length too long (max. 15 permitted)");
		return;
	}

	const u32 NUM_PIXELS = m_outputBuffer.get_num_pixels();
	const auto shadowStatusOffset = m_world.get_frame_current() * m_lightCount * (m_params.maxPathLength - 1) * NUM_PIXELS;
	for(int iter = 0; iter < m_params.importanceIterations; ++iter) {
#pragma PARALLEL_FOR
		for(int pixel = 0; pixel < (int)NUM_PIXELS; ++pixel) {
			const Pixel coord{ pixel % m_outputBuffer.get_width(), pixel / m_outputBuffer.get_width() };
			scene::PrimitiveHandle shadowPrim;
			combined::sample_importance_octree(m_outputBuffer, m_sceneDesc, m_params,
											   coord, m_rngs[pixel], m_viewOctrees[m_world.get_frame_current()].data(),
											   m_irradianceOctrees[m_world.get_frame_current()].data(),
											   &m_instanceImportanceSums[m_sceneDesc.numInstances * m_world.get_frame_current()],
											   &m_shadowStatus[shadowStatusOffset + pixel * m_lightCount * (m_params.maxPathLength - 1)]);
		}
		logPedantic("Finished importance iteration (", iter + 1, " of ", m_params.importanceIterations, ")");
	}

	// Fix the radiance estimate
#pragma PARALLEL_FOR
	for(int pixel = 0; pixel < (int)NUM_PIXELS; ++pixel) {
		const Pixel coord{ pixel % m_outputBuffer.get_width(), pixel / m_outputBuffer.get_width() };
		const auto rad = m_outputBuffer.template get<combined::RadianceTarget>(coord);
		m_outputBuffer.template set<combined::RadianceTarget>(coord, rad / static_cast<float>(m_params.importanceIterations));

		post_process_shadow(m_outputBuffer, m_sceneDesc, m_params, coord, pixel,
							m_params.importanceIterations, &m_shadowStatus.get()[shadowStatusOffset]);
	}


	// Post-processing
	// TODO
	// TODO: allow for this with proper reset "events"
}

void CpuCombinedReducer::display_importance(const bool accumulated) {
	// TODO: accumulated!
	const u32 NUM_PIXELS = m_outputBuffer.get_num_pixels();
#pragma PARALLEL_FOR
	for(int pixel = 0; pixel < (int)NUM_PIXELS; ++pixel) {
		const Pixel coord{ pixel % m_outputBuffer.get_width(), pixel / m_outputBuffer.get_width() };
		if(accumulated)
			combined::sample_vis_importance(m_outputBuffer, m_sceneDesc, coord,
											m_rngs[pixel], m_accumImpAccess.data(),
											m_importanceSums.data(),
											&m_instanceImportanceSums[m_sceneDesc.numInstances * m_world.get_frame_current()],
											m_world.get_frame_current());
		else
			combined::sample_vis_importance_octree(m_outputBuffer, m_sceneDesc, coord,
												   m_rngs[pixel], m_viewOctrees[m_world.get_frame_current()].data(),
												   m_importanceSums.data(),
												   &m_instanceImportanceSums[m_sceneDesc.numInstances * m_world.get_frame_current()], 
												   m_world.get_frame_current());
	}
}

void CpuCombinedReducer::initialize_decimaters() {
	auto& objects = m_currentScene->get_objects();
	const auto& instances = m_currentScene->get_instances();
	m_decimaters.clear();
	m_decimaters.resize(objects.size());

	const auto timeBegin = CpuProfileState::get_process_time();

	m_viewOctrees.clear();
	m_irradianceOctrees.clear();
	m_importanceSums.clear();
	m_viewOctrees.reserve(m_world.get_frame_count());
	m_irradianceOctrees.reserve(m_world.get_frame_count());
	m_viewOctreeAccess.resize(m_world.get_frame_count() * objects.size());
	m_irradianceOctreeAccess.resize(m_world.get_frame_count() * objects.size());
	m_importanceSums.resize(m_world.get_frame_count() * objects.size());

	for(std::size_t i = 0u; i < m_world.get_frame_count(); ++i) {
		m_viewOctrees.emplace_back(static_cast<u32>(m_params.impCapacity), static_cast<u32>(objects.size()));
		m_irradianceOctrees.emplace_back(static_cast<u32>(m_params.impCapacity), static_cast<u32>(objects.size()));

		std::size_t o = 0u;
		for(const auto& obj : objects) {
			// TODO: proper bounding box!
			const auto aabb = obj.first->get_lod(0).get_bounding_box();

			// We have to weight the splitting factor with the average instance scaling.
			// Since we weight importance with baseArea / area, it otherwise happens
			// that, if all instances are scaled up or down, importance isn't properly
			// gathered
			ei::Vec3 scaleSum{ 0.f };
			for(std::size_t j = obj.second.offset; j < (obj.second.offset + obj.second.count); ++j)
				scaleSum += scene::Instance::extract_scale(m_world.get_world_to_instance_transformation(instances[j]));
			scaleSum /= static_cast<float>(obj.second.count);
			const auto splitScale = ei::max(scaleSum) * ei::max(scaleSum);

			// TODO: proper splitting factors!
			m_viewOctrees.back().create(aabb, 8.f * splitScale);
			m_irradianceOctrees.back().create(aabb, 8u, 8.f * splitScale);

			m_viewOctreeAccess[m_world.get_frame_count() * o + i] = m_viewOctrees.back().data() + o;
			m_irradianceOctreeAccess[m_world.get_frame_count() * o + i] = m_irradianceOctrees.back().data() + o;

			++o;
		}
	}

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
		m_decimaters[i] = std::make_unique<combined::CombinedDecimater>(obj.first->get_name(),
																		lod, newLod, m_world.get_frame_count(),
																		&m_viewOctreeAccess[i * m_world.get_frame_count()],
																		&m_irradianceOctreeAccess[i * m_world.get_frame_count()],
																		&m_importanceSums[i * m_world.get_frame_count()],
																		m_params.lightWeight);
	}
	m_currentScene->clear_accel_structure();
	logInfo("Initial decimation: ", std::chrono::duration_cast<std::chrono::milliseconds>(CpuProfileState::get_process_time() - timeBegin).count(), "ms");
}

void CpuCombinedReducer::update_reduction_factors(u32 frameStart, u32 frameEnd) {
	// To compute the reduction factors for each mesh, we take the total importance
	// per mesh and assign the factors proportionally to them.
	// Bringing in animations, we have different options:
	// We can determine the reduction factors for every frame from the importance
	// sum of every frame, we can take averages, or any other kind of weighting
	// function really

	m_remainingVertices.clear();
	if(m_params.reduction == 0.f) {
		// Do not reduce anything
		for(const auto& decimater : m_decimaters)
			m_remainingVertices.push_back(decimater->get_original_vertex_count());
		return;
	}

	// Compute the total expected vertex count over all meshes
	double importanceSum = 0.0;
	std::size_t reducibleVertices = 0u;
	std::size_t nonReducibleVertices = 0u;

	std::vector<double> importance(m_decimaters.size());

	switch(m_params.vertexDistMethod) {
		case combined::PVertexDistMethod::Values::AVERAGE_ALL:
			frameStart = 0u;
			frameEnd = m_world.get_frame_count() - 1u;
			[[fallthrough]];
		case combined::PVertexDistMethod::Values::AVERAGE:
		{
			for(u32 frame = frameStart; frame <= frameEnd; ++frame) {
				for(std::size_t i = 0u; i < m_decimaters.size(); ++i) {
					auto& decimater = m_decimaters[i];
					if(decimater->get_original_vertex_count() > m_params.threshold)
						importance[i] += decimater->get_importance_sum(frame);
				}
			}

			for(std::size_t i = 0u; i < m_decimaters.size(); ++i) {
				importance[i] /= static_cast<double>(frameEnd - frameStart + 1u);
				importanceSum += importance[i];
				if(m_decimaters[i]->get_original_vertex_count() > m_params.threshold)
					reducibleVertices += m_decimaters[i]->get_original_vertex_count();
				else
					nonReducibleVertices += m_decimaters[i]->get_original_vertex_count();
			}
		}	break;
		case combined::PVertexDistMethod::Values::MAX_ALL:
			frameStart = 0u;
			frameEnd = m_world.get_frame_count() - 1u;
			[[fallthrough]];
		case combined::PVertexDistMethod::Values::MAX:
		{
			for(u32 frame = frameStart; frame <= frameEnd; ++frame) {
				for(std::size_t i = 0u; i < m_decimaters.size(); ++i) {
					auto& decimater = m_decimaters[i];
					if(decimater->get_original_vertex_count() > m_params.threshold) {
						importance[i] = std::max(importance[i], decimater->get_importance_sum(frame));
						reducibleVertices += m_decimaters[i]->get_original_vertex_count();
					} else {
						nonReducibleVertices += m_decimaters[i]->get_original_vertex_count();
					}
				}
			}
			reducibleVertices /= frameEnd - frameStart + 1u;
			nonReducibleVertices /= frameEnd - frameStart + 1u;
		}	break;
		default: throw std::runtime_error("Invalid vertex budget method");
	}

	// Determine the reduction parameters for each mesh
	const auto totalVertexCount = reducibleVertices + nonReducibleVertices;
	const std::size_t totalTargetVertexCount = static_cast<std::size_t>((1.f - m_params.reduction) * totalVertexCount);
	std::size_t reducedVertexPool = totalTargetVertexCount - nonReducibleVertices;

	for(std::size_t i = 0u; i < m_decimaters.size(); ++i) {
		if(m_decimaters[i]->get_original_vertex_count() > m_params.threshold) {
			const auto targetVertexCount = std::min(m_decimaters[i]->get_original_vertex_count(),
													static_cast<std::size_t>(importance[i]
																			 * static_cast<double>(reducedVertexPool) / importanceSum));
			reducedVertexPool -= targetVertexCount;
			importanceSum -= importance[i];
			m_remainingVertices.push_back(targetVertexCount);
		} else {
			m_remainingVertices.push_back(m_decimaters[i]->get_original_vertex_count());
		}
	}
}

void CpuCombinedReducer::init_rngs(int num) {
	m_rngs.resize(num);
	// TODO: incude some global seed into the initialization
	for(int i = 0; i < num; ++i)
		m_rngs[i] = math::Rng(i);
}

} // namespace mufflon::renderer::decimaters