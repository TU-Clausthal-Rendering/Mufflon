#include "cpu_combined_reducer.hpp"
#include "combined_gathering.hpp"
#include "core/scene/geometry/util.hpp"
#include "core/scene/clustering/uniform_clustering.hpp"
#include "core/renderer/decimaters/combined/modules/importance_quadrics.hpp"
#include "profiler/cpu_profiler.hpp"
#include <random>
#include <OpenMesh/Tools/Decimater/DecimaterT.hh>
#include <OpenMesh/Tools/Decimater/ModQuadricT.hh>

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

	if(get_reset_event() & ResetEvent::PARAMETER && !(get_reset_event() & (ResetEvent::RENDERER_ENABLE | ResetEvent::CAMERA)))
		m_stage = Stage::NONE;

	// Initialize the decimaters
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
		// TODO: cache the currently unneeded sums to disk
		if(m_instanceImportanceSums == nullptr)
			m_instanceImportanceSums = std::make_unique<util::SwappedVector<std::atomic<double>>>("instImpSwap.file", m_world.get_frame_count(),
																								  m_sceneDesc.activeInstances);
	}
}

void CpuCombinedReducer::on_world_clearing() {
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

		// Remove instances that are below our threshold
		// TODO: how to get access to other frames? Are they present in the scene?
		auto& objects = m_currentScene->get_objects();
		const auto& instances = m_currentScene->get_instances();
		std::vector<std::tuple<scene::ObjectHandle, u32, double>> instanceList;
		instanceList.reserve(instances.size());
		double currRemovedImp = 0.0;
		const auto& instImpSums = m_instanceImportanceSums->active_slot();
		for(auto& obj : objects) {
			// Beware: obj is a reference, and the count may be changed in 'remove_instance'!
			for(u32 curr = 0u; curr < obj.second.count; ++curr) {
				const auto instIdx = instances[obj.second.offset + curr]->get_index();
				if(const auto currSum = instImpSums[instIdx].load(std::memory_order_acquire);
				   currSum < m_params.maxInstanceDensity)
					instanceList.emplace_back(obj.first, curr, currSum);
			}
		}
		// Sort by importance sum
		std::sort(instanceList.begin(), instanceList.end(), [](const auto& a, const auto& b) { return std::get<2>(a) < std::get<2>(b); });
		// Find the cutoff (all instances to be removed, determined by sum of importance sums)
		double removedImpSum = 0.f;
		std::size_t maxIndex;
		for(maxIndex = 0u; maxIndex < instanceList.size(); ++maxIndex) {
			removedImpSum += std::get<2>(instanceList[maxIndex]);
			if(removedImpSum > static_cast<double>(m_params.maxInstanceDensity))
				break;
		}
		// Sort sub-range again by object and index
		std::sort(instanceList.begin(), instanceList.begin() + maxIndex, [](const auto& a, const auto& b) {
			if(std::get<0>(a) == std::get<0>(b))
				return std::get<1>(a) > std::get<1>(b);
			else
				return std::get<0>(a) < std::get<0>(b);
		});
		// Now we can remove the instances
		for(std::size_t i = 0u; i < maxIndex; ++i)
			m_currentScene->remove_instance(std::get<0>(instanceList[i]), std::get<1>(instanceList[i]));
		logInfo("Removed ", maxIndex, " instances with low importance");

		// Update the reduction factors
		this->update_reduction_factors(startFrame, endFrame);

		const auto processTime = CpuProfileState::get_process_time();
		const auto cycles = CpuProfileState::get_cpu_cycle();
		
		std::vector<std::vector<bool>> octreeNodeMasks(get_max_thread_num());
		std::vector<std::vector<renderer::decimaters::FloatOctree::NodeIndex>> currLevels(get_max_thread_num());
		std::vector<std::vector<renderer::decimaters::FloatOctree::NodeIndex>> nextLevels(get_max_thread_num());
		std::vector<scene::geometry::PolygonMeshType> meshes(get_max_thread_num());
		std::vector<std::vector<u32>> vertexPositions(get_max_thread_num());

		// Create curvature and importance properties for the meshes
		std::vector<OpenMesh::VPropHandleT<float>> curvProps;
		std::vector<OpenMesh::VPropHandleT<float>> accumImpProps;
		curvProps.reserve(get_max_thread_num());
		accumImpProps.reserve(get_max_thread_num());
		for(auto& mesh : meshes) {
			curvProps.emplace_back();
			accumImpProps.emplace_back();
			mesh.add_property(curvProps.back());
			mesh.add_property(accumImpProps.back());
		}

#pragma omp parallel for schedule(dynamic)
		for(i32 i = 0; i < static_cast<i32>(m_viewOctrees[m_world.get_frame_current()].octree_count()); ++i) {
			// Only update objects that have instances left
			auto& obj = objects.cbegin() + i;
			if(obj->second.count > 0u) {
				const auto threadIdx = get_current_thread_idx();
				octreeNodeMasks[threadIdx].reserve(m_viewOctrees.front().size());
				currLevels[threadIdx].reserve(50000u);
				nextLevels[threadIdx].reserve(50000u);

				// First create the mesh
				auto& mesh = meshes[threadIdx];
				mesh.clean_keep_reservation();
				// Reload the original LoD
				const auto lodLevel = 0u;
				if(!m_world.load_lod(*obj->first, lodLevel, true))
					throw std::runtime_error("Failed to reload LoD " + std::to_string(lodLevel) + " for object '"
											 + std::string(obj->first->get_name()) + "'");
				auto& lod = obj->first->get_lod(lodLevel);
				auto& polygons = lod.template get_geometry<scene::geometry::Polygons>();
				polygons.create_halfedge_structure(mesh);

				// Compute the per-vertex importance
				// TODO
				for(const auto vertex : mesh.vertices())
					mesh.property(curvProps[threadIdx], vertex) = 1.f;

				// Map the importance back to the original mesh
				for(const auto vertex : mesh.vertices()) {
					// Put importance into temporary storage
					// TODO: end of frame sequence!
					float importance = 0.f;
					const auto area = compute_area(mesh, vertex);
					const auto curv = std::abs(mesh.property(curvProps[threadIdx], vertex));

					auto actualStartFrame = startFrame;
					auto actualEndFrame = endFrame;

					switch(m_params.impWeightMethod) {
						case combined::PImpWeightMethod::Values::AVERAGE_ALL:
							actualStartFrame = 0u;
							actualEndFrame = m_world.get_frame_count() - 1u;
							[[fallthrough]];
						case combined::PImpWeightMethod::Values::AVERAGE:
							for(u32 f = actualStartFrame; f <= actualEndFrame; ++f) {
								// Fetch octree value (TODO)
								const auto imp = m_viewOctrees[f][i].get_density(util::pun<ei::Vec3>(mesh.point(vertex)),
																				 util::pun<ei::Vec3>(mesh.normal(vertex)));
								importance += imp;
							}
							importance /= static_cast<float>(actualEndFrame - actualStartFrame + 1u);
							break;
						case combined::PImpWeightMethod::Values::MAX_ALL:
							actualStartFrame = 0u;
							actualEndFrame = m_world.get_frame_count() - 1u;
							[[fallthrough]];
						case combined::PImpWeightMethod::Values::MAX:
							for(u32 f = actualStartFrame; f <= actualEndFrame; ++f) {
								// Fetch octree value (TODO)
								const auto imp = m_viewOctrees[f][i].get_density(util::pun<ei::Vec3>(mesh.point(vertex)),
																				 util::pun<ei::Vec3>(mesh.normal(vertex)));
								importance = std::max<float>(importance, imp);
							}
							break;
					}
					const auto weightedImportance = std::sqrt(importance * curv) / area;
					mesh.property(accumImpProps[threadIdx], vertex) = weightedImportance;
				}

				if(m_remainingVertices[i] < mesh.n_vertices()) {
					const auto t0 = std::chrono::high_resolution_clock::now();

					// Perform decimation
					OpenMesh::Decimater::DecimaterT<scene::geometry::PolygonMeshType> decimater{ mesh };
					combined::modules::ImportanceDecimationModule<>::Handle impHandle;
					decimater.add(impHandle);
					decimater.module(impHandle).set_properties(mesh, accumImpProps[threadIdx]);
					decimater.initialize();

					// TODO: cluster!
					/*
					collapses = m_decimatedPoly->cluster_decimate(*m_viewImportance[frame], decimater,
																  targetVertexCount, maxDensity, &octreeNodeMask,
																  &currLevel, &nextLevel);*/
					//collapses = m_decimatedPoly->decimate(decimater, targetVertexCount, false);
					//const auto collapses = m_decimatedPoly->decimate(decimater, targetCount, false);
					const auto collapses = decimater.decimate_to(m_remainingVertices[i]);
					const auto t1 = std::chrono::high_resolution_clock::now();
					const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0);
					logInfo("Collapse duration: ", duration.count(), "ms");
					if(collapses > 0u) {
						polygons.reconstruct_from_reduced_mesh(mesh, &vertexPositions[threadIdx]);
						lod.clear_accel_structure();
					}
					logInfo("Performed ", collapses, " collapses for object '", obj->first->get_name(),
								"', remaining vertices: ", polygons.get_vertex_count());
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
			//if(m_params.reduction == 0)
			//	display_importance();
		} else {
			logInfo("Gathering importance for frame ", m_world.get_frame_current());

			const auto processTime = CpuProfileState::get_process_time();
			const auto cycles = CpuProfileState::get_cpu_cycle();

			// Do the usual importance gathering
			m_instanceImportanceSums->change_slot(m_world.get_frame_current(), false, false);
			gather_importance();
			m_instanceImportanceSums->change_slot(m_world.get_frame_current(), true, false);

			// Update the importance sums
			{
				const auto t0 = std::chrono::high_resolution_clock::now();
				// Join the irradiance octrees into our main octrees
				for(std::size_t i = 0u; i < m_viewOctrees[m_world.get_frame_current()].octree_count(); ++i)
					m_viewOctrees[m_world.get_frame_current()][i].join(m_irradianceOctrees[m_world.get_frame_current()][i], m_params.lightWeight);
				const auto t1 = std::chrono::high_resolution_clock::now();
				logPedantic("Join time ", std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count(), "ms");
			}

			m_hasFrameImp[m_world.get_frame_current()] = true;
			logInfo("Finished importance gathering (",
					std::chrono::duration_cast<std::chrono::milliseconds>(processTime).count(),
					"ms, ", cycles / 1'000'000, " MCycles)");
			//if(m_params.reduction == 0) {
			//	display_importance();
			//}
		}
	} else if(m_stage == Stage::IMPORTANCE_GATHERED && m_params.reduction == 0) {
		// TODO: re-enable?
		/*std::size_t vertices = 0u;
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
		display_importance(true);*/
	}
}

void CpuCombinedReducer::gather_importance() {
	if(m_params.maxPathLength >= 16u) {
		logError("[CpuCombinedReducer::gather_importance] Max. path length too long (max. 15 permitted)");
		return;
	}

	const u32 NUM_PIXELS = m_outputBuffer.get_num_pixels();
	const auto shadowStatusOffset = m_world.get_frame_current() * m_lightCount * (m_params.maxPathLength - 1) * NUM_PIXELS;
	auto* instImpSums = m_instanceImportanceSums->active_slot().data();
	for(int iter = 0; iter < m_params.importanceIterations; ++iter) {
#pragma PARALLEL_FOR
		for(int pixel = 0; pixel < (int)NUM_PIXELS; ++pixel) {
			const Pixel coord{ pixel % m_outputBuffer.get_width(), pixel / m_outputBuffer.get_width() };
			scene::PrimitiveHandle shadowPrim;
			combined::sample_importance_octree(m_outputBuffer, m_sceneDesc, m_params,
											   coord, m_rngs[pixel], m_viewOctrees[m_world.get_frame_current()].data(),
											   m_irradianceOctrees[m_world.get_frame_current()].data(), instImpSums,
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

#if 0
void CpuCombinedReducer::display_importance(const bool accumulated) {
	// TODO: accumulated!
	const u32 NUM_PIXELS = m_outputBuffer.get_num_pixels();
#pragma PARALLEL_FOR
	for(int pixel = 0; pixel < (int)NUM_PIXELS; ++pixel) {
		const Pixel coord{ pixel % m_outputBuffer.get_width(), pixel / m_outputBuffer.get_width() };
		if(accumulated)
			combined::sample_vis_importance(m_outputBuffer, m_sceneDesc, coord,
											m_rngs[pixel], m_accumImpAccess.data(),
											&m_instanceImportanceSums[m_sceneDesc.numInstances * m_world.get_frame_current()],
											m_world.get_frame_current());
		else
			combined::sample_vis_importance_octree(m_outputBuffer, m_sceneDesc, coord,
												   m_rngs[pixel], m_viewOctrees[m_world.get_frame_current()].data(),
												   &m_instanceImportanceSums[m_sceneDesc.numInstances * m_world.get_frame_current()], 
												   m_world.get_frame_current());
	}
}
#endif // 0

double CpuCombinedReducer::get_lod_importance(const u32 frame, const scene::Scene::InstanceRef obj) const noexcept {
	double sum = 0.0;
	for(u32 i = obj.offset; i < (obj.offset + obj.count); ++i) {
		const auto idx = m_currentScene->get_instances()[i]->get_index();
		sum += m_instanceImportanceSums->active_slot()[idx];
	}
	return sum;
}

void CpuCombinedReducer::initialize_decimaters() {
	auto& objects = m_currentScene->get_objects();
	const auto& instances = m_currentScene->get_instances();
	m_originalVertices.resize(objects.size());

	const auto timeBegin = CpuProfileState::get_process_time();

	m_viewOctrees.clear();
	m_irradianceOctrees.clear();
	m_viewOctrees.reserve(m_world.get_frame_count());
	m_irradianceOctrees.reserve(m_world.get_frame_count());

	// TODO: only set this up for reduction
#ifndef DEBUG_ENABLED
	std::size_t threadCount = get_max_thread_num();
#else // DEBUG_ENABLED
	std::size_t threadCount = 1u;
#endif // DEBUG_ENABLED

	std::vector<scene::geometry::PolygonMeshType> meshes(threadCount);
	std::vector<std::pair<u32, scene::WorldContainer::LodMetadata>> lodSizeMap(objects.size());
	std::unique_ptr<std::pair<std::atomic_uint32_t, u32>[]> threadLodStarts;
	// We only need extra coordination if we reduce initially
	if(m_params.initialReduction > 0.f) {
		for(auto& mesh : meshes) {
			mesh.request_vertex_status();
			mesh.request_face_status();
			mesh.request_edge_status();
		}

		// Fetch all LoD sizes so we can keep the process beneath a given memory threshold
		const auto lodSizes = m_world.load_lods_metadata();
		double totalCost = 0.0;
		const auto estimateCost = [this](const scene::WorldContainer::LodMetadata& meta) {
			constexpr float loadWeight = 1.f;
			constexpr float collapseWeight = 10.f;
			constexpr float constantWeight = 0.f;
			const auto bytes = meta.vertices * (2u * sizeof(ei::Vec3) + sizeof(ei::Vec2))
				+ meta.triangles * 3u * sizeof(u32)
				+ meta.quads * 4u * sizeof(u32);
			auto cost = loadWeight * bytes + constantWeight;
			if(meta.vertices >= static_cast<std::size_t>(m_params.threshold))
				cost += collapseWeight * static_cast<float>(meta.vertices)* m_params.initialReduction;
			return cost;
		};
		{
			auto currObject = objects.cbegin();
			for(u32 o = 0; o < static_cast<u32>(objects.size()); ++o) {
				const auto objId = currObject->first->get_object_id();
				// TODO: proper LoD level
				// TODO: the metadata also contains data for multiple LoDs which we have to skip!
				if(currObject->first->get_lod_slot_count() > 1u)
					throw std::runtime_error("We currently cannot support multiple LoDs in file for this");
				lodSizeMap[o] = std::make_pair(o, lodSizes[objId]);
				totalCost += estimateCost(lodSizes[objId]);
				++currObject;
			}
		}
		// Sort by size to get an easy lookup for the worker threads
		std::sort(lodSizeMap.begin(), lodSizeMap.end(), [](const auto& lhs, const auto& rhs) {
			// Inverted comparison because we want it in descending order
			return lhs.second.vertices > rhs.second.vertices;
		});
		// Compute the start and end LoDs for each thread based on the assumption that each thread
		// should roughly perform an equal amount of collapses.
		// TODO: we somehow have to take into account memory limits
		{
			threadLodStarts = std::make_unique<std::pair<std::atomic_uint32_t, u32>[]>(threadCount);
			std::size_t partitionCount = 0u;
			u32 currStart = 0u;
			double currTotal = 0u;
			for(u32 i = 0u; i < static_cast<u32>(lodSizeMap.size()); ++i) {
				currTotal += estimateCost(lodSizeMap[i].second);
				if(currTotal >= totalCost / threadCount) {
					threadLodStarts[partitionCount++] = std::make_pair(currStart, i + 1u);
					currTotal = 0u;
					currStart = i + 1u;
				}
			}
			// Last thread picks up the slack
			if(currStart < lodSizeMap.size()) {
				if(partitionCount == threadCount)
					threadLodStarts[threadCount - 1u].second = static_cast<u32>(lodSizeMap.size());
				else
					threadLodStarts[partitionCount++] = std::make_pair(currStart, static_cast<u32>(lodSizeMap.size()));
			}

			if(partitionCount > 1u) {
				// Now we have to ensure that we do not go over our budget
				// For this we progressively shift LoDs up and compensate the lower threads to re-balance
				const auto estimateSize = [&lodSizeMap](const std::size_t index) {
					const auto verts = lodSizeMap[index].second.vertices;
					const auto faces = lodSizeMap[index].second.triangles + lodSizeMap[index].second.quads;
					const auto edges = lodSizeMap[index].second.edges;
					const auto lodSize = verts * (2u * sizeof(ei::Vec3) + sizeof(ei::Vec2))
						+ lodSizeMap[index].second.triangles * 3u * sizeof(u32)
						+ lodSizeMap[index].second.quads * 4u * sizeof(u32);
					// Estimate the amount of storage OpenMesh needs for the mesh
					const auto meshSize = verts * (sizeof(OpenMesh::ArrayItems::Vertex) + 2u * sizeof(OpenMesh::Vec3f) + sizeof(OpenMesh::Vec2f) + 1u)
						+ faces * (sizeof(OpenMesh::ArrayItems::Face) + sizeof(scene::MaterialIndex) + 1u)
						+ edges * (sizeof(OpenMesh::ArrayItems::Edge) + 3u);
					// Size needed for error quadrics/decimater etc.
					const auto decimaterSize = verts * (sizeof(OpenMesh::Geometry::Quadricd)			// Quadric error module
														+ sizeof(OpenMesh::HalfedgeHandle)				// Collapse target
														+ sizeof(float)									// Collapse priority
														+ sizeof(int)									// Heap position
														+ sizeof(OpenMesh::VertexHandle));				// Heap reference
					return lodSize + meshSize + decimaterSize;
				};
				const auto estimateCumSize = [&estimateSize, &lodSizeMap, &threadLodStarts, &partitionCount]() {
					std::size_t size = 0u;
					for(std::size_t t = 0u; t < partitionCount; ++t)
						size += estimateSize(threadLodStarts[t].first.load(std::memory_order_acquire));
					return size;
				};
				const auto estimateCumCost = [&estimateCost, &threadLodStarts, &lodSizeMap](const std::size_t threadId) {
					float cumCost = 0.f;
					for(u32 i = threadLodStarts[threadId].first.load(std::memory_order_acquire); i < threadLodStarts[threadId].second; ++i)
						cumCost += estimateCost(lodSizeMap[i].second);
					return cumCost;
				};
				const auto budgetSize = std::max<std::size_t>(8'000'000'000llu, estimateSize(threadLodStarts[0u].first.load(std::memory_order_acquire)));
				std::size_t cumSize;
				while((cumSize = estimateCumSize()) > budgetSize) {
					// TODO: we could also reallocate on the fly and add extra threads once the heavy meshes are done,
					// but that would be HELLA complicated...
					// Shift and reevaluate cost
					threadLodStarts[0u].second += 1u;
					printf("Cum size: %zu of %zu; thread limit: %u\n", cumSize, budgetSize, threadLodStarts[0u].second);
					fflush(stdout);
					const auto newThreadCost = estimateCumCost(0u);
					// Now shift the other threads to match the cost
					for(std::size_t t = 1u; t < partitionCount; ++t) {
						// Adjust the thread start according to the last thread end
						threadLodStarts[t].first.store(threadLodStarts[t - 1u].second, std::memory_order_release);
						threadLodStarts[t].second = std::max(threadLodStarts[t].second, threadLodStarts[t - 1u].second);
						float currCost = estimateCumCost(t);
						// Add new LoDs to the thread while we don't exceed the new cost (and have LoDs left)
						while(currCost < newThreadCost && threadLodStarts[t].second < static_cast<u32>(lodSizeMap.size())) {
							const auto additionalCost = estimateCost(lodSizeMap[threadLodStarts[t].second].second);
							threadLodStarts[t].second += 1u;
							currCost += additionalCost;
						}
					}
				}
				// Screen for empty threads
				for(std::size_t t = 1u; t < partitionCount; ++t) {
					if(threadLodStarts[t].first.load(std::memory_order_acquire) >= threadLodStarts[t].second) {
						partitionCount = t + 1u;
						break;
					}
				}
			}

			// Make sure that every thread has a valid lookup (even if it means they won't do work)
			threadCount = partitionCount;
		}
	}

	// First we load in the LoDs (possibly reduced version)
	std::atomic_uint32_t progress = 0u;
	auto loader = [&](const std::size_t threadId) {
		// Work all assigned LoDs, and once we're done with our work try to steal work from lower-memory threads
		// For this reason we over-prioritize lower-mem thread work queues
		bool first = true;
		for(std::size_t t = threadId; t < threadCount; ++t) {
			u32 currMapIndex;
			while((currMapIndex = threadLodStarts[t].first.fetch_add(1, std::memory_order_acq_rel)) < threadLodStarts[t].second) {
				if(const auto prev = progress.fetch_add(1u, std::memory_order_acq_rel) + 1u; prev % 100 == 0u)
					logInfo("Progess: ", prev, " of ", objects.size());
				const auto index = lodSizeMap[currMapIndex].first;
				auto objIter = objects.begin() + index;
				auto& obj = *objIter;

				// TODO: proper LoD levels!
				// Load and pre-reduce the LoDs if applicable
				const u32 lodIndex = 0u;
				if(!m_world.load_lod(*obj.first, lodIndex, m_params.initialReduction > 0.f))
					throw std::runtime_error("Failed to load object LoD!");
				m_originalVertices[index] = obj.first->get_lod(lodIndex).template get_geometry<scene::geometry::Polygons>().get_vertex_count();

				if(m_params.initialReduction > 0.f) {
					auto& lod = obj.first->get_reduced_lod(lodIndex);
					auto& polygons = lod.template get_geometry<scene::geometry::Polygons>();
					if(const auto origVertCount = polygons.get_vertex_count(); origVertCount >= m_params.threshold) {
						// Pre-reduce
						const auto targetVertexCount = static_cast<std::size_t>((1.f - m_params.initialReduction)
																				* static_cast<float>(origVertCount));
						const auto gridRes = static_cast<std::size_t>(std::ceil(std::cbrt(2.f * static_cast<float>(targetVertexCount))));
						auto& mesh = meshes[threadId];
						mesh.clean_keep_reservation();
						polygons.create_halfedge_structure(mesh);
						if(first) {
							logInfo("Thread ", threadId, " mesh size: ", mesh.n_vertices(), "/", mesh.n_faces(), "/", mesh.n_edges());
							first = false;
						}
						scene::clustering::UniformVertexClusterer clusterer{ ei::UVec3{ gridRes }};
						clusterer.cluster(mesh, polygons.get_bounding_box(), false);

						// TODO: persistent decimater? persistent error quadrics?
						//OpenMesh::Decimater::DecimaterT<scene::geometry::PolygonMeshType> decimater{ mesh };
						//OpenMesh::Decimater::ModQuadricT<scene::geometry::PolygonMeshType>::Handle modQuadricHandle;
						//decimater.add(modQuadricHandle);
						//decimater.initialize();
						// Possibly repeat until we reached the desired count
						//const auto performedCollapses = decimater.decimate_to(targetVertexCount);
						polygons.reconstruct_from_reduced_mesh(mesh);
						logPedantic("Loaded reduced LoD '", obj.first->get_name(), "' (",
									origVertCount, " -> ", polygons.get_vertex_count(), ")");
						lod.clear_accel_structure();
					}
					obj.first->remove_original_lod(lodIndex);
				}
			}
		}
	};
	std::vector<std::thread> workerThreads;
	workerThreads.reserve(threadCount);
	for(std::size_t i = 0u; i < threadCount; ++i)
		workerThreads.emplace_back(loader, i);
	for(auto& thread : workerThreads) {
		if(thread.joinable())
			thread.join();
	}


	for(std::size_t i = 0u; i < m_world.get_frame_count(); ++i) {
		m_viewOctrees.emplace_back(static_cast<u32>(m_params.impCapacity), static_cast<u32>(objects.size()), true);
		m_irradianceOctrees.emplace_back(static_cast<u32>(m_params.impCapacity), static_cast<u32>(objects.size()), false);

		for(const auto& obj : objects) {
			// TODO: proper LoD levels!
			// Load and pre-reduce the LoDs if applicable
			const u32 lodIndex = 0u;
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
		if(lowestLevel != 0u)
			throw std::runtime_error("Unsupported LoD level (will be fixed soonTM)!");
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
		for(const auto count : m_originalVertices)
			m_remainingVertices.push_back(count);
		return;
	}

	// Compute the total expected vertex count over all meshes
	double importanceSum = 0.0;
	std::size_t reducibleVertices = 0u;
	std::size_t nonReducibleVertices = 0u;
	std::size_t totalVertexCount = 0u;

	std::vector<double> importance(m_viewOctrees[m_world.get_frame_current()].octree_count());

	switch(m_params.vertexDistMethod) {
		case combined::PVertexDistMethod::Values::AVERAGE_ALL:
			frameStart = 0u;
			frameEnd = m_world.get_frame_count() - 1u;
			[[fallthrough]];
		case combined::PVertexDistMethod::Values::AVERAGE:
		{
			for(u32 frame = frameStart; frame <= frameEnd; ++frame) {
				m_instanceImportanceSums->change_slot(frame, false, true);
				auto iter = m_currentScene->get_objects().cbegin();
				for(std::size_t i = 0u; i < m_originalVertices.size(); ++i) {
					if(m_originalVertices[i] > m_params.threshold)
						importance[i] += get_lod_importance(frame, (iter++)->second);
				}
			}

			auto iter = m_currentScene->get_objects().cbegin();
			for(std::size_t i = 0u; i < m_originalVertices.size(); ++i) {
				totalVertexCount += m_originalVertices[i];
				if((iter++)->second.count == 0)
					continue;
				importance[i] /= static_cast<double>(frameEnd - frameStart + 1u);
				importanceSum += importance[i];
				if(m_originalVertices[i] > m_params.threshold)
					reducibleVertices += m_originalVertices[i];
				else
					nonReducibleVertices += m_originalVertices[i];
			}
		}	break;
		case combined::PVertexDistMethod::Values::MAX_ALL:
			frameStart = 0u;
			frameEnd = m_world.get_frame_count() - 1u;
			[[fallthrough]];
		case combined::PVertexDistMethod::Values::MAX:
		{
			for(u32 frame = frameStart; frame <= frameEnd; ++frame) {
				m_instanceImportanceSums->change_slot(frame, false, true);
				auto iter = m_currentScene->get_objects().cbegin();
				for(std::size_t i = 0u; i < m_originalVertices.size(); ++i) {
					totalVertexCount += m_originalVertices[i];
					if(iter->second.count == 0) {
						++iter;
						continue;
					}
					if(m_originalVertices[i] > m_params.threshold) {
						importance[i] = std::max(importance[i], get_lod_importance(frame, (iter++)->second));
						reducibleVertices += m_originalVertices[i];
					} else {
						nonReducibleVertices += m_originalVertices[i];
					}
				}
			}
			reducibleVertices /= frameEnd - frameStart + 1u;
			nonReducibleVertices /= frameEnd - frameStart + 1u;
		}	break;
		default: throw std::runtime_error("Invalid vertex budget method");
	}

	// Determine the reduction parameters for each mesh
	const std::size_t totalTargetVertexCount = static_cast<std::size_t>((1.f - m_params.reduction) * totalVertexCount);
	std::size_t reducedVertexPool = totalTargetVertexCount - nonReducibleVertices;

	auto iter = m_currentScene->get_objects().cbegin();
	for(std::size_t i = 0u; i < m_originalVertices.size(); ++i) {
		if(iter->second.count == 0) {
			m_remainingVertices.push_back(0u);
			continue;
		}
		if(m_originalVertices[i] > m_params.threshold) {
			const auto targetVertexCount = std::min(m_originalVertices[i],
													static_cast<std::size_t>(importance[i]
																			 * static_cast<double>(reducedVertexPool) / importanceSum));
			reducedVertexPool -= targetVertexCount;
			importanceSum -= importance[i];
			m_remainingVertices.push_back(targetVertexCount);
		} else {
			m_remainingVertices.push_back(m_originalVertices[i]);
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