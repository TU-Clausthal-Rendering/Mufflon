#include "ss_importance_gathering_pt.hpp"
#include "cpu_ss_sil_pt.hpp"
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

using namespace ss;

CpuSsSilPT::CpuSsSilPT() {
	std::random_device rndDev;
	m_rngs.emplace_back(static_cast<u32>(rndDev()));
}

void CpuSsSilPT::pre_reset() {
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
		m_shadowPrims.clear();
		m_shadowPrims.resize(m_params.importanceIterations * m_outputBuffer.get_num_pixels());
	}

	RendererBase<Device::CPU, ss::SilhouetteTargets>::pre_reset();
}

void CpuSsSilPT::post_reset() {
	// Post-reset because we need the light count
	if(m_currentDecimationIteration == 0u) {
		const auto lightCount = m_sceneDesc.lightTree.posLights.lightCount + m_sceneDesc.lightTree.dirLights.lightCount
			+ ((m_sceneDesc.lightTree.background.flux > 0.f) ? 1 : 0u);
		// Initialize the penumbra bits: at least one byte per pixel to make my life easier
		m_bytesPerPixel = (lightCount * 2u - 1u) / 8u + 1u;
		m_penumbra.clear();
		m_penumbra.resize(m_outputBuffer.get_num_pixels() * m_bytesPerPixel);
	}

}

void CpuSsSilPT::on_world_clearing() {
	m_decimaters.clear();
	m_currentDecimationIteration = 0u;
}

void CpuSsSilPT::post_iteration(IOutputHandler& outputBuffer) {
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
	RendererBase<Device::CPU, ss::SilhouetteTargets>::post_iteration(outputBuffer);
}

void CpuSsSilPT::iterate() {
	if((int)m_currentDecimationIteration < m_params.decimationIterations) {
		logInfo("Starting decimation iteration (", m_currentDecimationIteration + 1, " of ", m_params.decimationIterations, ")...");

		const auto processTime = CpuProfileState::get_process_time();
		const auto cycles = CpuProfileState::get_cpu_cycle();

		// First we need to reset the importance sums (which we may have kept for visualization)
		for(std::size_t i = 0u; i < m_decimaters.size(); ++i) {
			m_importanceSums[i].shadowImportance.store(0.f);
			m_importanceSums[i].shadowSilhouetteImportance.store(0.f);
		}

		gather_importance();

		this->normalize_radiance();
		this->update_silhouette_importance();

		if(m_decimaters.size() == 0u)
			return;

		// We need to update the importance density
		this->update_reduction_factors();
		//compute_max_importance();

		logInfo("Finished importance gathering (",
				std::chrono::duration_cast<std::chrono::milliseconds>(processTime).count(),
				"ms, ", cycles / 1'000'000, " MCycles)");

	} else {
		if((int)m_currentDecimationIteration == m_params.decimationIterations) {
			if(m_params.reduction == 0) {
				for(auto& decimater : m_decimaters)
					decimater->copy_back_normalized_importance();
				compute_max_importance();
			}
		}
		if(m_params.reduction == 0)
			display_importance();
	}
}

void CpuSsSilPT::gather_importance() {
	if(m_params.maxPathLength >= 16u) {
		logError("[CpuSsSilPT::gather_importance] Max. path length too long (max. 15 permitted)");
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
			silhouette::sample_importance(m_outputBuffer, m_sceneDesc, m_params, coord, m_rngs[pixel],
										  m_importances.data(), m_importanceSums.data(),
										  m_shadowPrims[iter * m_outputBuffer.get_num_pixels() + pixel],
										  &m_penumbra[pixel * m_bytesPerPixel]);
		}

		logPedantic("Finished importance iteration (", iter + 1, " of ", m_params.importanceIterations, ")");
	}
	// TODO: allow for this with proper reset "events"
}

void CpuSsSilPT::update_silhouette_importance() {
	logPedantic("Detecting shadow edges and penumbra...");
	const auto lightCount = m_sceneDesc.lightTree.posLights.lightCount + m_sceneDesc.lightTree.dirLights.lightCount
		+ 1u;
	const bool hasBackground = m_sceneDesc.lightTree.background.flux > 0.f;
	const auto contributePenumbraColor = [actualLightCount = lightCount - (hasBackground ? 0 : 1)](const std::size_t i,
																								   const bool shadowed,
																								   const bool lit) {
		if(shadowed) {
			if(lit)
				return ei::hsvToRgb(ei::Vec3{ (3.f * i + 1.f) / (3.f * actualLightCount), 1.f, 1.f });
			else
				return ei::hsvToRgb(ei::Vec3{ (3.f * i) / (3.f * actualLightCount), 1.f, 1.f });
		} else if(lit) {
			return ei::hsvToRgb(ei::Vec3{ (3.f * i + 2.f) / (3.f * actualLightCount), 1.f, 1.f });
		}
		return ei::Vec3{ 0.f };
	};


//#pragma PARALLEL_FOR
	for(int pixel = hasBackground ? 0 : 1; pixel < m_outputBuffer.get_num_pixels(); ++pixel) {
		const Pixel coord{ pixel % m_outputBuffer.get_width(), pixel / m_outputBuffer.get_width() };

		const u8* penumbraBits = &m_penumbra[pixel * m_bytesPerPixel];
		for(std::size_t i = 0u; i < lightCount; ++i) {
			// "Detect" penumbra
			const auto bitIndex = i / 4u;
			const auto bitOffset = 2u * (i % 4u);
			const bool shadowed = penumbraBits[bitIndex] & (1u << bitOffset);
			const bool lit = penumbraBits[bitIndex] & (1u << (bitOffset + 1u));

			m_outputBuffer.template contribute<PenumbraTarget>(coord, contributePenumbraColor(i, shadowed, lit));

			// Tracks whether it's a hard shadow border
			// Starting condition are viewport boundaries, otherwise purely shadowed pixels
			// with purely lit in their vicinity
			bool isBorder = coord.x == 0 || coord.y == 0 || coord.x == (m_outputBuffer.get_width() - 1)
				|| coord.y == (m_outputBuffer.get_height() - 1);
			// Tracks the transition from core shadow to penumbra
			bool isPenumbraTransition = false;
			// Average radiance of the surrounding not-shadowed pixels
			ei::Vec3 averageRadiance{ 0.f };
			u32 radianceCount = 0u;

			if(shadowed) {
				// Detect silhouettes as shadowed pixels with lit ones as direct neighbors or viewport edge
				for(int y = -1; y <= 1; ++y) {
					for(int x = -1; x <= 1; ++x) {
						if(x == 0 && y == 0)
							continue;
						const Pixel c{ coord.x + x, coord.y + y };
						if(c.x < 0 || c.y < 0 || c.x >= m_outputBuffer.get_width() || c.y >= m_outputBuffer.get_height())
							continue;
						const auto index = c.x + c.y * m_outputBuffer.get_width();
						const bool neighborShadowed = m_penumbra[index * m_bytesPerPixel + bitIndex] & (1u << bitOffset);
						const bool neighborLit = m_penumbra[index * m_bytesPerPixel + bitIndex] & (1u << (bitOffset + 1u));
						isBorder = isBorder || (neighborLit && !neighborShadowed);
						isPenumbraTransition = isPenumbraTransition || (neighborLit && neighborShadowed);
						if(neighborLit) {
							averageRadiance += m_outputBuffer.template get<RadianceTarget>(c);
							++radianceCount;
						}
					}
				}

				// Those flags are only valid if we're pure shadow
				if(lit) {
					isBorder = false;
					isPenumbraTransition = false;
				}

				/*if(isBorder)
					m_outputBuffer.template set<PenumbraTarget>(coord, ei::Vec3{ 0.f, 0.8f, 0.f });
				if(isPenumbraTransition)
					m_outputBuffer.template set<PenumbraTarget>(coord, ei::Vec3{ 0.8f, 0.8f, 0.8f });*/
				if(radianceCount > 0u) {
					averageRadiance -= static_cast<float>(radianceCount) * m_outputBuffer.template get<RadianceTarget>(coord);
					averageRadiance *= 1.f / static_cast<float>(radianceCount);
					averageRadiance = ei::max(averageRadiance, ei::Vec3{ 0.f });
				}
				m_outputBuffer.template set<RadianceTransitionTarget>(coord, averageRadiance);
			}

			// Add importance for transition from umbra to penumbra
			if(isPenumbraTransition) {
				float averageImportance = 0.f;
				u32 impCount = 0u;

				// TODO: only the edge vertices!
				for(int i = 0; i < m_params.importanceIterations; ++i) {
					const auto shadowPrim = m_shadowPrims[i * m_outputBuffer.get_num_pixels() + pixel];
					if(!shadowPrim.hitId.is_valid())
						continue;

					const auto importance = get_luminance(averageRadiance) / ei::pow(1.f + shadowPrim.weight, 3.f);

					const auto lodIdx = m_sceneDesc.lodIndices[shadowPrim.hitId.instanceId];
					const auto& lod = m_sceneDesc.lods[lodIdx];
					const auto& polygon = lod.polygon;
					if(static_cast<u32>(shadowPrim.hitId.primId) < (polygon.numTriangles + polygon.numQuads)) {
						const bool isTriangle = static_cast<u32>(shadowPrim.hitId.primId) < polygon.numTriangles;
						const auto vertexOffset = isTriangle ? 0u : 3u * polygon.numTriangles;
						const auto vertexCount = isTriangle ? 3u : 4u;
						const auto primIdx = static_cast<u32>(shadowPrim.hitId.primId) - (isTriangle ? 0u : polygon.numTriangles);
						for(u32 i = 0u; i < vertexCount; ++i) {
							const auto vertexId = vertexOffset + vertexCount * primIdx + i;
							const auto vertexIdx = polygon.vertexIndices[vertexId];
							mAssert(vertexIdx < polygon.numVertices);
							cuda::atomic_add<Device::CPU>(m_importances[lodIdx][vertexIdx].viewImportance, importance);
						}
					}
					cuda::atomic_add<Device::CPU>(m_importanceSums[lodIdx].shadowImportance, importance);
					cuda::atomic_add<Device::CPU>(m_importanceSums[lodIdx].numSilhouettePixels, 1u);
					averageImportance += importance;
					++impCount;
				}
				m_outputBuffer.template contribute<PenumbraTarget>(coord, ei::Vec3{ 1.f, 1.f, 1.f });
				m_outputBuffer.template contribute<SilhouetteWeightTarget>(coord, averageImportance / static_cast<float>(impCount));
			}

			// Attribute importance
			/*if((shadowed && lit)) {// || isBorder) {
				const float importance = get_luminance(averageRadiance);
				if(isnan(importance))
					__debugbreak();
				// TODO: only the edge vertices!
				for(int i = 0; i < m_params.importanceIterations; ++i) {
					const auto shadowPrim = m_shadowPrims[i * m_outputBuffer.get_num_pixels() + pixel];
					if(!shadowPrim.hitId.is_valid())
						continue;

					const auto lodIdx = m_sceneDesc.lodIndices[shadowPrim.hitId.instanceId];
					const auto& lod = m_sceneDesc.lods[lodIdx];
					const auto& polygon = lod.polygon;
					if(static_cast<u32>(shadowPrim.hitId.primId) < (polygon.numTriangles + polygon.numQuads)) {
						const bool isTriangle = static_cast<u32>(shadowPrim.hitId.primId) < polygon.numTriangles;
						const auto vertexOffset = isTriangle ? 0u : 3u * polygon.numTriangles;
						const auto vertexCount = isTriangle ? 3u : 4u;
						const auto primIdx = static_cast<u32>(shadowPrim.hitId.primId) - (isTriangle ? 0u : polygon.numTriangles);
						for(u32 i = 0u; i < vertexCount; ++i) {
							const auto vertexId = vertexOffset + vertexCount * primIdx + i;
							const auto vertexIdx = polygon.vertexIndices[vertexId];
							mAssert(vertexIdx < polygon.numVertices);
							cuda::atomic_add<Device::CPU>(m_importances[lodIdx][vertexIdx].viewImportance, importance * shadowPrim.weight);
							cuda::atomic_add<Device::CPU>(m_importanceSums[lodIdx].shadowImportance, importance * shadowPrim.weight);
						}
					}
				}

				//m_outputBuffer.template contribute<SilhouetteWeightTarget>(coord, importance);
			}*/

		}
	}
}

void CpuSsSilPT::display_importance() {
	// TODO: disable the displaying after switching scenarios? May lead to crash
	const u32 NUM_PIXELS = m_outputBuffer.get_num_pixels();
#pragma PARALLEL_FOR
	for(int pixel = 0; pixel < (int)NUM_PIXELS; ++pixel) {
		const Pixel coord{ pixel % m_outputBuffer.get_width(), pixel / m_outputBuffer.get_width() };
		silhouette::sample_vis_importance(m_outputBuffer, m_sceneDesc, coord, m_rngs[pixel],
										  m_importances.data(), m_importanceSums.data(),
										  m_maxImportance == 0.f ? 1.f : m_maxImportance);
	}
}

void CpuSsSilPT::normalize_radiance() {
	// Since the output logic won't normalize the radiance
	// target by the iteration count we have to do it
	const u32 NUM_PIXELS = m_outputBuffer.get_num_pixels();
#pragma PARALLEL_FOR
	for(int pixel = 0; pixel < (int)NUM_PIXELS; ++pixel) {
		const Pixel coord{ pixel % m_outputBuffer.get_width(), pixel / m_outputBuffer.get_width() };
		const auto radiance = m_outputBuffer.template get<RadianceTarget>(coord);
		m_outputBuffer.template set<RadianceTarget>(coord, radiance * (1.f / static_cast<float>(m_params.importanceIterations)));
	}
}

void CpuSsSilPT::compute_max_importance() {
	m_maxImportance = 0.f;

	// Compute the maximum normalized importance for visualization
	for(i32 i = 0u; i < m_decimaters.size(); ++i)
		m_maxImportance = std::max(m_maxImportance, m_decimaters[i]->get_current_max_importance());
}

void CpuSsSilPT::initialize_decimaters() {
	auto& objects = m_currentScene->get_objects();
	m_decimaters.clear();
	m_decimaters.resize(objects.size());
	auto objIter = objects.begin();

	const auto timeBegin = CpuProfileState::get_process_time();
	m_importanceSums = std::vector<ss::DeviceImportanceSums<Device::CPU>>(m_decimaters.size());
	m_importances = std::vector<ArrayDevHandle_t<Device::CPU, ss::Importances<Device::CPU>>>(m_decimaters.size());

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
		m_decimaters[i] = std::make_unique<ss::ImportanceDecimater<Device::CPU>>(obj.first->get_name(), lod, newLod, collapses,
																				 m_params.viewWeight, m_params.lightWeight,
																				 m_params.shadowWeight, m_params.shadowSilhouetteWeight);
		m_importanceSums[i].shadowImportance.store(0.f);
		m_importanceSums[i].shadowSilhouetteImportance.store(0.f);
		// TODO: this reeeeally breaks instancing
		scene::WorldContainer::instance().get_current_scenario()->set_custom_lod(obj.first, newLodLevel);
	}

	logInfo("Initial decimation: ", std::chrono::duration_cast<std::chrono::milliseconds>(CpuProfileState::get_process_time() - timeBegin).count(), "ms");
}

void CpuSsSilPT::update_reduction_factors() {
	m_remainingVertexFactor.clear();
	if(m_params.reduction == 0.f) {
		// Do not reduce anything
		for(std::size_t i = 0u; i < m_decimaters.size(); ++i) {
			ss::ImportanceSums sums{ m_importanceSums[i].shadowImportance, m_importanceSums[i].shadowSilhouetteImportance };
			m_decimaters[i]->update_importance_density(sums);
			m_remainingVertexFactor.push_back(1.0);
		}
		return;
	}

	double expectedVertexCount = 0.0;
	for(std::size_t i = 0u; i < m_decimaters.size(); ++i) {
		auto& decimater = m_decimaters[i];
		ss::ImportanceSums sums{ m_importanceSums[i].shadowImportance, m_importanceSums[i].shadowSilhouetteImportance };
		m_decimaters[i]->update_importance_density(sums);
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

void CpuSsSilPT::init_rngs(int num) {
	m_rngs.resize(num);
	// TODO: incude some global seed into the initialization
	for(int i = 0; i < num; ++i)
		m_rngs[i] = math::Rng(i);
}

} // namespace mufflon::renderer::decimaters::silhouette