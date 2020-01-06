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

void CpuSsSilPT::iterate() {
	auto scope = Profiler::core().start<CpuProfileState>("CPU PT iteration", ProfileLevel::HIGH);

	m_sceneDesc.lightTree.posGuide = m_params.neeUsePositionGuide;
	const auto NUM_PIXELS = m_outputBuffer.get_num_pixels();
#pragma PARALLEL_FOR
	for(int pixel = 0; pixel < (int)NUM_PIXELS; ++pixel) {
		const Pixel coord{ pixel % m_outputBuffer.get_width(), pixel / m_outputBuffer.get_width() };
		silhouette::sample_importance(m_outputBuffer, m_sceneDesc, m_params, coord, m_rngs[pixel],
									  &m_shadowStatus[pixel * m_lightCount * m_params.maxPathLength]);
	}

	// Post-processing
#pragma PARALLEL_FOR
	for(int pixel = 0; pixel < (int)NUM_PIXELS; ++pixel) {
		const Pixel coord{ pixel % m_outputBuffer.get_width(), pixel / m_outputBuffer.get_width() };

		post_process_shadow(m_outputBuffer, m_sceneDesc, m_params, coord, pixel,
							m_currentIteration, m_shadowStatus.get());
	}
}

void CpuSsSilPT::post_reset() {
	init_rngs(m_outputBuffer.get_num_pixels());

	// We always account for the background, even if it may be black
	m_lightCount = 1u + m_sceneDesc.lightTree.posLights.lightCount + m_sceneDesc.lightTree.dirLights.lightCount;
	const auto statusCount = m_params.maxPathLength * m_lightCount
		* static_cast<std::size_t>(m_outputBuffer.get_num_pixels());
	m_shadowStatus = make_udevptr_array<Device::CPU, ss::ShadowStatus, false>(statusCount);
	std::memset(m_shadowStatus.get(), 0, sizeof(ss::ShadowStatus) * statusCount);
}

#if 0
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
		for(std::size_t l = 0u; l < lightCount; ++l) {
			// "Detect" penumbra
			const auto bitIndex = l / 4u;
			const auto bitOffset = 2u * (l % 4u);
			const bool shadowed = penumbraBits[bitIndex] & (1u << bitOffset);
			const bool lit = penumbraBits[bitIndex] & (1u << (bitOffset + 1u));

			m_outputBuffer.template contribute<PenumbraTarget>(coord, contributePenumbraColor(l, shadowed, lit));

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
			if(isPenumbraTransition || isBorder) {
				float averageImportance = 0.f;
				u32 impCount = 0u;

				// TODO: only the edge vertices!
				for(int i = 0; i < m_params.importanceIterations; ++i) {
					const auto shadowPrim = m_shadowPrims[i * m_outputBuffer.get_num_pixels() + pixel];
					if(!shadowPrim.hitId.is_valid())
						continue;

					float importance;
					switch(m_params.penumbraWeight) {
						case PPenumbraWeight::Values::LDIVDP3:
							importance = shadowPrim.weight;
							break;
						case PPenumbraWeight::Values::SMEAR:
							importance = get_luminance(averageRadiance) / (1.f + shadowPrim.weight);
							break;
						case PPenumbraWeight::Values::SMEARDIV2:
							importance = get_luminance(averageRadiance) / ei::pow(1.f + shadowPrim.weight, 2.f);
							break;
						case PPenumbraWeight::Values::SMEARDIV3:
							importance = get_luminance(averageRadiance) / ei::pow(1.f + shadowPrim.weight, 3.f);
							break;
					}

					const auto lodIdx = m_sceneDesc.lodIndices[shadowPrim.hitId.instanceId];
					const auto& lod = m_sceneDesc.lods[lodIdx];
					const auto& polygon = lod.polygon;
					if(static_cast<u32>(shadowPrim.hitId.primId) < (polygon.numTriangles + polygon.numQuads)) {
						const bool isTriangle = static_cast<u32>(shadowPrim.hitId.primId) < polygon.numTriangles;
						const auto vertexOffset = isTriangle ? 0u : 3u * polygon.numTriangles;
						const auto vertexCount = isTriangle ? 3u : 4u;
						const auto primIdx = static_cast<u32>(shadowPrim.hitId.primId) - (isTriangle ? 0u : polygon.numTriangles);
						for(u32 v = 0u; v < vertexCount; ++v) {
							const auto vertexId = vertexOffset + vertexCount * primIdx + v;
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
		}
	}
}
#endif

void CpuSsSilPT::init_rngs(int num) {
	m_rngs.resize(num);
	// TODO: incude some global seed into the initialization
	for(int i = 0; i < num; ++i)
		m_rngs[i] = math::Rng(i);
}

} // namespace mufflon::renderer::decimaters::silhouette