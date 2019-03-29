#pragma once

#include "cpu_silhouette.hpp"
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
#include <stdexcept>
#include <queue>

namespace mufflon::renderer::decimaters {

using namespace silhouette;

namespace {

struct LightData {
	scene::lights::LightType type;
	float flux;
	u32 offset;
};

inline LightData get_light_data(const u32 lightIdx, const scene::lights::LightSubTree& tree) {
	// Special case for only single light
	if(tree.lightCount == 1) {
		return LightData{
			static_cast<scene::lights::LightType>(tree.root.type),
			tree.root.flux,
			0u
		};
	}

	// Determine what level of the tree the light is on
	const u32 level = std::numeric_limits<u32>::digits - 1u - static_cast<u32>(std::log2(tree.internalNodeCount + lightIdx + 1u));
	// Determine the light's node index within its level
	const u32 levelIndex = (tree.internalNodeCount + lightIdx) - ((1u << level) - 1u);
	// The corresponding parent node's level index is then the level index / 2
	const u32 parentLevelIndex = levelIndex / 2u;
	// Finally, compute the tree index of the node
	const u32 parentIndex = (1u << (level - 1u)) - 1u + parentLevelIndex;
	const scene::lights::LightSubTree::Node& node = *tree.get_node(parentIndex * sizeof(scene::lights::LightSubTree::Node));

	// Left vs. right node
	// TODO: for better divergence let all even indices be processes first, then the uneven ones
	if(levelIndex % 2 == 0) {
		mAssert(node.left.type < static_cast<u16>(scene::lights::LightType::NUM_LIGHTS));
		return LightData{
			static_cast<scene::lights::LightType>(node.left.type),
			node.left.flux, node.left.offset
		};
	} else {
		mAssert(node.right.type < static_cast<u16>(scene::lights::LightType::NUM_LIGHTS));
		return LightData{
			static_cast<scene::lights::LightType>(node.right.type),
			node.right.flux, node.right.offset
		};
	}
}

inline float get_luminance(const ei::Vec3& vec) {
	constexpr ei::Vec3 LUM_WEIGHT{ 0.212671f, 0.715160f, 0.072169f };
	return ei::dot(LUM_WEIGHT, vec);
}

inline std::string pretty_print_size(u32 bytes) {
	std::string str;

	const u32 decimals = 1u + static_cast<u32>(std::log10(bytes));
	const u32 sizeBox = decimals / 3u;
	const float val = static_cast<float>(bytes) / std::pow(10.f, static_cast<float>(sizeBox) * 3.f);
	StringView suffix;


	switch(sizeBox) {
		case 0u: suffix = "bytes"; break;
		case 1u: suffix = "KBytes"; break;
		case 2u: suffix = "MBytes"; break;
		case 3u: suffix = "GBytes"; break;
		case 4u: suffix = "TBytes"; break;
		default: suffix = "?Bytes"; break;
	}
	const u32 numberSize = (decimals - sizeBox * 3u) + 4u;
	str.resize(numberSize + suffix.size());

	std::snprintf(str.data(), str.size(), "%.2f ", val);
	// Gotta print the suffix separately to overwrite the terminating '\0'
	std::strncpy(str.data() + numberSize, suffix.data(), suffix.size());
	return str;
}

u32 get_lod_memory(const scene::Lod& lod) {
	const auto& polygons = lod.template get_geometry<scene::geometry::Polygons>();
	return static_cast<u32>(polygons.get_triangle_count() * 3u * sizeof(u32)
							+ polygons.get_quad_count() * 3u * sizeof(u32)
							+ polygons.get_vertex_count() * (2u * sizeof(ei::Vec3) + sizeof(ei::Vec2)));
}

u32 get_vertices_for_memory(const u32 memory) {
	// Assumes that one "additional" vertex (uncollapsed) results in one "extra" edge and two "extra" triangles
	return memory / (2u * 3u * sizeof(u32) + 2u * sizeof(ei::Vec3) + sizeof(ei::Vec2));
}

} // namespace

CpuShadowSilhouettes::CpuShadowSilhouettes()
{
	// TODO: init one RNG per thread?
	std::random_device rndDev;
	m_rngs.emplace_back(static_cast<u32>(rndDev()));
}

void CpuShadowSilhouettes::on_scene_load() {
	if(m_params.resetOnReload) {
		m_currentDecimationIteration = 0u;
	} else if(m_currentDecimationIteration != 0u) {
		// At least activate the created LoDs
		for(auto& obj : m_currentScene->get_objects()) {
			if(obj.second.size() != 1u)
				throw std::runtime_error("We cannot deal with instancing yet");

			const u32 newLodLevel = static_cast<u32>(obj.first->get_lod_slot_count() - 1u);
			// TODO: this reeeeally breaks instancing
			for(scene::InstanceHandle inst : obj.second) {
				// Modify the scenario to use this lod instead
				scene::WorldContainer::instance().get_current_scenario()->set_custom_lod(inst, newLodLevel);
			}
		}
	}
}

void CpuShadowSilhouettes::on_scene_unload() {
	m_decimaters.clear();
	m_currentDecimationIteration = 0u;
}

bool CpuShadowSilhouettes::pre_iteration(OutputHandler& outputBuffer) {	
	return RendererBase<Device::CPU>::pre_iteration(outputBuffer);
}

void CpuShadowSilhouettes::post_iteration(OutputHandler& outputBuffer) {
	if((int)m_currentDecimationIteration == m_params.decimationIterations) {
		// Finalize the decimation process
		logInfo("Finished decimation process");
		++m_currentDecimationIteration;
		m_reset = true;
		// TODO
	} else if((int)m_currentDecimationIteration < m_params.decimationIterations) {
		// TODO: compute threshold!
		constexpr float threshold = 20.0f;

		logInfo("Performing decimation/undecimation iteration");
		const auto processTime = CpuProfileState::get_process_time();
		const auto cycles = CpuProfileState::get_cpu_cycle();
		auto scope = Profiler::instance().start<CpuProfileState>("Silhouette decimation");
		for(std::size_t i = 0u; i < m_decimaters.size(); ++i) {
			m_decimaters[i].iterate(static_cast<std::size_t>(m_params.threshold), threshold, (float)(1.0 - m_remainingVertexFactor[i]));
		}
		logPedantic("Duration: ", std::chrono::duration_cast<std::chrono::milliseconds>(CpuProfileState::get_process_time() - processTime).count(),
					"ms, ", (CpuProfileState::get_cpu_cycle() - cycles) / 1'000'000, " MCycles");

		m_currentScene->clear_accel_structure();
		m_reset = true;
		++m_currentDecimationIteration;
	}
	RendererBase<Device::CPU>::post_iteration(outputBuffer);
}

void CpuShadowSilhouettes::pre_descriptor_requery() {
	init_rngs(m_outputBuffer.get_num_pixels());

	// Initialize the decimaters
	// TODO: how to deal with instancing
	if(m_currentDecimationIteration == 0u)
		this->initialize_decimaters();
}

void CpuShadowSilhouettes::iterate() {
	if((int)m_currentDecimationIteration < m_params.decimationIterations) {
		logInfo("Starting decimation iteration (", m_currentDecimationIteration + 1, " of ", m_params.decimationIterations, ")");
		gather_importance();

		if(m_decimaters.size() == 0u)
			return;

		// We need to update the importance density
		this->update_reduction_factors();

		if(m_params.importanceIterations > 0 && m_outputBuffer.is_target_enabled(RenderTargets::IMPORTANCE)) {
			if(m_params.displayProjection) {
				// For debugging: display the original mesh instead of the LoD
				for(auto& obj : m_currentScene->get_objects()) {
					auto& lod = obj.first->get_lod(0u);
					const auto& polygons = lod.template get_geometry<scene::geometry::Polygons>();
					for(scene::InstanceHandle inst : obj.second) 
						scene::WorldContainer::instance().get_current_scenario()->set_custom_lod(inst, 0u);
				}
				m_currentScene->clear_accel_structure();
				m_sceneDesc = m_currentScene->get_descriptor<Device::CPU>({}, {}, {}, m_outputBuffer.get_resolution());
			}

			compute_max_importance();
			logInfo("Max. importance: ", m_maxImportance);
			display_importance();

			if(m_params.displayProjection) {
				// For debugging: undo the LoD change
				for(auto& obj : m_currentScene->get_objects()) {
					auto& lod = obj.first->get_lod(0u);
					const auto& polygons = lod.template get_geometry<scene::geometry::Polygons>();
					for(scene::InstanceHandle inst : obj.second)
						scene::WorldContainer::instance().get_current_scenario()->set_custom_lod(inst, static_cast<u32>(obj.first->get_lod_slot_count() - 1u));
				}
				m_currentScene->clear_accel_structure();
				m_sceneDesc = m_currentScene->get_descriptor<Device::CPU>({}, {}, {}, m_outputBuffer.get_resolution());
			}

		}
	} else {
		const u32 NUM_PIXELS = m_outputBuffer.get_num_pixels();
#pragma PARALLEL_FOR
		for(int i = 0; i < (int)NUM_PIXELS; ++i) {
			this->pt_sample(Pixel{ i % m_outputBuffer.get_width(), i / m_outputBuffer.get_width() });
		}
	}
}


void CpuShadowSilhouettes::gather_importance() {
	const u32 NUM_PIXELS = m_outputBuffer.get_num_pixels();
#pragma PARALLEL_FOR
	for(int i = 0; i < m_params.importanceIterations * (int)NUM_PIXELS; ++i) {
		const int pixel = i / m_params.importanceIterations;
		this->importance_sample(Pixel{ pixel % m_outputBuffer.get_width(), pixel / m_outputBuffer.get_width() });
	}
	// TODO: allow for this with proper reset "events"
}

void CpuShadowSilhouettes::importance_sample(const Pixel coord) {
	int pixel = coord.x + coord.y * m_outputBuffer.get_width();

	math::Throughput throughput{ ei::Vec3{1.0f}, 1.0f };
	// We gotta keep track of our vertices
	thread_local std::vector<SilPathVertex> vertices(std::max(2, m_params.maxPathLength + 1));
	// Create a start for the path
	(void)SilPathVertex::create_camera(&vertices.front(), &vertices.front(), m_sceneDesc.camera.get(), coord, m_rngs[pixel].next());

	float sharpness = 1.f;
	
	// Andreas' algorithm mapped to path tracing:
	// Increasing importance for photons is equivalent to increasing
	// importance by the irradiance. Thus we need to compute "both
	// direct and indirect" irradiance for the path tracer as well.
	// They differ, however, in the types of paths that they
	// preferably find.

	int pathLen = 0;
	do {
		vertices[pathLen].ext().pathRadiance = ei::Vec3{ 0.f };
		// Add direct contribution as importance as well
		if(pathLen > 0 && pathLen + 1 <= m_params.maxPathLength) {
			u64 neeSeed = m_rngs[pixel].next();
			math::RndSet2 neeRnd = m_rngs[pixel].next();
			auto nee = connect(m_sceneDesc.lightTree, 0, 1, neeSeed,
							   vertices[pathLen].get_position(), m_currentScene->get_bounding_box(),
							   neeRnd, scene::lights::guide_flux);
			Pixel projCoord;
			auto value = vertices[pathLen].evaluate(nee.direction, m_sceneDesc.media, projCoord);
			mAssert(!isnan(value.value.x) && !isnan(value.value.y) && !isnan(value.value.z));
			Spectrum radiance = value.value * nee.diffIrradiance;
			if(any(greater(radiance, 0.0f)) && value.cosOut > 0.0f) {
				vertices[pathLen].ext().shadowRay = ei::Ray{ nee.lightPoint, -nee.direction };
				vertices[pathLen].ext().lightDistance = nee.dist;

				auto shadowHit = scene::accel_struct::first_intersection(m_sceneDesc, vertices[pathLen].ext().shadowRay,
																		 vertices[pathLen].get_primitive_id(), nee.dist);
				vertices[pathLen].ext().shadowHit = shadowHit.hitId;
				vertices[pathLen].ext().firstShadowDistance = shadowHit.hitT;
				AreaPdf hitPdf = value.pdfF.to_area_pdf(nee.cosOut, nee.distSq);
				float mis = 1.0f / (1.0f + hitPdf / nee.creationPdf);
				vertices[pathLen].ext().pathRadiance = mis * radiance * value.cosOut;
				if(shadowHit.hitId.instanceId < 0 && m_params.enableDirectImportance) {
					mAssert(!isnan(mis));
					// Save the radiance for the later indirect lighting computation
					// Compute how much radiance arrives at the previous vertex from the direct illumination
					// Add the importance
					const ei::Vec3 irradiance = nee.diffIrradiance * value.cosOut; // [W/m²]
					const float weightedIrradianceLuminance = mis * get_luminance(throughput.weight * irradiance) * (1.f - ei::abs(vertices[pathLen - 1].ext().outCos));

					const auto& hitId = vertices[pathLen].get_primitive_id();
					const auto& lod = m_sceneDesc.lods[m_sceneDesc.lodIndices[hitId.instanceId]];
					const u32 numVertices = hitId.primId < (i32)lod.polygon.numTriangles ? 3u : 4u;
					const u32 vertexOffset = hitId.primId < (i32)lod.polygon.numTriangles ? 0u : 3u * lod.polygon.numTriangles;
					m_decimaters[m_sceneDesc.lodIndices[hitId.instanceId]].record_face_contribution(&lod.polygon.vertexIndices[vertexOffset + numVertices * hitId.primId],
																									numVertices, vertices[pathLen].get_position(),
																									weightedIrradianceLuminance);
				}
			}
		}

		// Walk
		scene::Point lastPosition = vertices[pathLen].get_position();
		math::RndSet2_1 rnd{ m_rngs[pixel].next(), m_rngs[pixel].next() };
		VertexSample sample;

		if(!walk(m_sceneDesc, vertices[pathLen], rnd, -1.0f, false, throughput, vertices[pathLen + 1], sample))
			break;

		// Update old vertex with accumulated throughput
		vertices[pathLen].ext().updateBxdf(sample, throughput);

		// Fetch the relevant information for attributing the instance to the correct vertices
		const auto& hitId = vertices[pathLen + 1].get_primitive_id();
		const auto& lod = m_sceneDesc.lods[m_sceneDesc.lodIndices[hitId.instanceId]];
		const u32 numVertices = hitId.primId < (i32)lod.polygon.numTriangles ? 3u : 4u;
		const u32 vertexOffset = hitId.primId < (i32)lod.polygon.numTriangles ? 0u : 3u * lod.polygon.numTriangles;

		if(pathLen > 0 && m_params.enableEyeImportance) {
			// Direct hits are being scaled down in importance by a sigmoid of the BxDF to get an idea of the "sharpness"
			const float importance = sharpness * (1.f - ei::abs(ei::dot(vertices[pathLen].get_normal(), vertices[pathLen].get_incident_direction())));
			m_decimaters[m_sceneDesc.lodIndices[hitId.instanceId]].record_face_contribution(&lod.polygon.vertexIndices[vertexOffset + numVertices * hitId.primId],
																							numVertices, vertices[pathLen + 1].get_position(), importance);

			const ei::Vec3 bxdf = vertices[pathLen].ext().bxdfPdf * (float)vertices[pathLen].ext().pdf;
			sharpness *= 2.f / (1.f + ei::exp(-get_luminance(bxdf))) - 1.f;
		}

		++pathLen;
	} while(pathLen < m_params.maxPathLength);

	// Go back over the path and add up the irradiance from indirect illumination
	if (m_params.enableIndirectImportance) {
		ei::Vec3 accumRadiance{ 0.f };
		float accumThroughout = 1.f;
		for (int p = pathLen - 2; p >= 1; --p) {
			accumRadiance = vertices[p].ext().throughput * accumRadiance + (vertices[p + 1].ext().shadowHit.instanceId < 0 ?
				vertices[p + 1].ext().pathRadiance : ei::Vec3{ 0.f });
			const ei::Vec3 irradiance = vertices[p].ext().accumThroughput * vertices[p].ext().outCos * accumRadiance;

			const auto& hitId = vertices[p].get_primitive_id();
			const auto* lod = &m_sceneDesc.lods[m_sceneDesc.lodIndices[hitId.instanceId]];
			const u32 numVertices = hitId.primId < (i32)lod->polygon.numTriangles ? 3u : 4u;
			const u32 vertexOffset = hitId.primId < (i32)lod->polygon.numTriangles ? 0u : 3u * lod->polygon.numTriangles;

			const float importance = get_luminance(irradiance) * (1.f - ei::abs(vertices[p].ext().outCos));
			m_decimaters[m_sceneDesc.lodIndices[hitId.instanceId]].record_face_contribution(&lod->polygon.vertexIndices[vertexOffset + numVertices * hitId.primId],
																							numVertices, vertices[p].get_position(), importance);

			// TODO: store accumulated sharpness
			// Check if it is sensible to keep shadow silhouettes intact
			// TODO: replace threshold with something sensible
			if (p == 1 && vertices[p].ext().shadowHit.instanceId >= 0) {
				const float indirectLuminance = get_luminance(accumRadiance);
				const float totalLuminance = get_luminance(vertices[p].ext().pathRadiance) + indirectLuminance;
				const float ratio = totalLuminance / indirectLuminance - 1.f;
				if (ratio > 0.02f) {
					constexpr float DIST_EPSILON = 0.000125f;
					constexpr float FACTOR = 2000'000'000.f;
					// TODO: proper factor!
					trace_shadow_silhouette(vertices[p].ext().shadowRay, vertices[p], FACTOR * (totalLuminance - indirectLuminance));

					// Also add importance if we're at the edge of the viewport
					if(coord.x == 0 || coord.x == m_outputBuffer.get_resolution().x - 1
					   || coord.y == 0 || coord.y == m_outputBuffer.get_resolution().y - 1) {
						m_decimaters[m_sceneDesc.lodIndices[hitId.instanceId]].record_face_contribution(&lod->polygon.vertexIndices[vertexOffset + numVertices * hitId.primId],
																										numVertices, vertices[p].get_position(),
																										FACTOR * (totalLuminance - indirectLuminance));
					}
				}
			}
		}
	}
}

void CpuShadowSilhouettes::pt_sample(const Pixel coord) {
	int pixel = coord.x + coord.y * m_outputBuffer.get_width();

	auto& rng = m_rngs[pixel];
	math::Throughput throughput{ ei::Vec3{1.0f}, 1.0f };
	PtPathVertex vertex;
	VertexSample sample;
	// Create a start for the path
	PtPathVertex::create_camera(&vertex, &vertex, m_sceneDesc.camera.get(), coord, rng.next());

	int pathLen = 0;
	do {
		if(pathLen > 0 && pathLen + 1 <= m_params.maxPathLength) {
			// Call NEE member function for recursive vertices.
			// Do not connect to the camera, because this makes the renderer much more
			// complicated. Our decision: The PT should be as simple as possible!
			// What means more complicated?
			// A connnection to the camera results in a different pixel. In a multithreaded
			// environment this means that we need a write mutex for each pixel.
			// TODO: test/parametrize mulievent estimation (more indices in connect) and different guides.
			u64 neeSeed = rng.next();
			math::RndSet2 neeRnd = rng.next();
			auto nee = connect(m_sceneDesc.lightTree, 0, 1, neeSeed,
							   vertex.get_position(), m_sceneDesc.aabb,
							   neeRnd, scene::lights::guide_flux);
			Pixel outCoord;
			auto value = vertex.evaluate(nee.direction, m_sceneDesc.media, outCoord);
			if(nee.cosOut != 0) value.cosOut *= nee.cosOut;
			mAssert(!isnan(value.value.x) && !isnan(value.value.y) && !isnan(value.value.z));
			Spectrum radiance = value.value * nee.diffIrradiance;
			if(any(greater(radiance, 0.0f)) && value.cosOut > 0.0f) {
				bool anyhit = scene::accel_struct::any_intersection(
					m_sceneDesc, { vertex.get_position(), nee.direction },
					vertex.get_primitive_id(), nee.dist);
				if(!anyhit) {
					AreaPdf hitPdf = value.pdfF.to_area_pdf(nee.cosOut, nee.distSq);
					float mis = 1.0f / (1.0f + hitPdf / nee.creationPdf);
					mAssert(!isnan(mis));
					m_outputBuffer.contribute(coord, throughput, { Spectrum{1.0f}, 1.0f },
											value.cosOut, radiance * mis);
				}
			}
		}

		// Walk
		scene::Point lastPosition = vertex.get_position();
		math::RndSet2_1 rnd{ rng.next(), rng.next() };
		if(!walk(m_sceneDesc, vertex, rnd, -1.0f, false, throughput, vertex, sample)) {
			if(throughput.weight != Spectrum{ 0.f }) {
				// Missed scene - sample background
				auto background = evaluate_background(m_sceneDesc.lightTree.background, sample.excident);
				if(any(greater(background.value, 0.0f))) {
					float mis = 1.0f / (1.0f + background.pdfB / sample.pdfF);
					background.value *= mis;
					m_outputBuffer.contribute(coord, throughput, background.value,
											  ei::Vec3{ 0, 0, 0 }, ei::Vec3{ 0, 0, 0 },
											  ei::Vec3{ 0, 0, 0 });
				}
			}
			break;
		}
		++pathLen;

		// Query importance if the target is active
		if(pathLen == 1 && m_params.importanceIterations > 0 && m_outputBuffer.is_target_enabled(RenderTargets::IMPORTANCE)) {
			const auto& decimater = m_decimaters[m_sceneDesc.lodIndices[vertex.get_primitive_id().instanceId]];
			const bool isSilhouette = decimater.is_current_marked_as_silhouette(vertex.get_primitive_id().primId);
			m_outputBuffer.contribute(coord, RenderTargets::IMPORTANCE, ei::Vec4{ query_importance(vertex.get_position(), vertex.get_primitive_id()) });
			m_outputBuffer.contribute(coord, RenderTargets::SHADOW_SILHOUETTE, ei::Vec4{ isSilhouette ? 1.f : 0.f });
		}

		// Evaluate direct hit of area ligths
		Spectrum emission = vertex.get_emission().value;
		if(emission != 0.0f) {
			AreaPdf backwardPdf = connect_pdf(m_sceneDesc.lightTree, vertex.get_primitive_id(),
												vertex.get_surface_params(),
												lastPosition, scene::lights::guide_flux);
			float mis = pathLen == 1 ? 1.0f
				: 1.0f / (1.0f + backwardPdf / vertex.ext().incidentPdf);
			emission *= mis;
		}
		m_outputBuffer.contribute(coord, throughput, emission, vertex.get_position(),
								  vertex.get_normal(), vertex.get_albedo());
	} while(pathLen < m_params.maxPathLength);
}

void CpuShadowSilhouettes::compute_max_importance() {
	m_maxImportance = 0.f;

	// Compute the maximum normalized importance for visualization
//#pragma omp parallel for reduction(max:m_maxImportance)
	for(i32 i = 0u; i < m_sceneDesc.numInstances; ++i) {
		if(m_params.displayProjection && m_currentDecimationIteration < static_cast<u32>(m_params.decimationIterations))
			m_maxImportance = std::max(m_maxImportance, m_decimaters[m_sceneDesc.lodIndices[i]].get_mapped_max_importance());
		else
			m_maxImportance = std::max(m_maxImportance, m_decimaters[m_sceneDesc.lodIndices[i]].get_current_max_importance());
	}
}

void CpuShadowSilhouettes::display_importance() {
	const u32 NUM_PIXELS = m_outputBuffer.get_num_pixels();
#pragma PARALLEL_FOR
	for(int pixel = 0; pixel < (int)NUM_PIXELS; ++pixel) {
		const ei::IVec2 coord{ pixel % m_outputBuffer.get_width(), pixel / m_outputBuffer.get_width() };
		//m_params.maxPathLength = 2;

		math::Throughput throughput{ ei::Vec3{1.0f}, 1.0f };
		PtPathVertex vertex;
		// Create a start for the path
		(void)PtPathVertex::create_camera(&vertex, &vertex, m_sceneDesc.camera.get(), coord, m_rngs[pixel].next());

		// Walk
		scene::Point lastPosition = vertex.get_position();
		math::RndSet2_1 rnd{ m_rngs[pixel].next(), m_rngs[pixel].next() };
		VertexSample sample;
		if(walk(m_sceneDesc, vertex, rnd, -1.0f, false, throughput, vertex, sample)) {
			const auto& decimater = m_decimaters[m_sceneDesc.lodIndices[vertex.get_primitive_id().instanceId]];
			const bool isSilhouette = m_params.displayProjection ? decimater.is_original_marked_as_silhouette(vertex.get_primitive_id().primId)
				: decimater.is_current_marked_as_silhouette(vertex.get_primitive_id().primId);
			m_outputBuffer.contribute(coord, RenderTargets::IMPORTANCE, ei::Vec4{ query_importance(vertex.get_position(), vertex.get_primitive_id()) });
			m_outputBuffer.contribute(coord, RenderTargets::SHADOW_SILHOUETTE, ei::Vec4{ isSilhouette ? 1.f : 0.f });
		}
	}

}

float CpuShadowSilhouettes::query_importance(const ei::Vec3& hitPoint, const scene::PrimitiveHandle& hitId) {
	// TODO: density or importance?
	if(m_params.displayProjection && m_currentDecimationIteration < static_cast<u32>(m_params.decimationIterations))
		return m_decimaters[m_sceneDesc.lodIndices[hitId.instanceId]].get_mapped_importance(hitId.primId, hitPoint) / m_maxImportance;
	else
		return m_decimaters[m_sceneDesc.lodIndices[hitId.instanceId]].get_current_importance(hitId.primId, hitPoint)/ m_maxImportance;
}

bool CpuShadowSilhouettes::trace_shadow_silhouette(const ei::Ray& shadowRay, const SilPathVertex& vertex, const float importance) {
	constexpr float DIST_EPSILON = 0.001f;

	// TODO: worry about spheres?
	const ei::Ray backfaceRay{ shadowRay.origin + shadowRay.direction * (vertex.ext().firstShadowDistance + DIST_EPSILON), shadowRay.direction };

	const auto secondHit = scene::accel_struct::first_intersection(m_sceneDesc, backfaceRay, vertex.ext().shadowHit,
																   vertex.ext().lightDistance - vertex.ext().firstShadowDistance + DIST_EPSILON);
	// We absolutely have to have a second hit - either us (since we weren't first hit) or something else
	if(secondHit.hitId.instanceId >= 0 && secondHit.hitId != vertex.get_primitive_id()
	   && secondHit.hitId.instanceId == vertex.ext().shadowHit.instanceId) {
		// Check for silhouette - get the vertex indices of the primitives
		const auto& obj = m_sceneDesc.lods[m_sceneDesc.lodIndices[vertex.ext().shadowHit.instanceId]];
		const i32 firstNumVertices = vertex.ext().shadowHit.primId < (i32)obj.polygon.numTriangles ? 3 : 4;
		const i32 secondNumVertices = secondHit.hitId.primId < (i32)obj.polygon.numTriangles ? 3 : 4;
		const i32 firstPrimIndex = vertex.ext().shadowHit.primId - (vertex.ext().shadowHit.primId < (i32)obj.polygon.numTriangles
													  ? 0 : (i32)obj.polygon.numTriangles);
		const i32 secondPrimIndex = secondHit.hitId.primId - (secondHit.hitId.primId < (i32)obj.polygon.numTriangles
															  ? 0 : (i32)obj.polygon.numTriangles);
		const i32 firstVertOffset = vertex.ext().shadowHit.primId < (i32)obj.polygon.numTriangles ? 0 : 3 * (i32)obj.polygon.numTriangles;
		const i32 secondVertOffset = secondHit.hitId.primId < (i32)obj.polygon.numTriangles ? 0 : 3 * (i32)obj.polygon.numTriangles;

		// Check if we have "shared" vertices: cannot do it by index, since they might be
		// split vertices, but instead need to go by proximity
		i32 sharedVertices = 0;
		i32 edgeIdxFirst[2];
		i32 edgeIdxSecond[2];
		for(i32 i0 = 0; i0 < firstNumVertices; ++i0) {
			for(i32 i1 = 0; i1 < secondNumVertices; ++i1) {
				const i32 idx0 = obj.polygon.vertexIndices[firstVertOffset + firstNumVertices * firstPrimIndex + i0];
				const i32 idx1 = obj.polygon.vertexIndices[secondVertOffset + secondNumVertices * secondPrimIndex + i1];
				const ei::Vec3& p0 = obj.polygon.vertices[idx0];
				const ei::Vec3& p1 = obj.polygon.vertices[idx1];
				if(idx0 == idx1 || p0 == p1) {
					edgeIdxFirst[sharedVertices] = idx0;
					edgeIdxSecond[sharedVertices] = idx1;
					++sharedVertices;
				}
				if(sharedVertices >= 2)
					break;
			}
		}

		if(sharedVertices >= 1) {
			// Got at least a silhouette point - now make sure we're seeing the silhouette
			const ei::Ray silhouetteRay{ shadowRay.origin + shadowRay.direction * (vertex.ext().firstShadowDistance + secondHit.hitT), shadowRay.direction };

			const auto thirdHit = scene::accel_struct::first_intersection(m_sceneDesc, silhouetteRay, secondHit.hitId,
																		  vertex.ext().lightDistance - vertex.ext().firstShadowDistance - secondHit.hitT + DIST_EPSILON);
			if(thirdHit.hitId == vertex.get_primitive_id()) {
				for(i32 i = 0; i < sharedVertices; ++i) {
					const auto ab = obj.polygon.vertices[obj.polygon.vertexIndices[firstVertOffset + firstNumVertices * firstPrimIndex + 0]]
						- obj.polygon.vertices[obj.polygon.vertexIndices[firstVertOffset + firstNumVertices * firstPrimIndex + 1]];;
					const auto ac = obj.polygon.vertices[obj.polygon.vertexIndices[firstVertOffset + firstNumVertices * firstPrimIndex + 0]]
						- obj.polygon.vertices[obj.polygon.vertexIndices[firstVertOffset + firstNumVertices * firstPrimIndex + 2]];;
					const auto area = ei::len(ei::cross(ab, ac));
					const float solidAngle = area / (vertex.ext().lightDistance*vertex.ext().lightDistance);

					// x86_64 doesn't support atomic_fetch_add for floats FeelsBadMan
					const auto lodIdx = m_sceneDesc.lodIndices[vertex.ext().shadowHit.instanceId];
					m_decimaters[lodIdx].record_silhouette_vertex_contribution(edgeIdxFirst[i], importance / solidAngle);
					m_decimaters[lodIdx].record_silhouette_vertex_contribution(edgeIdxSecond[i], importance);
				}
				return true;
			} else {
				mAssert(thirdHit.hitId.instanceId >= 0);
				// TODO: store a shadow photon?
			}
		}
	}
	return false;
}

u32 CpuShadowSilhouettes::get_memory_requirement() const {
	u32 memory = 0u;

	for(auto& obj : m_currentScene->get_objects()) {
		mAssert(obj.first != nullptr);
		mAssert(obj.second.size() != 0u);

		if(obj.second.size() != 1u)
			throw std::runtime_error("We cannot deal with instancing yet");

		for(scene::InstanceHandle inst : obj.second) {
			mAssert(inst != nullptr);
			const u32 instanceLod = scene::WorldContainer::instance().get_current_scenario()->get_effective_lod(inst);
			mAssert(obj.first->has_lod_available(instanceLod));
			const auto& lod = obj.first->get_lod(instanceLod);

			const u32 lodMemory = get_lod_memory(lod);
			memory += lodMemory;
		}
	}

	return memory;
}

void CpuShadowSilhouettes::initialize_decimaters() {
	m_decimaters.clear();
	// Request status once to remember which vertices we deleted

	// TODO: clean temporary LoDs!

	for(auto& obj : m_currentScene->get_objects()) {
		if(obj.second.size() != 1u)
			throw std::runtime_error("We cannot deal with instancing yet");

		auto& lod = obj.first->get_lod(0u);
		const auto& polygons = lod.template get_geometry<scene::geometry::Polygons>();

		std::size_t collapses = 0u;

		if(polygons.get_vertex_count() >= m_params.threshold && m_params.initialReduction) {
			collapses = static_cast<std::size_t>(std::ceil(m_params.reduction * polygons.get_vertex_count()));
			logInfo("Reducing LoD 0 of object '", obj.first->get_name(), "' by ", collapses, " vertices");
		}
		const u32 newLodLevel = static_cast<u32>(obj.first->get_lod_slot_count());
		auto& newLod = obj.first->add_lod(newLodLevel, lod);
		m_decimaters.emplace_back(lod, newLod, Degrees(m_params.maxNormalDeviation),
								  static_cast<CollapseMode>(m_params.collapseMode),
								  collapses);

		// TODO: this reeeeally breaks instancing
		for(scene::InstanceHandle inst : obj.second) {
			// Modify the scenario to use this lod instead
			scene::WorldContainer::instance().get_current_scenario()->set_custom_lod(inst, newLodLevel);
		}
	}
}

void CpuShadowSilhouettes::update_reduction_factors() {
	m_remainingVertexFactor.clear();
	double expectedVertexCount = 0.0;
	for(auto& decimater : m_decimaters) {
		decimater.udpate_importance_density();
		if(decimater.get_original_vertex_count() > m_params.threshold) {
			m_remainingVertexFactor.push_back(decimater.get_importance_sum());
			expectedVertexCount += (1.f - m_params.reduction) * decimater.get_original_vertex_count();
		} else {
			m_remainingVertexFactor.push_back(1.0);
			expectedVertexCount += decimater.get_original_vertex_count();
		}
	}

	// Determine the reduction parameters for each mesh
	constexpr u32 MAX_ITERATION_COUNT = 10u;
	for(u32 iteration = 0u; iteration < MAX_ITERATION_COUNT; ++iteration) {
		double vertexCountAfterDecimation = 0.0;
		for(std::size_t i = 0u; i < m_decimaters.size(); ++i)
			vertexCountAfterDecimation += m_remainingVertexFactor[i] * m_decimaters[i].get_original_vertex_count();
		const double normalizationFactor = expectedVertexCount / vertexCountAfterDecimation;

		bool anyAboveOne = false;

		for(std::size_t i = 0u; i < m_decimaters.size(); ++i) {
			if(m_decimaters[i].get_original_vertex_count() > m_params.threshold) {
				m_remainingVertexFactor[i] *= normalizationFactor;
				anyAboveOne |= m_remainingVertexFactor[i] > 1.0;
				m_remainingVertexFactor[i] = std::clamp(m_remainingVertexFactor[i], 0.0, 1.0);
			}
		}

		if(!anyAboveOne)
			break;
	}
}

void CpuShadowSilhouettes::init_rngs(int num) {
	m_rngs.resize(num);
	// TODO: incude some global seed into the initialization
	for(int i = 0; i < num; ++i)
		m_rngs[i] = math::Rng(i);
}

} // namespace mufflon::renderer::decimaters