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
#include <queue>

namespace mufflon::renderer::decimaters {

using namespace silhouette;

namespace {

inline float get_luminance(const ei::Vec3& vec) {
	constexpr ei::Vec3 LUM_WEIGHT{ 0.212671f, 0.715160f, 0.072169f };
	return ei::dot(LUM_WEIGHT, vec);
}

} // namespace

CpuShadowSilhouettes::CpuShadowSilhouettes()
{
	// TODO: init one RNG per thread?
	std::random_device rndDev;
	m_rngs.emplace_back(static_cast<u32>(rndDev()));
}

void CpuShadowSilhouettes::on_scene_load() {
	if(m_currentDecimationIteration != 0u) {
		// At least activate the created LoDs
		for(auto& obj : m_currentScene->get_objects()) {
			const u32 newLodLevel = static_cast<u32>(obj.first->get_lod_slot_count() - 1u);
			// TODO: this reeeeally breaks instancing
			scene::WorldContainer::instance().get_current_scenario()->set_custom_lod(obj.first, newLodLevel);
		}
	}
}

void CpuShadowSilhouettes::on_scene_unload() {
	m_decimaters.clear();
	m_currentDecimationIteration = 0u;
}

void CpuShadowSilhouettes::post_iteration(OutputHandler& outputBuffer) {
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

void CpuShadowSilhouettes::pre_descriptor_requery() {
	init_rngs(m_outputBuffer.get_num_pixels());

	// Initialize the decimaters
	// TODO: how to deal with instancing
	if(m_currentDecimationIteration == 0u)
		this->initialize_decimaters();
}

void CpuShadowSilhouettes::iterate() {
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
			compute_max_importance();
		}

		if(m_params.renderUpdate || (int)m_currentDecimationIteration >= m_params.decimationIterations) {
			const u32 NUM_PIXELS = m_outputBuffer.get_num_pixels();
#pragma PARALLEL_FOR
			for(int i = 0; i < (int)NUM_PIXELS; ++i) {
				this->imp_vis_sample(Pixel{ i % m_outputBuffer.get_width(), i / m_outputBuffer.get_width() });
			}
		}
	}
}


void CpuShadowSilhouettes::gather_importance() {
	for(auto& decimater : m_decimaters)
		decimater->start_iteration();

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
	vertices.clear();
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
			auto value = vertices[pathLen].evaluate(nee.dir.direction, m_sceneDesc.media, projCoord);
			mAssert(!isnan(value.value.x) && !isnan(value.value.y) && !isnan(value.value.z));
			Spectrum radiance = value.value * nee.diffIrradiance;
			// TODO: use multiple NEEs
			if(any(greater(radiance, 0.0f)) && value.cosOut > 0.0f) {
				vertices[pathLen].ext().shadowRay = ei::Ray{ nee.position, -nee.dir.direction };
				vertices[pathLen].ext().lightDistance = nee.dist;

				auto shadowHit = scene::accel_struct::first_intersection(m_sceneDesc, vertices[pathLen].ext().shadowRay,
																		 vertices[pathLen].get_geometric_normal(), nee.dist);
				vertices[pathLen].ext().shadowHit = shadowHit.hitId;
				vertices[pathLen].ext().firstShadowDistance = shadowHit.hitT;
				AreaPdf hitPdf = value.pdf.forw.to_area_pdf(nee.cosOut, nee.distSq);
				float mis = 1.0f / (1.0f + hitPdf / nee.creationPdf);
				const ei::Vec3 irradiance = nee.diffIrradiance * value.cosOut; // [W/m²]
				vertices[pathLen].ext().pathRadiance = mis * radiance * value.cosOut;

				const float weightedIrradianceLuminance = get_luminance(throughput.weight * irradiance) *(1.f - ei::abs(vertices[pathLen - 1].ext().outCos));
				if(shadowHit.hitId.instanceId < 0) {
					mAssert(!isnan(mis));
					// Save the radiance for the later indirect lighting computation
					// Compute how much radiance arrives at the previous vertex from the direct illumination
					// Add the importance

					const auto& hitId = vertices[pathLen].get_primitive_id();
					const auto& lod = m_sceneDesc.lods[m_sceneDesc.lodIndices[hitId.instanceId]];
					const u32 numVertices = hitId.primId < (i32)lod.polygon.numTriangles ? 3u : 4u;
					const u32 vertexOffset = hitId.primId < (i32)lod.polygon.numTriangles ? 0u : 3u * lod.polygon.numTriangles;
					m_decimaters[m_sceneDesc.lodIndices[hitId.instanceId]]->record_direct_irradiance(&lod.polygon.vertexIndices[vertexOffset + numVertices * hitId.primId],
																									numVertices, vertices[pathLen].get_position(), weightedIrradianceLuminance);
				} else {
					//m_decimaters[m_sceneDesc.lodIndices[shadowHit.hitId.instanceId]]->record_shadow(get_luminance(throughput.weight * irradiance));
					this->trace_shadow(vertices[pathLen].ext().shadowRay, vertices[pathLen], weightedIrradianceLuminance);
				}
			}
		}

		// Walk
		scene::Point lastPosition = vertices[pathLen].get_position();
		math::RndSet2_1 rnd{ m_rngs[pixel].next(), m_rngs[pixel].next() };
		VertexSample sample;
		float rndRoulette = math::sample_uniform(u32(m_rngs[pixel].next()));
		if(!walk(m_sceneDesc, vertices[pathLen], rnd, rndRoulette, false, throughput, vertices[pathLen + 1], sample))
			break;

		// Update old vertex with accumulated throughput
		vertices[pathLen].ext().updateBxdf(sample, throughput);

		// Don't update sharpness for camera vertex
		if(pathLen > 0) {
			const ei::Vec3 bxdf = sample.throughput * (float)sample.pdf.forw;
			const float bxdfLum = get_luminance(bxdf);
			if(isnan(bxdfLum))
				return;
			sharpness *= 2.f / (1.f + ei::exp(-bxdfLum / m_params.sharpnessFactor)) - 1.f;
		}

		// Fetch the relevant information for attributing the instance to the correct vertices
		const auto& hitId = vertices[pathLen + 1].get_primitive_id();
		const auto& lod = m_sceneDesc.lods[m_sceneDesc.lodIndices[hitId.instanceId]];
		const u32 numVertices = hitId.primId < (i32)lod.polygon.numTriangles ? 3u : 4u;
		const u32 vertexOffset = hitId.primId < (i32)lod.polygon.numTriangles ? 0u : 3u * lod.polygon.numTriangles;

		m_decimaters[m_sceneDesc.lodIndices[hitId.instanceId]]->record_direct_hit(&lod.polygon.vertexIndices[vertexOffset + numVertices * hitId.primId],
																					numVertices, vertices[pathLen].get_position(),
																					-ei::dot(vertices[pathLen + 1].get_incident_direction(),
																							vertices[pathLen + 1].get_normal()),
																				  sharpness);

		++pathLen;
	} while(pathLen < m_params.maxPathLength);

	// Go back over the path and add up the irradiance from indirect illumination
	ei::Vec3 accumRadiance{ 0.f };
	float accumThroughout = 1.f;
	for (int p = pathLen - 2; p >= 1; --p) {
		accumRadiance = vertices[p].ext().throughput * accumRadiance + (vertices[p + 1].ext().shadowHit.instanceId < 0 ?
			vertices[p + 1].ext().pathRadiance : ei::Vec3{ 0.f });
		const ei::Vec3 irradiance = vertices[p].ext().outCos * accumRadiance;

		const auto& hitId = vertices[p].get_primitive_id();
		const auto* lod = &m_sceneDesc.lods[m_sceneDesc.lodIndices[hitId.instanceId]];
		const u32 numVertices = hitId.primId < (i32)lod->polygon.numTriangles ? 3u : 4u;
		const u32 vertexOffset = hitId.primId < (i32)lod->polygon.numTriangles ? 0u : 3u * lod->polygon.numTriangles;

		const float importance = get_luminance(irradiance) * (1.f - ei::abs(vertices[p].ext().outCos));
		m_decimaters[m_sceneDesc.lodIndices[hitId.instanceId]]->record_indirect_irradiance(&lod->polygon.vertexIndices[vertexOffset + numVertices * hitId.primId],
																							numVertices, vertices[pathLen].get_position(), importance);
		// TODO: store accumulated sharpness
		// Check if it is sensible to keep shadow silhouettes intact
		// TODO: replace threshold with something sensible
		if (p == 1 && vertices[p].ext().shadowHit.instanceId >= 0) {
			const float indirectLuminance = get_luminance(accumRadiance);
			const float totalLuminance = get_luminance(vertices[p].ext().pathRadiance) + indirectLuminance;
			const float ratio = totalLuminance / indirectLuminance - 1.f;
			if (ratio > 0.02f) {
				constexpr float DIST_EPSILON = 0.000125f;
				constexpr float FACTOR = 2'000.f;

				// TODO: proper factor!
				trace_shadow_silhouette(vertices[p].ext().shadowRay, vertices[p], FACTOR * (totalLuminance - indirectLuminance));
			}
		}
	}
}

void CpuShadowSilhouettes::imp_vis_sample(const Pixel coord) {
	int pixel = coord.x + coord.y * m_outputBuffer.get_width();

	auto& rng = m_rngs[pixel];
	math::Throughput throughput{ ei::Vec3{1.0f}, 1.0f };
	PtPathVertex vertex;
	VertexSample sample;
	// Create a start for the path
	PtPathVertex::create_camera(&vertex, &vertex, m_sceneDesc.camera.get(), coord, rng.next());

	scene::Point lastPosition = vertex.get_position();
	math::RndSet2_1 rnd{ rng.next(), rng.next() };
	float rndRoulette = math::sample_uniform(u32(rng.next()));
	if(walk(m_sceneDesc, vertex, rnd, rndRoulette, false, throughput, vertex, sample)) {
		const auto& hitpoint = vertex.get_position();
		const auto& hitId = vertex.get_primitive_id();
		const auto lodIdx = m_sceneDesc.lodIndices[hitId.instanceId];
		const auto& polygon = m_sceneDesc.lods[lodIdx].polygon;
		const u32 vertexCount = hitId.primId < (i32)polygon.numTriangles ? 3u : 4u;
		const u32 vertexOffset = vertexCount == 3u ? 0u : (polygon.numTriangles * 3u);
		const u32 primIdx = vertexCount == 3u ? hitId.primId : (hitId.primId - polygon.numTriangles);

		float importance = 0.f;
		float distSqrSum = 0.f;
		for(u32 i = 0u; i < vertexCount; ++i)
			distSqrSum += ei::lensq(hitpoint - polygon.vertices[polygon.vertexIndices[vertexOffset + vertexCount * primIdx + i]]);
		for(u32 i = 0u; i < vertexCount; ++i) {
			const auto vertexIndex = polygon.vertexIndices[vertexOffset + vertexCount * primIdx + i];
			const float distSqr = ei::lensq(hitpoint - polygon.vertices[vertexIndex]);
			importance += m_decimaters[lodIdx]->get_current_importance(hitId.primId, vertex.get_position());
		}

		m_outputBuffer.contribute(coord, RenderTargets::RADIANCE, ei::Vec4{ importance / m_maxImportance });
	}
}

void CpuShadowSilhouettes::compute_max_importance() {
	m_maxImportance = 0.f;

	// Compute the maximum normalized importance for visualization
//#pragma omp parallel for reduction(max:m_maxImportance)
	for(i32 i = 0u; i < m_decimaters.size(); ++i)
		m_maxImportance = std::max(m_maxImportance, m_decimaters[i]->get_current_max_importance());
}

bool CpuShadowSilhouettes::trace_shadow_silhouette(const ei::Ray& shadowRay, const SilPathVertex& vertex, const float importance) {
	constexpr float DIST_EPSILON = 0.001f;

	// TODO: worry about spheres?
	ei::Ray backfaceRay{ shadowRay.origin + shadowRay.direction * (vertex.ext().firstShadowDistance + DIST_EPSILON), shadowRay.direction };

	const auto secondHit = scene::accel_struct::first_intersection(m_sceneDesc, backfaceRay, vertex.get_geometric_normal(),
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
			ei::Ray silhouetteRay{ shadowRay.origin + shadowRay.direction * (vertex.ext().firstShadowDistance + secondHit.hitT), shadowRay.direction };

			/*const auto thirdHit = scene::accel_struct::any_intersection(m_sceneDesc, silhouetteRay, secondHit.hitId,
																		vertex.ext().lightDistance - vertex.ext().firstShadowDistance - secondHit.hitT + DIST_EPSILON);
			if(!thirdHit) {*/
			const auto thirdHit = scene::accel_struct::first_intersection(m_sceneDesc, silhouetteRay, secondHit.normal,
																		  vertex.ext().lightDistance - vertex.ext().firstShadowDistance - secondHit.hitT + DIST_EPSILON);
			if(thirdHit.hitId == vertex.get_primitive_id()) {
				for(i32 i = 0; i < sharedVertices; ++i) {
					// x86_64 doesn't support atomic_fetch_add for floats FeelsBadMan
					const auto lodIdx = m_sceneDesc.lodIndices[vertex.ext().shadowHit.instanceId];
					m_decimaters[lodIdx]->record_silhouette_vertex_contribution(edgeIdxFirst[i], importance);
					m_decimaters[lodIdx]->record_silhouette_vertex_contribution(edgeIdxSecond[i], importance);
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

bool CpuShadowSilhouettes::trace_shadow(const ei::Ray& shadowRay, const SilPathVertex& vertex, const float importance) {
	constexpr float DIST_EPSILON = 0.001f;

	// TODO: worry about spheres?
	ei::Ray backfaceRay{ shadowRay.origin + shadowRay.direction * (vertex.ext().firstShadowDistance + DIST_EPSILON), shadowRay.direction };

	const auto secondHit = scene::accel_struct::first_intersection(m_sceneDesc, backfaceRay, vertex.get_geometric_normal(),
																   vertex.ext().lightDistance - vertex.ext().firstShadowDistance + DIST_EPSILON);
	if(secondHit.hitId.instanceId < 0 || secondHit.hitId == vertex.get_primitive_id())
		return false;

	ei::Ray silhouetteRay{ shadowRay.origin + shadowRay.direction * (vertex.ext().firstShadowDistance + secondHit.hitT), shadowRay.direction };
	const auto thirdHit = scene::accel_struct::first_intersection(m_sceneDesc, silhouetteRay, secondHit.normal,
																  vertex.ext().lightDistance - vertex.ext().firstShadowDistance - secondHit.hitT + DIST_EPSILON);
	if(thirdHit.hitId == vertex.get_primitive_id()) {
		const auto& hitId = vertex.ext().shadowHit;
		const auto lodIdx = m_sceneDesc.lodIndices[hitId.instanceId];
		m_decimaters[lodIdx]->record_shadow(importance);
		return true;
	}
	return false;
}

void CpuShadowSilhouettes::initialize_decimaters() {
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
		m_decimaters[i] = std::make_unique<CpuImportanceDecimater>(lod, newLod, collapses,
																m_params.viewWeight, m_params.lightWeight,
																m_params.shadowWeight, m_params.shadowSilhouetteWeight);

		// TODO: this reeeeally breaks instancing
		scene::WorldContainer::instance().get_current_scenario()->set_custom_lod(obj.first, newLodLevel);
	}

	logInfo("Initial decimation: ", std::chrono::duration_cast<std::chrono::milliseconds>(CpuProfileState::get_process_time() - timeBegin).count(), "ms");
}

void CpuShadowSilhouettes::update_reduction_factors() {
	m_remainingVertexFactor.clear();
	if(m_params.reduction == 0.f) {
		// Do not reduce anything
		for(std::size_t i = 0u; i < m_decimaters.size(); ++i) {
			m_decimaters[i]->udpate_importance_density();
			m_remainingVertexFactor.push_back(1.0);
		}
		return;
	}

	double expectedVertexCount = 0.0;
	for(auto& decimater : m_decimaters) {
		decimater->udpate_importance_density();
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

void CpuShadowSilhouettes::init_rngs(int num) {
	m_rngs.resize(num);
	// TODO: incude some global seed into the initialization
	for(int i = 0; i < num; ++i)
		m_rngs[i] = math::Rng(i);
}

} // namespace mufflon::renderer::decimaters