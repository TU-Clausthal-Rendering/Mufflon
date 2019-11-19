#pragma once

#include "cpu_importance.hpp"
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

namespace mufflon::renderer::decimaters {

using namespace importance;

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

CpuImportanceDecimater::CpuImportanceDecimater() {
	// TODO: init one RNG per thread?
	std::random_device rndDev;
	m_rngs.emplace_back(static_cast<u32>(rndDev()));
}

void CpuImportanceDecimater::pre_reset() {
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

	// Initialize the decimaters
	// TODO: how to deal with instancing
	if(m_currentDecimationIteration == 0u)
		this->initialize_decimaters();

	RendererBase<Device::CPU, importance::ImportanceTargets>::pre_reset();
}

void CpuImportanceDecimater::on_world_clearing() {
	m_decimaters.clear();
	m_currentDecimationIteration = 0u;
}

void CpuImportanceDecimater::post_iteration(IOutputHandler& outputBuffer) {
	if((int)m_currentDecimationIteration == m_params.decimationIterations) {
		// Finalize the decimation process
		logInfo("Finished decimation process");
		++m_currentDecimationIteration;
		this->on_manual_reset();
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
	RendererBase<Device::CPU, importance::ImportanceTargets>::post_iteration(outputBuffer);
}

void CpuImportanceDecimater::iterate() {
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

		logInfo("Finished importance gathering (max. importance: ", m_maxImportance, "; ",
				std::chrono::duration_cast<std::chrono::milliseconds>(processTime).count(),
				"ms, ", cycles / 1'000'000, " MCycles)");
	}

	if(m_outputBuffer.template is_target_enabled<RadianceTarget>()
	   || (int)m_currentDecimationIteration >= m_params.decimationIterations) {
		const u32 NUM_PIXELS = m_outputBuffer.get_num_pixels();
#pragma PARALLEL_FOR
		for(int i = 0; i < (int)NUM_PIXELS; ++i) {
			this->pt_sample(Pixel{ i % m_outputBuffer.get_width(), i / m_outputBuffer.get_width() });
		}
	}
}


void CpuImportanceDecimater::gather_importance() {
	const u32 NUM_PIXELS = m_outputBuffer.get_num_pixels();
#pragma PARALLEL_FOR
	for(int i = 0; i < m_params.importanceIterations * (int)NUM_PIXELS; ++i) {
		const int pixel = i / m_params.importanceIterations;
		this->importance_sample(Pixel{ pixel % m_outputBuffer.get_width(), pixel / m_outputBuffer.get_width() });
	}
	// TODO: allow for this with proper reset "events"
}

void CpuImportanceDecimater::importance_sample(const Pixel coord) {
	int pixel = coord.x + coord.y * m_outputBuffer.get_width();

	Spectrum throughput{ ei::Vec3{1.0f} };
	// We gotta keep track of our vertices
	thread_local std::vector<ImpPathVertex> vertices(std::max(2, m_params.maxPathLength + 1));
	vertices.clear();
	// Create a start for the path
	(void)ImpPathVertex::create_camera(&vertices.front(), &vertices.front(), m_sceneDesc.camera.get(), coord, m_rngs[pixel].next());

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
			auto nee = scene::lights::connect(m_sceneDesc, 0, 1, neeSeed,
											  vertices[pathLen].get_position(),
											  neeRnd);
			Pixel projCoord;
			auto value = vertices[pathLen].evaluate(nee.dir.direction, m_sceneDesc.media, projCoord);
			mAssert(!isnan(value.value.x) && !isnan(value.value.y) && !isnan(value.value.z));
			Spectrum radiance = value.value * nee.diffIrradiance;
			// TODO: use multiple NEEs
			if(any(greater(radiance, 0.0f)) && value.cosOut > 0.0f) {
				auto shadowRay = ei::Ray{ nee.position, -nee.dir.direction };

				auto shadowHit = scene::accel_struct::first_intersection(m_sceneDesc, shadowRay, vertices[pathLen].get_geometric_normal(), nee.dist);
				AreaPdf hitPdf = value.pdf.forw.to_area_pdf(nee.cosOut, nee.distSq);
				float mis = 1.0f / (1.0f + hitPdf / nee.creationPdf);
				const ei::Vec3 irradiance = nee.diffIrradiance * value.cosOut; // [W/m²]
				const float weightedIrradianceLuminance = get_luminance(throughput * irradiance) *(1.f - ei::abs(vertices[pathLen - 1].ext().outCos));
				if(shadowHit.hitId.instanceId < 0) {
					mAssert(!isnan(mis));
					// Save the radiance for the later indirect lighting computation
					// Compute how much radiance arrives at the previous vertex from the direct illumination
					// Add the importance
					vertices[pathLen].ext().pathRadiance = mis * radiance * value.cosOut;

					const auto& hitId = vertices[pathLen].get_primitive_id();
					const auto& lod = m_sceneDesc.lods[m_sceneDesc.lodIndices[hitId.instanceId]];
					const u32 numVertices = hitId.primId < (i32)lod.polygon.numTriangles ? 3u : 4u;
					const u32 vertexOffset = hitId.primId < (i32)lod.polygon.numTriangles ? 0u : 3u * lod.polygon.numTriangles;
					m_decimaters[m_sceneDesc.lodIndices[hitId.instanceId]]->record_direct_irradiance(&lod.polygon.vertexIndices[vertexOffset + numVertices * hitId.primId],
																									 numVertices, vertices[pathLen].get_position(), weightedIrradianceLuminance);
				}
			}
		}

		// Walk
		scene::Point lastPosition = vertices[pathLen].get_position();
		math::RndSet2_1 rnd{ m_rngs[pixel].next(), m_rngs[pixel].next() };
		VertexSample sample;
		float rndRoulette = math::sample_uniform(u32(m_rngs[pixel].next()));
		if(walk(m_sceneDesc, vertices[pathLen], rnd, rndRoulette, false, throughput, vertices[pathLen + 1], sample) == WalkResult::CANCEL)
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
	for(int p = pathLen - 2; p >= 1; --p) {
		accumRadiance = vertices[p].ext().throughput * accumRadiance + vertices[p + 1].ext().pathRadiance;
		const ei::Vec3 irradiance = vertices[p].ext().outCos * accumRadiance;

		const auto& hitId = vertices[p].get_primitive_id();
		const auto* lod = &m_sceneDesc.lods[m_sceneDesc.lodIndices[hitId.instanceId]];
		const u32 numVertices = hitId.primId < (i32)lod->polygon.numTriangles ? 3u : 4u;
		const u32 vertexOffset = hitId.primId < (i32)lod->polygon.numTriangles ? 0u : 3u * lod->polygon.numTriangles;

		const float importance = ei::max(0.f, get_luminance(irradiance) * (1.f - ei::abs(vertices[p].ext().outCos)));
		m_decimaters[m_sceneDesc.lodIndices[hitId.instanceId]]->record_indirect_irradiance(&lod->polygon.vertexIndices[vertexOffset + numVertices * hitId.primId],
																							numVertices, vertices[pathLen].get_position(), importance);
	}
}

void CpuImportanceDecimater::pt_sample(const Pixel coord) {
	int pixel = coord.x + coord.y * m_outputBuffer.get_width();

	auto& rng = m_rngs[pixel];
	Spectrum throughput{ ei::Vec3{1.0f} };
	float guideWeight = 1.f;
	PtPathVertex vertex;
	VertexSample sample;
	// Create a start for the path
	PtPathVertex::create_camera(&vertex, &vertex, m_sceneDesc.camera.get(), coord, rng.next());

	int pathLen = 0;
	do {
		if(pathLen > 0 && pathLen + 1 >= m_params.minPathLength && pathLen + 1 <= m_params.maxPathLength) {
			// Call NEE member function for recursive vertices.
			// Do not connect to the camera, because this makes the renderer much more
			// complicated. Our decision: The PT should be as simple as possible!
			// What means more complicated?
			// A connnection to the camera results in a different pixel. In a multithreaded
			// environment this means that we need a write mutex for each pixel.
			u64 neeSeed = rng.next();
			for(int i = 0; i < m_params.neeCount; ++i) {
				math::RndSet2 neeRnd = rng.next();
				auto nee = scene::lights::connect(m_sceneDesc, i, m_params.neeCount, neeSeed,
												  vertex.get_position(), neeRnd);
				Pixel outCoord;
				auto value = vertex.evaluate(nee.dir.direction, m_sceneDesc.media, outCoord);
				if(nee.cosOut != 0) value.cosOut *= nee.cosOut;
				mAssert(!isnan(value.value.x) && !isnan(value.value.y) && !isnan(value.value.z));
				Spectrum radiance = value.value * nee.diffIrradiance;
				if(any(greater(radiance, 0.0f)) && value.cosOut > 0.0f) {
					bool anyhit = scene::accel_struct::any_intersection(
						m_sceneDesc, vertex.get_position(), nee.position,
						vertex.get_geometric_normal(), nee.geoNormal,
						nee.dir.direction);
					if(!anyhit) {
						AreaPdf hitPdf = value.pdf.forw.to_area_pdf(nee.cosOut, nee.distSq);
						float mis = 1.0f / (m_params.neeCount + hitPdf / nee.creationPdf);
						mAssert(!isnan(mis));
						m_outputBuffer.template contribute<RadianceTarget>(coord, throughput * value.cosOut * radiance * mis);
					}
				}
			}
		}

		// Walk
		scene::Point lastPosition = vertex.get_position();
		math::RndSet2_1 rnd{ rng.next(), rng.next() };
		float rndRoulette = math::sample_uniform(u32(rng.next()));
		if(walk(m_sceneDesc, vertex, rnd, rndRoulette, false, throughput, vertex, sample, guideWeight) != WalkResult::CANCEL)
			break;
		++pathLen;
		
		// Evaluate direct hit of area ligths
		if(pathLen >= m_params.minPathLength) {
			EmissionValue emission = vertex.get_emission(m_sceneDesc, lastPosition);
			if(emission.value != 0.0f) {
				float mis = 1.f / (1.f + m_params.neeCount * (emission.connectPdf / vertex.ext().incidentPdf));
				emission.value *= mis;
			}
			m_outputBuffer.template contribute<RadianceTarget>(coord, throughput * emission.value);
		}
		if(vertex.is_end_point())
			break;
	} while(pathLen < m_params.maxPathLength);
}

void CpuImportanceDecimater::compute_max_importance() {
	m_maxImportance = 0.f;

	// Compute the maximum normalized importance for visualization
//#pragma omp parallel for reduction(max:m_maxImportance)
	for(i32 i = 0u; i < m_sceneDesc.numInstances; ++i)
		m_maxImportance = std::max(m_maxImportance, m_decimaters[m_sceneDesc.lodIndices[i]]->get_current_max_importance());
}

float CpuImportanceDecimater::query_importance(const ei::Vec3& hitPoint, const scene::PrimitiveHandle& hitId) {
	// TODO: density or importance?
	return m_decimaters[m_sceneDesc.lodIndices[hitId.instanceId]]->get_current_importance(hitId.primId, hitPoint) / m_maxImportance;
}

u32 CpuImportanceDecimater::get_memory_requirement() const {
	u32 memory = 0u;

	for(auto& obj : m_currentScene->get_objects()) {
		mAssert(obj.first != nullptr);
		mAssert(obj.second.size() != 0u);

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

void CpuImportanceDecimater::initialize_decimaters() {
	auto& objects = m_currentScene->get_objects();
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
		for(const auto& inst : obj.second)
			if(const auto level = scenario.get_effective_lod(inst); level < lowestLevel)
				lowestLevel = level;

		// TODO: this only works if instances don't specify LoD levels
		auto& lod = obj.first->get_lod(lowestLevel);
		auto& newLod = lod.create_reduced_version();
		const auto& polygons = newLod.template get_geometry<scene::geometry::Polygons>();

		std::size_t collapses = 0u;

		if(polygons.get_vertex_count() >= m_params.threshold && m_params.initialReduction > 0.f) {
			collapses = static_cast<std::size_t>(std::ceil(m_params.initialReduction * polygons.get_vertex_count()));
			logInfo("Reducing LoD 0 of object '", obj.first->get_name(), "' by ", collapses, " vertices");
		}
		m_decimaters[i] = std::make_unique<ImportanceDecimater>(lod, newLod, collapses,
																Degrees(m_params.maxNormalDeviation),
																m_params.viewWeight, m_params.lightWeight);

		// Make sure that all LoDs reference the correct reduced version
		for(const auto& inst : obj.second)
			obj.first->get_lod(scenario.get_effective_lod(inst)).reference_reduced_version(lod);
	}

	logInfo("Initial decimation: ", std::chrono::duration_cast<std::chrono::milliseconds>(CpuProfileState::get_process_time() - timeBegin).count(), "ms");
}

void CpuImportanceDecimater::update_reduction_factors() {
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

void CpuImportanceDecimater::init_rngs(int num) {
	m_rngs.resize(num);
	// TODO: incude some global seed into the initialization
	for(int i = 0; i < num; ++i)
		m_rngs[i] = math::Rng(i);
}

} // namespace mufflon::renderer::decimaters