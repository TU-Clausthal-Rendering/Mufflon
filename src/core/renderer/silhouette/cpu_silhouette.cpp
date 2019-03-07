#pragma once

#include "cpu_silhouette.hpp"
#include "sil_decimater.hpp"
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

namespace mufflon::renderer {

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
	m_addedLods = false;
	m_importanceMap.clear();
	// Request stati once to remember which vertices we deleted

	for(auto object : m_currentScene->get_objects())
		for(u32 i = 0u; i < object.first->get_lod_slot_count(); ++i)
			if(object.first->has_lod_available(i))
				object.first->get_lod(i).template get_geometry<scene::geometry::Polygons>().get_mesh().request_vertex_status();
}

bool CpuShadowSilhouettes::pre_iteration(OutputHandler& outputBuffer) {
	if(!m_reset) {
		if((int)m_currentDecimationIteration == m_params.decimationIterations) {
			// Finalize the decimation process

			logInfo("Finished importance gathering; starting decimation (target reduction of ", m_params.reduction, ")");

			const double averageDensity = m_importanceMap.get_importance_density_sum();
			const double currentAvImpDenVert = averageDensity / static_cast<double>(m_importanceMap.get_not_deleted_vertex_count());
			const u32 currentMemory = this->get_memory_requirement();

			const u32 allowedVertices = get_vertices_for_memory(m_params.memoryConstraint);
			const double threshold = 10.f;//averageDensity / static_cast<double>(allowedVertices);

			logInfo("Total average importance density: ", averageDensity);
			logInfo("Total average importance density per vertex: ", currentAvImpDenVert);
			logInfo("Threshold: ", threshold);

			u32 meshIndex = 0u;
			for(auto object : m_currentScene->get_objects()) {
				for(u32 i = 0u; i < object.first->get_lod_slot_count(); ++i) {
					if(object.first->has_lod_available(i)) {
						auto& lod = object.first->get_lod(i);
						auto& polygons = lod.template get_geometry<scene::geometry::Polygons>();
						const u32 vertexCount = static_cast<u32>(polygons.get_vertex_count());

						if(static_cast<int>(vertexCount) >= m_params.threshold) {
							// TODO: decimate with given bounds! (importance density threshold or memory constraint)


							// Create decimater and attach modules
							auto& mesh = polygons.get_mesh();
							auto decimater = polygons.create_decimater();


							ImportanceModule<>::Handle impModHdl;
							decimater.add(impModHdl);
							decimater.module(impModHdl).set_importance_map(m_importanceMap, meshIndex, threshold);
							/*MaxNormalDeviation<>::Handle normModHdl;
							decimater.add(normModHdl);
							decimater.module(normModHdl).set_max_deviation(60.0);*/
							polygons.decimate(decimater, 0u, false);

							//polygons.garbage_collect();
							lod.clear_accel_structure();
						}
						++meshIndex;
					}
				}
			}

			m_importanceMap.update_normalized();
			const double postAverageDensity = m_importanceMap.get_importance_density_sum();
			const double postAvImpDenVert = postAverageDensity / static_cast<double>(m_importanceMap.get_not_deleted_vertex_count());

			logInfo("Post total average importance density: ", postAverageDensity);
			logInfo("Post total average importance density per vertex: ", postAvImpDenVert);


			m_currentScene->clear_accel_structure();
			m_reset = true;
			++m_currentDecimationIteration;

		} else if((int)m_currentDecimationIteration < m_params.decimationIterations) {
			if(m_params.decimationEnabled)
				this->decimate();
			m_finishedDecimation = true;
			m_reset = true;
		}
	} else if(m_params.isConstraintInitial) {
		u32 memory = 0u;
		std::vector<u32> reducible;

		// TODO: this doesn't work with instancing
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
				const auto& polygons = lod.template get_geometry<scene::geometry::Polygons>();

				const u32 lodMemory = static_cast<u32>(polygons.get_triangle_count() * 3u * sizeof(u32)
					+ polygons.get_quad_count() * 3u * sizeof(u32)
					+ polygons.get_vertex_count() * (2u * sizeof(ei::Vec3) + sizeof(ei::Vec2)));
				if(polygons.get_vertex_count() >= m_params.threshold)
					reducible.push_back(lodMemory);
				else
					reducible.push_back(0u);
				memory += lodMemory;
			}
		}

		logInfo("Required memory for scene: ", pretty_print_size(memory), " (available: ", pretty_print_size(m_params.memoryConstraint), ")");

		if(static_cast<u32>(m_params.memoryConstraint) < memory) {
			// Compute the memory shares for each LoD
			const float reduction = 1.f - std::min(1.f, static_cast<float>(m_params.memoryConstraint) / std::max(1.f, static_cast<float>(memory)));
			for(auto& mem : reducible)
				mem = static_cast<u32>(mem * reduction);

			u32 index = 0u;
			for(auto& obj : m_currentScene->get_objects()) {
				for(scene::InstanceHandle inst : obj.second) {
					const u32 instanceLod = scene::WorldContainer::instance().get_current_scenario()->get_effective_lod(inst);
					auto& lod = obj.first->get_lod(instanceLod);
					const auto& polygons = lod.template get_geometry<scene::geometry::Polygons>();

					if(polygons.get_vertex_count() >= m_params.threshold) {
						// Figure out how many edges we need to collapse - for a manifold, every collapse removes one vertex and two triangles
						constexpr u32 MEM_PER_COLLAPSE = 2u * sizeof(ei::Vec3) + sizeof(ei::Vec2) + 2u * 3u * sizeof(u32);
						const u32 collapses = static_cast<u32>(std::ceil(static_cast<float>(reducible[index]) / static_cast<float>(MEM_PER_COLLAPSE)));

						// Copy the LoD and decimate it!
						const u32 newLodLevel = static_cast<u32>(obj.first->get_lod_slot_count());
						auto& newLod = obj.first->add_lod(newLodLevel, lod);
						auto& newPolygons = newLod.template get_geometry<scene::geometry::Polygons>();
						newPolygons.get_mesh().request_vertex_status();
						auto decimater = newPolygons.create_decimater();
						OpenMesh::Decimater::ModQuadricT<scene::geometry::PolygonMeshType>::Handle modQuadricHandle;
						decimater.add(modQuadricHandle);

						logInfo("Reducing LoD ", instanceLod, " of object '", obj.first->get_name(), "', instance '",
								inst->get_name(), "' by ", pretty_print_size(reducible[index]), "/", collapses, " vertices");

						u32 performedCollapses;
						do {
							newPolygons.decimate(decimater, polygons.get_vertex_count() - collapses, true);
							performedCollapses = static_cast<u32>(polygons.get_vertex_count() - newPolygons.get_vertex_count());
						} while(performedCollapses < collapses);

						// Modify the scenario to use this lod instead
						scene::WorldContainer::instance().get_current_scenario()->set_custom_lod(inst, newLodLevel);
					}
					++index;
				}
			}
		}
		logInfo("Finished scene reduction");
	}
	
	return RendererBase<Device::CPU>::pre_iteration(outputBuffer);
}

void CpuShadowSilhouettes::on_descriptor_requery() {
	init_rngs(m_outputBuffer.get_num_pixels());

	if(m_sceneDesc.numInstances != m_sceneDesc.numLods)
		throw std::runtime_error("We do not support instancing yet");

	// Initialize the importance map
	// TODO: how to deal with instancing
	initialize_importance_map();
	// TODO
	//m_currentDecimationIteration = 0u;
}

void CpuShadowSilhouettes::iterate() {
	// TODO: incorporate decimation/undecimation loop!

	if((int)m_currentDecimationIteration < m_params.decimationIterations) {
		logInfo("Starting decimation iteration (", m_currentDecimationIteration + 1, " of ", m_params.decimationIterations, ")");
		gather_importance();

		if(m_params.importanceIterations > 0 && m_outputBuffer.is_target_enabled(RenderTargets::IMPORTANCE)) {
			compute_max_importance();
			logInfo("Max. importance: ", m_maxImportance);
			display_importance();
		}
		++m_currentDecimationIteration;
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
	m_importanceMap.update_normalized();
}

void CpuShadowSilhouettes::decimate() {
	logInfo("Finished importance gathering; starting decimation (target reduction of ", m_params.reduction, ")");
#if 0
	u32 meshIndex = 0u;
	for(auto object : m_currentScene->get_objects()) {
		for(u32 i = 0u; i < object.first->get_lod_slot_count(); ++i) {
			if(object.first->has_lod_available(i)) {
				auto& lod = object.first->get_lod(i);
				auto& polygons = lod.get_geometry<scene::geometry::Polygons>();
				const u32 vertexCount = static_cast<u32>(polygons.get_vertex_count());

				if(static_cast<int>(vertexCount) < m_params.threshold) {
					logInfo("Skipping object ", object.first->get_object_id(), ", LoD ", i,
							" (too little vertices)");
				} else {
					const u32 targetVertexCount = static_cast<u32>(vertexCount * (1.f - m_params.reduction));
					logInfo("Decimating object ", object.first->get_object_id(), ", LoD ", i,
							" (", vertexCount, " => ", targetVertexCount, ")");

					// Create decimater and attach modules
					auto& mesh = polygons.get_mesh();
					auto decimater = polygons.create_decimater();


					ImportanceModule<>::Handle impModHdl;
					decimater.add(impModHdl);
					decimater.module(impModHdl).set_importance_map(m_importanceMap, meshIndex);
					/*MaxNormalDeviation<>::Handle normModHdl;
					decimater.add(normModHdl);
					decimater.module(normModHdl).set_max_deviation(60.0);*/
					polygons.decimate(decimater, targetVertexCount, false);

					lod.clear_accel_structure();
				}
				++meshIndex;
			}
		}
	}
#endif

	// We need to re-build the scene
	m_currentScene->clear_accel_structure();
	m_reset = true;
}

void CpuShadowSilhouettes::undecimate() {
	logInfo("Undecimating important regions...");
#if 0
	u32 meshIndex = 0u;
	for(auto object : m_currentScene->get_objects()) {
		for(u32 i = 0u; i < object.first->get_lod_slot_count(); ++i) {
			if(object.first->has_lod_available(i)) {
				auto& lod = object.first->get_lod(i);
				auto& polygons = lod.get_geometry<scene::geometry::Polygons>();
				const u32 vertexCount = static_cast<u32>(polygons.get_vertex_count());
					
				if(static_cast<int>(vertexCount) < m_params.threshold) {
					logInfo("Skipping object ", object.first->get_object_id(), ", LoD ", i,
							" (too little vertices)");
				} else {
					const u32 targetVertexCount = static_cast<u32>(vertexCount * (1.f - m_params.reduction));
					logInfo("Decimating object ", object.first->get_object_id(), ", LoD ", i,
							" (", vertexCount, " => ", targetVertexCount, ")");

					auto& mesh = polygons.get_mesh();
					OpenMesh::VPropHandleT<float> impHdl;

					// Create decimater and attach modules
					auto decimater = polygons.create_decimater();


					ImportanceModule<>::Handle impModHdl;
					decimater.add(impModHdl);
					decimater.module(impModHdl).set_importance_map(m_importanceMap, meshIndex);
					/*MaxNormalDeviation<>::Handle normModHdl;
					decimater.add(normModHdl);
					decimater.module(normModHdl).set_max_deviation(60.0);*/
					polygons.decimate(decimater, targetVertexCount, false);

					lod.clear_accel_structure();
				}
				++meshIndex;
			}
		}
	}

	// We need to re-build the scene
	m_currentScene->clear_accel_structure();
	m_reset = true;
#endif

	/*constexpr StringView importanceAttrName{ "importance" };

	u32 vertexOffset = 0u;
	for(auto object : m_currentScene->get_objects()) {
		for(u32 i = 0u; i < object.first->get_lod_slot_count(); ++i) {
			if(object.first->has_lod_available(i)) {
				auto& lod = object.first->get_lod(i);
				auto& polygons = lod.get_geometry<scene::geometry::Polygons>();
				const u32 vertexCount = static_cast<u32>(polygons.get_vertex_count());

				auto& mesh = polygons.get_mesh();

				for(auto edge : mesh.edges()) {
					auto heh = mesh.halfedge_handle(edge, 0);
					auto v0 = mesh.from_vertex_handle(heh);
					auto v1 = mesh.to_vertex_handle(heh);
					auto imp0 = m_importanceMap[vertexOffset + v0.idx()].load();
					auto imp1 = m_importanceMap[vertexOffset + v1.idx()].load();

					if(imp0 > 0.f || imp1 > 0.f) {
						if(imp0 > imp1) {

						} else {

						}
					}
				}

				if(static_cast<int>(vertexCount) < m_params.threshold) {
					logInfo("Skipping object ", object.first->get_object_id(), ", LoD ", i,
							" (too little vertices)");
				} else {
					const u32 targetVertexCount = static_cast<u32>(vertexCount * (1.f - m_params.reduction));
					logInfo("Decimating object ", object.first->get_object_id(), ", LoD ", i,
							" (", vertexCount, " => ", targetVertexCount, ")");

					auto& mesh = polygons.get_mesh();
					OpenMesh::VPropHandleT<float> impHdl;
					mesh.add_property(impHdl, std::string(importanceAttrName));

					for(u32 v = 0u; v < vertexCount; ++v)
						mesh.property(impHdl, OpenMesh::VertexHandle {(int)v}) = m_importanceMap[vertexOffset + v].load();

					// Create decimater and attach modules
					auto decimater = polygons.create_decimater();

					ImportanceModule<scene::geometry::PolygonMeshType>::Handle impModHdl;
					ImportanceBinaryModule<scene::geometry::PolygonMeshType>::Handle impBinModHdl;
					decimater.add(impModHdl);
					decimater.add(impBinModHdl);
					decimater.module(impModHdl).set_importance_property(impHdl);
					decimater.module(impBinModHdl).set_importance_property(impHdl);
					//OpenMesh::Decimater::ModNormalFlippingT<scene::geometry::PolygonMeshType>::Handle normal_flipping;
					//decimater.add(normal_flipping);
					//decimater.module(normal_flipping).set_max_normal_deviation(m_params.maxNormalDeviation);

					// Perform actual decimation
					// TODO: only create LoD
					polygons.decimate(decimater, targetVertexCount, false);

					lod.clear_accel_structure();
				}
				vertexOffset += vertexCount;
			}
		}
	}

	// We need to re-build the scene
	m_currentScene->clear_accel_structure();
	m_sceneDesc = m_currentScene->get_descriptor<Device::CPU>({}, {}, {}, m_outputBuffer.get_resolution());*/
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
			auto value = vertices[pathLen].evaluate(nee.direction, m_sceneDesc.media);
			mAssert(!isnan(value.value.x) && !isnan(value.value.y) && !isnan(value.value.z));
			Spectrum radiance = value.value * nee.diffIrradiance;
			if(any(greater(radiance, 0.0f)) && value.cosOut > 0.0f) {
				vertices[pathLen].ext().shadowRay = ei::Ray{ nee.lightPoint, -nee.direction };
				vertices[pathLen].ext().lightDistance = nee.dist;

				auto shadowHit = scene::accel_struct::first_intersection_scene_lbvh<Device::CPU>(
					m_sceneDesc, vertices[pathLen].ext().shadowRay, vertices[pathLen].get_primitive_id(), nee.dist);
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
					for(u32 i = 0u; i < numVertices; ++i) {
						const u32 vertexIndex = lod.polygon.vertexIndices[vertexOffset + numVertices * hitId.primId + i];
						m_importanceMap.add(hitId.instanceId, vertexIndex, weightedIrradianceLuminance);
					}
				}
			}
		}

		// Walk
		scene::Point lastPosition = vertices[pathLen].get_position();
		math::RndSet2_1 rnd{ m_rngs[pixel].next(), m_rngs[pixel].next() };

		if(!walk(m_sceneDesc, vertices[pathLen], rnd, -1.0f, false, throughput, vertices[pathLen + 1]))
			break;

		// Update old vertex with accumulated throughput
		vertices[pathLen].ext().accumThroughput = throughput.weight;

		// Fetch the relevant information for attributing the instance to the correct vertices
		const auto& hitId = vertices[pathLen + 1].get_primitive_id();
		const auto& lod = m_sceneDesc.lods[m_sceneDesc.lodIndices[hitId.instanceId]];
		const u32 numVertices = hitId.primId < (i32)lod.polygon.numTriangles ? 3u : 4u;
		const u32 vertexOffset = hitId.primId < (i32)lod.polygon.numTriangles ? 0u : 3u * lod.polygon.numTriangles;

		if(pathLen > 0 && m_params.enableEyeImportance) {
			// Direct hits are being scaled down in importance by a sigmoid of the BxDF to get an idea of the "sharpness"
			const float importance = sharpness * (1.f - ei::abs(ei::dot(vertices[pathLen].get_normal(), vertices[pathLen].get_incident_direction())));
			for(u32 i = 0u; i < numVertices; ++i) {
				const u32 vertexIndex = lod.polygon.vertexIndices[vertexOffset + numVertices * hitId.primId + i];
				m_importanceMap.add(hitId.instanceId, vertexIndex, importance);
			}

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
			const auto& lod = &m_sceneDesc.lods[m_sceneDesc.lodIndices[hitId.instanceId]];
			const u32 numVertices = hitId.primId < (i32)lod->polygon.numTriangles ? 3u : 4u;
			const u32 vertexOffset = hitId.primId < (i32)lod->polygon.numTriangles ? 0u : 3u * lod->polygon.numTriangles;

			const float importance = get_luminance(irradiance) * (1.f - ei::abs(vertices[p].ext().outCos));
			for (u32 i = 0u; i < numVertices; ++i) {
				const u32 vertexIndex = lod->polygon.vertexIndices[vertexOffset + numVertices * hitId.primId + i];
				m_importanceMap.add(hitId.instanceId, vertexIndex, importance);
			}

			// TODO: store accumulated sharpness
			// Check if it is sensible to keep shadow silhouettes intact
			// TODO: replace threshold with something sensible
			if (p == 1 && vertices[p].ext().shadowHit.instanceId >= 0) {
				const float indirectLuminance = get_luminance(accumRadiance);
				const float totalLuminance = get_luminance(vertices[p].ext().pathRadiance) + indirectLuminance;
				const float ratio = totalLuminance / indirectLuminance - 1.f;
				if (ratio > 0.02f) {
					constexpr float DIST_EPSILON = 0.000125f;
					// TODO: proper factor!
					trace_shadow_silhouette(vertices[p].ext().shadowRay, vertices[p], 2000.0f * (totalLuminance - indirectLuminance));
				}
			}
		}
	}
}

void CpuShadowSilhouettes::pt_sample(const Pixel coord) {
	int pixel = coord.x + coord.y * m_outputBuffer.get_width();

	//m_params.maxPathLength = 2;

	math::Throughput throughput{ ei::Vec3{1.0f}, 1.0f };
	PtPathVertex vertex;
	// Create a start for the path
	(void)PtPathVertex::create_camera(&vertex, &vertex, m_sceneDesc.camera.get(), coord, m_rngs[pixel].next());

	int pathLen = 0;
	do {
		if(pathLen > 0.0f && pathLen + 1 <= m_params.maxPathLength) {
			// Call NEE member function for recursive vertices.
			// Do not connect to the camera, because this makes the renderer much more
			// complicated. Our decision: The PT should be as simple as possible!
			// What means more complicated?
			// A connnection to the camera results in a different pixel. In a multithreaded
			// environment this means that we need a write mutex for each pixel.
			// TODO: test/parametrize mulievent estimation (more indices in connect) and different guides.
			u64 neeSeed = m_rngs[pixel].next();
			math::RndSet2 neeRnd = m_rngs[pixel].next();
			auto nee = connect(m_sceneDesc.lightTree, 0, 1, neeSeed,
							   vertex.get_position(), m_currentScene->get_bounding_box(),
							   neeRnd, scene::lights::guide_flux);
			auto value = vertex.evaluate(nee.direction, m_sceneDesc.media);
			mAssert(!isnan(value.value.x) && !isnan(value.value.y) && !isnan(value.value.z));
			Spectrum radiance = value.value * nee.diffIrradiance;
			if(any(greater(radiance, 0.0f)) && value.cosOut > 0.0f) {
				bool anyhit = scene::accel_struct::any_intersection_scene_lbvh<Device::CPU>(
					m_sceneDesc, { vertex.get_position() , nee.direction },
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
		math::RndSet2_1 rnd{ m_rngs[pixel].next(), m_rngs[pixel].next() };
		if(!walk(m_sceneDesc, vertex, rnd, -1.0f, false, throughput, vertex)) {
			if(throughput.weight != Spectrum{ 0.f }) {
				// Missed scene - sample background
				auto background = evaluate_background(m_sceneDesc.lightTree.background, vertex.ext().excident);
				if(any(greater(background.value, 0.0f))) {
					float mis = 1.0f / (1.0f + background.pdfB / vertex.ext().pdf);
					background.value *= mis;
					m_outputBuffer.contribute(coord, throughput, background.value,
											ei::Vec3{ 0, 0, 0 }, ei::Vec3{ 0, 0, 0 },
											ei::Vec3{ 0, 0, 0 });
				}
			}
			break;
		}

		if(pathLen == 0 && m_params.importanceIterations > 0 && m_outputBuffer.is_target_enabled(RenderTargets::IMPORTANCE)) {
			m_outputBuffer.contribute(coord, RenderTargets::IMPORTANCE, ei::Vec4{ query_importance(vertex.get_position(), vertex.get_primitive_id()) });
		}

		++pathLen;

		// Evaluate direct hit of area ligths
		if(pathLen <= m_params.maxPathLength) {
			Spectrum emission = vertex.get_emission();
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
		}
	} while(pathLen < m_params.maxPathLength);
}

void CpuShadowSilhouettes::compute_max_importance() {
	m_maxImportance = 0.f;

	// Compute the maximum normalized importance for visualization
//#pragma omp parallel for reduction(max:m_maxImportance)
	for(i32 i = 0u; i < m_sceneDesc.numInstances; ++i) {
		for(u32 v = 0u; v < m_importanceMap.get_vertex_count(i); ++v) {
			m_maxImportance = std::max(m_maxImportance, m_importanceMap.normalized(i, v));
		}
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
		if(walk(m_sceneDesc, vertex, rnd, -1.0f, false, throughput, vertex))
			m_outputBuffer.contribute(coord, RenderTargets::IMPORTANCE, ei::Vec4{ query_importance(vertex.get_position(), vertex.get_primitive_id()) });
	}

}



float CpuShadowSilhouettes::query_importance(const ei::Vec3& hitPoint, const scene::PrimitiveHandle& hitId) {
	const auto& lod = m_sceneDesc.lods[m_sceneDesc.lodIndices[hitId.instanceId]];

	float importance = 0.f;

	// Compute the closest vertex
	const u32 vertexCount = ((u32)hitId.primId < lod.polygon.numTriangles) ? 3u : 4u;
	if(vertexCount == 3u) {
		// Triangle
		const u32 i0 = lod.polygon.vertexIndices[3u * hitId.primId + 0u];
		const u32 i1 = lod.polygon.vertexIndices[3u * hitId.primId + 1u];
		const u32 i2 = lod.polygon.vertexIndices[3u * hitId.primId + 2u];
		const float d0 = ei::lensq(hitPoint - lod.polygon.vertices[i0]);
		const float d1 = ei::lensq(hitPoint - lod.polygon.vertices[i1]);
		const float d2 = ei::lensq(hitPoint - lod.polygon.vertices[i2]);
		if(d0 < d1) {
			if(d0 < d2)
				importance = m_importanceMap.normalized(hitId.instanceId, i0);
			else
				importance = m_importanceMap.normalized(hitId.instanceId, i2);
		} else {
			if(d1 < d2)
				importance = m_importanceMap.normalized(hitId.instanceId, i1);
			else
				importance = m_importanceMap.normalized(hitId.instanceId, i2);
		}

	} else {
		mAssert(vertexCount == 4u);
		// Quad
		const u32 vertexOffset = 3u * static_cast<u32>(lod.polygon.numTriangles);
		const u32 i0 = lod.polygon.vertexIndices[vertexOffset + 4u * hitId.primId + 0u];
		const u32 i1 = lod.polygon.vertexIndices[vertexOffset + 4u * hitId.primId + 1u];
		const u32 i2 = lod.polygon.vertexIndices[vertexOffset + 4u * hitId.primId + 2u];
		const u32 i3 = lod.polygon.vertexIndices[vertexOffset + 4u * hitId.primId + 3u];
		const float d0 = ei::lensq(hitPoint - lod.polygon.vertices[i0]);
		const float d1 = ei::lensq(hitPoint - lod.polygon.vertices[i1]);
		const float d2 = ei::lensq(hitPoint - lod.polygon.vertices[i2]);
		const float d3 = ei::lensq(hitPoint - lod.polygon.vertices[i3]);

		if(d0 < d1) {
			if(d0 < d2) {
				if(d0 < d3)
					importance = m_importanceMap.normalized(hitId.instanceId, i0);
				else
					importance = m_importanceMap.normalized(hitId.instanceId, i3);
			} else {
				if(d2 < d3)
					importance = m_importanceMap.normalized(hitId.instanceId, i2);
				else
					importance = m_importanceMap.normalized(hitId.instanceId, i3);
			}
		} else {
			if(d1 < d2) {
				if(d1 < d3)
					importance = m_importanceMap.normalized(hitId.instanceId, i1);
				else
					importance = m_importanceMap.normalized(hitId.instanceId, i3);
			} else {
				if(d2 < d3)
					importance = m_importanceMap.normalized(hitId.instanceId, i2);
				else
					importance = m_importanceMap.normalized(hitId.instanceId, i3);
			}
		}
	}
	return importance / m_maxImportance;
}

bool CpuShadowSilhouettes::trace_shadow_silhouette(const ei::Ray& shadowRay, const SilPathVertex& vertex,const float importance) {
	constexpr float DIST_EPSILON = 0.001f;

	// TODO: worry about spheres?
	const ei::Ray backfaceRay{ shadowRay.origin + shadowRay.direction * (vertex.ext().firstShadowDistance + DIST_EPSILON), shadowRay.direction };

	const auto secondHit = scene::accel_struct::first_intersection_scene_lbvh(m_sceneDesc, backfaceRay, vertex.ext().shadowHit,
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

			const auto thirdHit = scene::accel_struct::first_intersection_scene_lbvh(m_sceneDesc, silhouetteRay, secondHit.hitId,
																					 vertex.ext().lightDistance - vertex.ext().firstShadowDistance - secondHit.hitT + DIST_EPSILON);
			if(thirdHit.hitId == vertex.get_primitive_id()) {
				for(i32 i = 0; i < sharedVertices; ++i) {
					// x86_64 doesn't support atomic_fetch_add for floats FeelsBadMan
					m_importanceMap.add(vertex.ext().shadowHit.instanceId, edgeIdxFirst[i], importance);
					m_importanceMap.add(vertex.ext().shadowHit.instanceId, edgeIdxSecond[i], importance);
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

void CpuShadowSilhouettes::initialize_importance_map() {
	std::vector<scene::geometry::PolygonMeshType*> meshes;

	for(auto object : m_currentScene->get_objects()) {
		for(u32 i = 0u; i < object.first->get_lod_slot_count(); ++i) {
			if(object.first->has_lod_available(i)) {
				meshes.push_back(&object.first->get_lod(i).template get_geometry<scene::geometry::Polygons>().get_mesh());
			}
		}
	}
	// We do it in this ugly fashion because create -> move -> destroy-old doesn't work due to how OpenMesh treats properties
	m_importanceMap.~ImportanceMap();
	new(&m_importanceMap) ImportanceMap(std::move(meshes));
}

void CpuShadowSilhouettes::init_rngs(int num) {
	m_rngs.resize(num);
	// TODO: incude some global seed into the initialization
	for(int i = 0; i < num; ++i)
		m_rngs[i] = math::Rng(i);
}

} // namespace mufflon::renderer