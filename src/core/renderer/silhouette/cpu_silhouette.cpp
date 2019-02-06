#pragma once

#include "cpu_silhouette.hpp"
#include "util/parallel.hpp"
#include "core/renderer/output_handler.hpp"
#include "core/renderer/path_util.hpp"
#include "core/renderer/random_walk.hpp"
#include "core/scene/descriptors.hpp"
#include "core/scene/scene.hpp"
#include "core/scene/lights/lights.hpp"
#include "core/scene/lights/light_sampling.hpp"
#include "core/scene/lights/light_tree_sampling.hpp"
#include <random>

namespace mufflon::renderer {

using PtPathVertex = PathVertex<u8, 4>;

CpuShadowSilhouettes::CpuShadowSilhouettes()
{
	// TODO: init one RNG per thread?
	std::random_device rndDev;
	m_rngs.emplace_back(static_cast<u32>(rndDev()));
}

void CpuShadowSilhouettes::iterate(OutputHandler& outputBuffer) {
	// (Re) create the random number generators
	if(m_rngs.size() != outputBuffer.get_num_pixels()
	   || m_reset)
		init_rngs(outputBuffer.get_num_pixels());

	RenderBuffer<Device::CPU> buffer = outputBuffer.begin_iteration<Device::CPU>(m_reset);
	if(m_reset) {
		// TODO: reset output buffer
		// Reacquire scene descriptor (partially?)
		m_sceneDesc = m_currentScene->get_descriptor<Device::CPU>({}, {}, {}, outputBuffer.get_resolution());
	}
	m_reset = false;

	// TODO: call sample in a parallel way for each output pixel
	// TODO: better pixel order?
	// TODO: different scheduling?
#pragma PARALLEL_FOR
	for(int pixel = 0; pixel < outputBuffer.get_num_pixels(); ++pixel) {
		this->sample(Pixel{ pixel % outputBuffer.get_width(), pixel / outputBuffer.get_width() }, buffer, m_sceneDesc);
	}

	outputBuffer.end_iteration<Device::CPU>();
}

void CpuShadowSilhouettes::reset() {
	this->m_reset = true;
}

void CpuShadowSilhouettes::sample(const Pixel coord, RenderBuffer<Device::CPU>& outputBuffer,
						   const scene::SceneDescriptor<Device::CPU>& scene) {
	int pixel = coord.x + coord.y * outputBuffer.get_width();

	//m_params.maxPathLength = 2;

	math::Throughput throughput{ ei::Vec3{1.0f}, 1.0f };
	u8 vertexBuffer[256]; // TODO: depends on materials::MAX_MATERIAL_PARAMETER_SIZE
	PtPathVertex* vertex = as<PtPathVertex>(vertexBuffer);
	// Create a start for the path
	int s = PtPathVertex::create_camera(vertex, vertex, scene.camera.get(), coord, m_rngs[pixel].next());
	mAssertMsg(s < 256, "vertexBuffer overflow.");

	// Direct hit
	scene::Point lastPosition = vertex->get_position();
	math::RndSet2_1 rnd{ m_rngs[pixel].next(), m_rngs[pixel].next() };
	math::DirectionSample lastDir;
	if(!walk(scene, *vertex, rnd, -1.0f, false, throughput, vertex, lastDir)) {
		/*if(throughput.weight != Spectrum{ 0.f }) {
			// Missed scene - sample background
			auto background = evaluate_background(scene.lightTree.background, lastDir.direction);
			if(any(greater(background.value, 0.0f))) {
				float mis = 1.0f / (1.0f + background.pdfB / lastDir.pdf);
				background.value *= mis;
				outputBuffer.contribute(coord, throughput, background.value,
										ei::Vec3{ 0, 0, 0 }, ei::Vec3{ 0, 0, 0 },
										ei::Vec3{ 0, 0, 0 });
			}
		}*/
	} else {
		// TODO: multiple lights!
		if(scene.lightTree.posLights.lightCount != 1)
			throw std::runtime_error("Silhouettes for != 1 positional light are not implemented yet");

		const char* lightMem = scene.lightTree.posLights.memory;
		const auto lightType = scene.lightTree.posLights.root.type;
		const ei::Vec3 lightCenter = scene::lights::lighttree_detail::get_center(lightMem, lightType);
		ei::Vec3 fromLight = vertex->get_position() - lightCenter;
		const float lightDist = ei::len(fromLight);
		fromLight /= lightDist;

		// Conditions: either point light or in cone of spot light
		bool inLight = lightType == static_cast<u16>(scene::lights::LightType::POINT_LIGHT);
		if(lightType == static_cast<u16>(scene::lights::LightType::SPOT_LIGHT)) {
			const auto& spotLight = as<scene::lights::SpotLight>(lightMem);
			if(ei::dot(fromLight, spotLight->direction) >= spotLight->cosThetaMax)
				inLight = true;
		}

		if(inLight) {
			constexpr float DIST_EPSILON = 0.001f;

			const ei::Ray shadowRay{ lightCenter, fromLight };

			const auto firstHit = scene::accel_struct::first_intersection_scene_lbvh(scene, shadowRay, vertex->get_primitive_id(), lightDist + DIST_EPSILON);
			// TODO: worry about spheres?
			if(firstHit.hitId.instanceId >= 0 && firstHit.hitId != vertex->get_primitive_id()) {
				const ei::Ray backfaceRay{ lightCenter + fromLight * firstHit.hitT, fromLight };

				const auto secondHit = scene::accel_struct::first_intersection_scene_lbvh(scene, backfaceRay, firstHit.hitId,
																						  lightDist - firstHit.hitT + DIST_EPSILON);
				// We absolutely have to have a second hit - either us (since we weren't first hit) or something else
				if(secondHit.hitId.instanceId >= 0 && secondHit.hitId != vertex->get_primitive_id()
				   && secondHit.hitId.instanceId == firstHit.hitId.instanceId) {
					// Check for silhouette - get the vertex indices of the primitives
					const auto& obj = scene.lods[scene.lodIndices[firstHit.hitId.instanceId]];
					const i32 firstNumVertices = firstHit.hitId.primId < (i32)obj.polygon.numTriangles ? 3 : 4;
					const i32 secondNumVertices = secondHit.hitId.primId < (i32)obj.polygon.numTriangles ? 3 : 4;
					const i32 firstPrimIndex = firstHit.hitId.primId - (firstHit.hitId.primId < (i32)obj.polygon.numTriangles
						? 0 : (i32)obj.polygon.numTriangles);
					const i32 secondPrimIndex = secondHit.hitId.primId - (secondHit.hitId.primId < (i32)obj.polygon.numTriangles
						? 0 : (i32)obj.polygon.numTriangles);
					const i32 firstVertOffset = firstHit.hitId.primId < (i32)obj.polygon.numTriangles ? 0 : 3 * (i32)obj.polygon.numTriangles;
					const i32 secondVertOffset = secondHit.hitId.primId < (i32)obj.polygon.numTriangles ? 0 : 3 * (i32)obj.polygon.numTriangles;

					// Check if we have "shared" vertices: cannot do it by index, since they might be
					// split vertices, but instead need to go by proximity
					i32 sharedVertices = 0;
					for(i32 i0 = 0; i0 < firstNumVertices; ++i0) {
						for(i32 i1 = 0; i1 < secondNumVertices; ++i1) {
							const i32 idx0 = obj.polygon.vertexIndices[firstVertOffset + firstNumVertices * firstPrimIndex + i0];
							const i32 idx1 = obj.polygon.vertexIndices[secondVertOffset + secondNumVertices * secondPrimIndex + i1];
							const ei::Vec3& p0 = obj.polygon.vertices[idx0];
							const ei::Vec3& p1 = obj.polygon.vertices[idx1];
							if(ei::lensq(p0 - p1) < DIST_EPSILON)
								++sharedVertices;
							if(sharedVertices >= 1)
								break;
						}
					}

					if(sharedVertices >= 1) {
						// Got a silhouette
						// TODO: store shadow edge importance
						const ei::Ray silhouetteRay{ lightCenter + fromLight * (firstHit.hitT + secondHit.hitT), fromLight };

						const auto thirdHit = scene::accel_struct::first_intersection_scene_lbvh(scene, silhouetteRay, secondHit.hitId,
																								 lightDist - firstHit.hitT - secondHit.hitT + DIST_EPSILON);
						if(thirdHit.hitId == vertex->get_primitive_id()) {
							outputBuffer.contribute(coord, throughput, ei::Vec3{ 1.f, 0.0f, 0.0f }, vertex->get_position(),
													vertex->get_normal(), vertex->get_albedo());
						} else {
							mAssert(thirdHit.hitId.instanceId >= 0);
							// Shadow - silhouette blocked by other object
							outputBuffer.contribute(coord, throughput, ei::Vec3{ 0.0f, 1.0f, 0.0f }, vertex->get_position(),
													vertex->get_normal(), vertex->get_albedo());
						}
					} else {
						// Not a silhouette - we don't care
						outputBuffer.contribute(coord, throughput, ei::Vec3{ 0.f }, vertex->get_position(),
												vertex->get_normal(), vertex->get_albedo());
					}
				}
			} else {
				// Not shadowed
				outputBuffer.contribute(coord, throughput, ei::Vec3{ 1.f }, vertex->get_position(),
										vertex->get_normal(), vertex->get_albedo());
			}
		} else {
			// No light at all
			outputBuffer.contribute(coord, throughput, ei::Vec3{ 0.f }, vertex->get_position(),
									vertex->get_normal(), vertex->get_albedo());
		}
	}

#if 0
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
			auto nee = connect(scene.lightTree, 0, 1, neeSeed,
							   vertex->get_position(), m_currentScene->get_bounding_box(),
							   neeRnd, scene::lights::guide_flux);
			auto value = vertex->evaluate(nee.direction, scene.media);
			mAssert(!isnan(value.value.x) && !isnan(value.value.y) && !isnan(value.value.z));
			Spectrum radiance = value.value * nee.diffIrradiance;
			if(any(greater(radiance, 0.0f)) && value.cosOut > 0.0f) {
				bool anyhit = scene::accel_struct::any_intersection_scene_lbvh<Device::CPU>(
					scene, { vertex->get_position() , nee.direction },
					vertex->get_primitive_id(), nee.dist);
				if(!anyhit) {
					AreaPdf hitPdf = value.pdfF.to_area_pdf(nee.cosOut, nee.distSq);
					float mis = 1.0f / (1.0f + hitPdf / nee.creationPdf);
					mAssert(!isnan(mis));
					outputBuffer.contribute(coord, throughput, { Spectrum{1.0f}, 1.0f },
											value.cosOut, radiance * mis);
				}
			}
		}

		// Walk
		scene::Point lastPosition = vertex->get_position();
		math::RndSet2_1 rnd{ m_rngs[pixel].next(), m_rngs[pixel].next() };
		math::DirectionSample lastDir;
		if(!walk(scene, *vertex, rnd, -1.0f, false, throughput, vertex, lastDir)) {
			if(throughput.weight != Spectrum{ 0.f }) {
				// Missed scene - sample background
				auto background = evaluate_background(scene.lightTree.background, lastDir.direction);
				if(any(greater(background.value, 0.0f))) {
					float mis = 1.0f / (1.0f + background.pdfB / lastDir.pdf);
					background.value *= mis;
					outputBuffer.contribute(coord, throughput, background.value,
											ei::Vec3{ 0, 0, 0 }, ei::Vec3{ 0, 0, 0 },
											ei::Vec3{ 0, 0, 0 });
				}
			}
			break;
		}
		++pathLen;

		// Evaluate direct hit of area ligths
		if(pathLen <= m_params.maxPathLength) {
			Spectrum emission = vertex->get_emission();
			if(emission != 0.0f) {
				AreaPdf backwardPdf = connect_pdf(scene.lightTree, vertex->get_primitive_id(),
												  vertex->get_surface_params(),
												  lastPosition, scene::lights::guide_flux);
				float mis = pathLen == 1 ? 1.0f
					: 1.0f / (1.0f + backwardPdf / vertex->get_incident_pdf());
				emission *= mis;
			}
			outputBuffer.contribute(coord, throughput, emission, vertex->get_position(),
									vertex->get_normal(), vertex->get_albedo());
		}
	} while(pathLen < m_params.maxPathLength);
#endif // 0
}

void CpuShadowSilhouettes::init_rngs(int num) {
	m_rngs.resize(num);
	// TODO: incude some global seed into the initialization
	for(int i = 0; i < num; ++i)
		m_rngs[i] = math::Rng(i);
}

void CpuShadowSilhouettes::load_scene(scene::SceneHandle scene, const ei::IVec2& resolution) {
	if(scene != m_currentScene) {
		m_currentScene = scene;
		m_sceneDesc = m_currentScene->get_descriptor<Device::CPU>({}, {}, {}, resolution);
		m_reset = true;
	}
}

} // namespace mufflon::renderer