#include "cpu_shadow_photons.hpp"
#include "util/parallel.hpp"
#include "core/renderer/random_walk.hpp"

namespace mufflon::renderer::decimaters::spm {

namespace {

template < std::size_t N >
static std::size_t get_closest_point_index(const ei::Vec3& point, const std::array<ei::Vec3, N>& points) {
	float minDistSq = std::numeric_limits<float>::max();
	std::size_t index = 0u;
	for(std::size_t i = 0u; i < N; ++i) {
		const float distSq = ei::lensq(point - points[i]);
		if(distSq < minDistSq) {
			minDistSq = distSq;
			index = i;
		}
	}
	return index;
}

inline float get_luminance(const ei::Vec3& vec) {
	constexpr ei::Vec3 LUM_WEIGHT{ 0.212671f, 0.715160f, 0.072169f };
	return ei::dot(LUM_WEIGHT, vec);
}

} // namespace 

void ShadowPhotonVisualizer::post_reset() {
	const auto resetFlags = get_reset_event();
	init_rngs(m_outputBuffer.get_num_pixels());
	if(resetFlags.is_set(ResetEvent::RENDERER_ENABLE) || resetFlags.resolution_changed()
	   || resetFlags.is_set(ResetEvent::PARAMETER)) {
		// TODO: proper capacity
		// TODO: it would be much better to not have one per light...
	
		m_densityShadowPhotonsHashgrid.clear();
		m_densityPhotonsHashgrid.clear();
		m_densityShadowPhotonsOctree.clear();
		m_densityPhotonsOctree.clear();
		const std::size_t lightCount = 1u + m_sceneDesc.lightTree.posLights.lightCount + m_sceneDesc.lightTree.dirLights.lightCount;
		m_densityShadowPhotonsHashgrid.reserve(lightCount);
		m_densityPhotonsHashgrid.reserve(lightCount);
		m_densityShadowPhotonsOctree.reserve(lightCount);
		m_densityPhotonsOctree.reserve(lightCount);
		for(std::size_t i = 0u; i < lightCount; ++i) {
			if(m_params.mode == PSpvMode::Values::OCTREE) {
				m_densityShadowPhotonsOctree.emplace_back(m_sceneDesc.aabb, 1024 * 1024 * 32, m_params.splitFactor,
														  1.f);
				m_densityPhotonsOctree.emplace_back(m_sceneDesc.aabb, 1024 * 1024 * 32, m_params.splitFactor, 1.f);
				m_densityShadowPhotonsOctree.back().clear();
				m_densityPhotonsOctree.back().clear();
			} else if(m_params.mode == PSpvMode::Values::HASHGRID) {
				m_densityShadowPhotonsHashgrid.emplace_back(1024 * 1024 * 32);
				m_densityPhotonsHashgrid.emplace_back(1024 * 1024 * 32);
				m_densityShadowPhotonsHashgrid.back().set_cell_size(m_params.cellSize);
				m_densityPhotonsHashgrid.back().set_cell_size(m_params.cellSize);
				m_densityShadowPhotonsHashgrid.back().clear();
				m_densityPhotonsHashgrid.back().clear();
			}
		}
	}

	if(resetFlags.is_set(ResetEvent::SCENARIO)) {
		m_photonMapManager.resize(m_params.maxPathLength * m_outputBuffer.get_num_pixels());
		m_photonMap = m_photonMapManager.acquire<Device::CPU>();
		m_importance = std::make_unique<data_structs::DmOctree<float>>(m_sceneDesc.aabb, 1024 * 1024 * 128, m_params.splitFactor,
																	   1.f);

		// Pre-split the octree (TODO: multithreaded)
		m_importance->set_iteration(1);
		if(m_params.mode == PSpvMode::Values::OCTREE) {
			logPedantic("Pre-splitting the octree");
			for(i32 i = 0u; i < m_sceneDesc.numInstances; ++i) {
				const auto& lod = m_sceneDesc.lods[m_sceneDesc.lodIndices[i]];
				const auto& polygon = lod.polygon;
				for(u32 v = 0u; v < polygon.numVertices; ++v) {
					const ei::Vec3 vertex = ei::transform(polygon.vertices[v], m_sceneDesc.compute_instance_to_world_transformation(i));
					// Transform the normal to world space (per-vertex normals are in object space)
					// TODO: shouldn't we rather use the geometric normal?
					const ei::Mat3x3 rotationInvScale = transpose(ei::Mat3x3{ m_sceneDesc.worldToInstance[i] });
					const ei::Vec3 vertexNormal = ei::normalize(rotationInvScale * polygon.normals[v]);
					m_importance->increase_count(vertex, vertexNormal, 1.f);
				}
			}
		}
		m_importance->clear_counters();
	}
}

void ShadowPhotonVisualizer::iterate() {
	for(auto& density : m_densityPhotonsOctree)
		density.set_iteration(1 + m_currentIteration);
	for(auto& density : m_densityShadowPhotonsOctree)
		density.set_iteration(1 + m_currentIteration);
	for(auto& density : m_densityPhotonsHashgrid)
		density.set_iteration(1 + m_currentIteration);
	for(auto& density : m_densityShadowPhotonsHashgrid)
		density.set_iteration(1 + m_currentIteration);

	const float currentMergeRadius = m_params.mergeRadius * m_sceneDesc.diagSize;
	const float mergeRadiusSq = currentMergeRadius * currentMergeRadius;
	m_importance->set_iteration(m_currentIteration + 1);
	m_photonMap.clear(currentMergeRadius * 2.0001f);

	//if(m_params.mode == PSpvMode::Values::HASHGRID || m_params.mode == PSpvMode::Values::OCTREE) {
	logPedantic("Tracing photons...");
	const u64 photonSeed = m_rngs[0].next();
	const int numPhotons = m_outputBuffer.get_num_pixels();
#pragma PARALLEL_FOR
	for(int i = 0; i < numPhotons; ++i) {
		this->trace_photon(i, numPhotons, photonSeed);
	}
	//}

	if(m_params.mode == PSpvMode::Values::OCTREE && m_params.balanceOctree) {
		logPedantic("Balancing octrees...");
		for(auto& density : m_densityPhotonsOctree)
			density.balance();
		for(auto& density : m_densityShadowPhotonsOctree)
			density.balance();
	}

	const int numPixels = m_outputBuffer.get_num_pixels();

	logPedantic("Computing shadow sizes...");
#pragma PARALLEL_FOR
	for(int pixel = 0; pixel < numPixels; ++pixel) {
		Pixel coord{ pixel % m_outputBuffer.get_width(), pixel / m_outputBuffer.get_width() };

		// Dummies, we don't care about that
		Spectrum throughput;
		VertexSample sample;
		SpvPathVertex vertex;
		SpvPathVertex::create_camera(&vertex, nullptr, m_sceneDesc.camera.get(), coord, m_rngs[pixel].next());
		int viewPathLength = 0;

		math::RndSet2_1 rnd{ m_rngs[pixel].next(), m_rngs[pixel].next() };
		float rndRoulette = math::sample_uniform(u32(m_rngs[pixel].next()));
		if(walk(m_sceneDesc, vertex, rnd, rndRoulette, true, throughput, vertex, sample) != WalkResult::HIT)
			continue;
		++viewPathLength;

		const ei::Vec3 currentPos = vertex.get_position();
		// Query photons (regular)
		auto photonIt = m_photonMap.find_first(currentPos);
		while(photonIt) {
			// Only merge photons which are within the sphere around our position.
			// and which have the correct full path length.
			const int pathLen = viewPathLength + photonIt->pathLen;
			if(pathLen >= m_params.minPathLength && pathLen <= m_params.maxPathLength
			   && ei::lensq(photonIt->position - vertex.get_position()) < mergeRadiusSq) {
				m_importance->increase_count(photonIt->closestVertexPos, photonIt->geoNormal, 1.f);
				// Attribute importance for this and all previous photons
				/*const auto* currPhotonData = &(*photonIt);
				while(currPhotonData != nullptr) {
					// Attribute importance
					// TODO: proper amount
					m_importance->increase_count(currPhotonData->closestVertexPos, get_luminance(currPhotonData->flux));
					currPhotonData = currPhotonData->prevPhoton;
				}*/
			}
			++photonIt;
		}

		display_photon_densities(coord, vertex);
	}

	if(m_outputBuffer.template is_target_enabled<ImportanceTarget>()) {
		// Query the importance
		logPedantic("Querying importance...");

#pragma PARALLEL_FOR
		for(int pixel = 0; pixel < numPixels; ++pixel) {
			Pixel coord{ pixel % m_outputBuffer.get_width(), pixel / m_outputBuffer.get_width() };

			// Dummies, we don't care about that
			Spectrum throughput;
			VertexSample sample;
			SpvPathVertex vertex;
			SpvPathVertex::create_camera(&vertex, nullptr, m_sceneDesc.camera.get(), coord, m_rngs[pixel].next());

			math::RndSet2_1 rnd{ m_rngs[pixel].next(), m_rngs[pixel].next() };
			float rndRoulette = math::sample_uniform(u32(m_rngs[pixel].next()));
			if(walk(m_sceneDesc, vertex, rnd, rndRoulette, true, throughput, vertex, sample) != WalkResult::HIT)
				continue;
			
			const auto& lod = m_sceneDesc.lods[m_sceneDesc.lodIndices[vertex.get_primitive_id().instanceId]];
			const auto& polygon = lod.polygon;

			std::optional<std::pair<ei::Vec3, ei::Vec3>> closestVertex = std::nullopt;
			if(static_cast<u32>(vertex.get_primitive_id().primId) < polygon.numTriangles) {
				// Triangle
				const auto vertices = get_triangle_vertices(vertex.get_primitive_id());
				const std::size_t closestVertexIndex = get_closest_point_index(vertex.get_position(), vertices);
				closestVertex = std::make_pair(polygon.vertices[closestVertexIndex], polygon.normals[closestVertexIndex]);
			} else if(vertex.get_primitive_id().primId < lod.numPrimitives) {
				// Quad
				const auto vertices = get_quad_vertices(vertex.get_primitive_id());
				const std::size_t closestVertexIndex = get_closest_point_index(vertex.get_position(), vertices);
				closestVertex = std::make_pair(polygon.vertices[closestVertexIndex], polygon.normals[closestVertexIndex]);
			}

			if(closestVertex.has_value()) {
				// Query the importance map
				const float importanceDensity = m_importance->get_density(closestVertex.value().first, closestVertex.value().second);
				m_outputBuffer.template contribute<ImportanceTarget>(coord, importanceDensity);
			}
		}
	}
}

void ShadowPhotonVisualizer::display_photon_densities(const ei::IVec2& coord, const SpvPathVertex& vertex) {
	ei::Vec3 photonDensities{ 0.f };
	ei::Vec3 shadowPhotonDensities{ 0.f };
	ei::Vec3 ratios{ 0.f };
	ei::Vec3 shadowGradients{ 0.f };


	if(coord.x == 0 && coord.y == 0 && m_sceneDesc.lightTree.dirLights.lightCount + m_sceneDesc.lightTree.posLights.lightCount + 1u > 3u)
		logWarning("Too many lights for direct color mapping; only the first three lights will be displaced");

	// TODO: this only works for up to three lights (including background)
	const std::size_t lightCount = 1u + m_sceneDesc.lightTree.dirLights.lightCount + m_sceneDesc.lightTree.posLights.lightCount;
	for(std::size_t i = 0u; i < std::min<std::size_t>(3u, lightCount); ++i) {
		ei::Vec3 currPhotonGradient{ 0.f };
		ei::Vec3 currShadowGradient{ 0.f };
		const float photonDensity = query_photon_density(vertex, i, &currPhotonGradient);
		const float shadowPhotonDensity = query_shadow_photon_density(vertex, i, &currShadowGradient);
		const float ratio = sdiv(photonDensity, photonDensity + shadowPhotonDensity);

		ei::Vec3 colorMask{ 0.f };
		switch(i) {
			case 0: colorMask.x = 1.f; break;
			case 1: colorMask.y = 1.f; break;
			case 2: colorMask.z = 1.f; break;
			default: break;
		}

		photonDensities += colorMask * photonDensity;
		shadowPhotonDensities += colorMask * shadowPhotonDensity;
		if(ratio < 0.5f)
			ratios += colorMask * 2.f * ratio;
		else
			ratios += colorMask * (1.f - 2.f * (ratio - 0.5f));
		if(std::abs(ratio - 0.5f) < 0.49f) {
			const float projPhotonGradient = ei::len(currPhotonGradient - ei::dot(currPhotonGradient, vertex.get_normal()) * vertex.get_normal());
			const float projShadowGradient = ei::len(currShadowGradient - ei::dot(currShadowGradient, vertex.get_normal()) * vertex.get_normal());

			const float avgGradient = (projPhotonGradient + projShadowGradient) / 2.f;
			//shadowGradients += colorMask * projShadowGradient;
			shadowGradients += colorMask * avgGradient;
		}
	}

	//m_outputBuffer.set(coord, RenderTargets::RADIANCE, ei::Vec3{ photonDensity, 0.f, shadowPhotonDensity });
	m_outputBuffer.template contribute<LightDensityTarget>(coord, photonDensities);
	m_outputBuffer.template contribute<ShadowDensityTarget>(coord, photonDensities);
	m_outputBuffer.template contribute<ShadowGradientTarget>(coord, shadowPhotonDensities);
}

void ShadowPhotonVisualizer::display_silhouette(const ei::IVec2& coord, const i32 index, const SpvPathVertex& vertex) {
	// Find silhouette
	// TODO: base this off of shadow photons?
	// TODO: connect against n light sources
#if 0
	using namespace scene::lights;

	math::Rng& rng = m_rngs[index];
	const u64 neeSeed = rng.next();
	const math::RndSet2 neeRnd = rng.next();
	u32 lightIndex = 0u;
	auto nee = scene::lights::connect(m_sceneDesc, 0, 1, neeSeed, vertex.get_position(),
									  neeRnd, &lightIndex);

	// We check if the ratio between the photon types suggests a silhouette
	const float photonDensity = query_photon_density(vertex, lightIndex);
	const float shadowPhotonDensity = query_shadow_photon_density(vertex, lightIndex);
	const float ratio = sdiv(photonDensity, photonDensity + shadowPhotonDensity);

	if(std::abs(ratio - 0.5f) < 0.45f) {

		m_outputBuffer.contribute(coord, RenderTargets::POSITION, ei::Vec3{ 1.f });

		// Check if we're actually shadowed
		Pixel projCoord;
		auto value = vertex.evaluate(nee.dir.direction, m_sceneDesc.media, projCoord);
		mAssert(!isnan(value.value.x) && !isnan(value.value.y) && !isnan(value.value.z));
		Spectrum radiance = value.value * nee.diffIrradiance;
		// TODO: use multiple NEEs
		if(any(greater(radiance, 0.0f)) && value.cosOut > 0.0f) {
			ei::Ray shadowRay{ nee.position, -nee.dir.direction };
			auto shadowHit = scene::accel_struct::first_intersection(m_sceneDesc, shadowRay, vertex.get_geometric_normal(), nee.dist);
			if(shadowHit.hitId.instanceId >= 0) {
				// We're close enough to a silhouette for our purposes -> estimate the size of the light source
				// TODO: we cannot tell which way our silhouette is oriented (since the two faces
				// do not necessarily share an edge currently), thus we conservatively use the shortest one
				if(lightIndex == 0u) {
					// TODO: what will we do with envmaps?
					m_outputBuffer.contribute(coord, RenderTargets::RADIANCE, ei::Vec3{ 1.f, 0.f, 1.f });
				} else if(lightIndex < 1u + m_sceneDesc.lightTree.dirLights.lightCount) {
					// Directional lights are always "point lights" (produce sharp shadows)
					m_outputBuffer.contribute(coord, RenderTargets::RADIANCE, ei::Vec3{ 1.f });
				} else {
					const auto lightInfo = m_sceneDesc.lightTree.posLights.get_light_info(lightIndex - static_cast<u32>(1u + m_sceneDesc.lightTree.dirLights.lightCount));

					float shortestDistanceInside;
					switch(lightInfo.second) {
						case LightType::POINT_LIGHT:
						case LightType::SPOT_LIGHT:
							// Always zero distance
							shortestDistanceInside = 0.f;
							break;
						case LightType::AREA_LIGHT_SPHERE: {
							const auto& light = *reinterpret_cast<const AreaLightSphere<Device::CPU>*>(lightInfo.first);
							// TODO: is this right? should need a correction for the smaller visible side (TODO: scaling!)
							shortestDistanceInside = 2.f * light.radius;
						}	break;
						case LightType::AREA_LIGHT_TRIANGLE: {
							const auto& light = *reinterpret_cast<const AreaLightTriangle<Device::CPU>*>(lightInfo.first);
							// TODO: Get the proper shortest distance inside!
							shortestDistanceInside = ei::len(light.posV[1u]);
						}	break;
						case LightType::AREA_LIGHT_QUAD: {
							const auto& light = *reinterpret_cast<const AreaLightQuad<Device::CPU>*>(lightInfo.first);
							// TODO: Get the proper shortest distance inside!
							shortestDistanceInside = ei::len(light.posV[2u]);
						}	break;
						default: mAssert(false);
					}
					// TODO: project that distance
					if(isnan(shortestDistanceInside))
						__debugbreak();
					m_outputBuffer.contribute(coord, RenderTargets::ALBEDO, ei::Vec3{ 1.f, shortestDistanceInside, 0.f });
					m_outputBuffer.contribute(coord, RenderTargets::RADIANCE, ei::Vec3{ ei::exp(-2.f * shortestDistanceInside) });
				}
			}
		}
	}
#endif
}

bool ShadowPhotonVisualizer::trace_shadow_silhouette(const ei::Ray& shadowRay, const float shadowDistance,
													 const float lightDistance, const scene::PrimitiveHandle& shadowHit,
													 const SpvPathVertex& vertex) const {
	constexpr float DIST_EPSILON = 0.001f;

	// TODO: worry about spheres?
	ei::Ray backfaceRay{ shadowRay.origin + shadowRay.direction * (lightDistance + DIST_EPSILON), shadowRay.direction };

	// TODO: which one is the correct normal?
	const auto secondHit = scene::accel_struct::first_intersection(m_sceneDesc, backfaceRay, vertex.get_geometric_normal(),
																   lightDistance - shadowDistance + DIST_EPSILON);
	// We absolutely have to have a second hit - either us (since we weren't first hit) or something else
	if(secondHit.hitId.instanceId >= 0 && secondHit.hitId != vertex.get_primitive_id()
	   && secondHit.hitId.instanceId == shadowHit.instanceId) {
		// Check for silhouette - get the vertex indices of the primitives
		const auto& obj = m_sceneDesc.lods[m_sceneDesc.lodIndices[shadowHit.instanceId]];
		const i32 firstNumVertices = shadowHit.primId < (i32)obj.polygon.numTriangles ? 3 : 4;
		const i32 secondNumVertices = secondHit.hitId.primId < (i32)obj.polygon.numTriangles ? 3 : 4;
		const i32 firstPrimIndex = shadowHit.primId - (shadowHit.primId < (i32)obj.polygon.numTriangles
																	? 0 : (i32)obj.polygon.numTriangles);
		const i32 secondPrimIndex = secondHit.hitId.primId - (secondHit.hitId.primId < (i32)obj.polygon.numTriangles
															  ? 0 : (i32)obj.polygon.numTriangles);
		const i32 firstVertOffset = shadowHit.primId < (i32)obj.polygon.numTriangles ? 0 : 3 * (i32)obj.polygon.numTriangles;
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
			ei::Ray silhouetteRay{ shadowRay.origin + shadowRay.direction * (shadowDistance + secondHit.distance), shadowRay.direction };

			/*const auto thirdHit = scene::accel_struct::any_intersection(m_sceneDesc, silhouetteRay, secondHit.hitId,
																		vertex.ext().lightDistance - vertex.ext().firstShadowDistance - secondHit.distance + DIST_EPSILON);
			if(!thirdHit) {*/
			const auto thirdHit = scene::accel_struct::first_intersection(m_sceneDesc, silhouetteRay, secondHit.normal,
																		  lightDistance - shadowDistance - secondHit.distance + DIST_EPSILON);
			if(thirdHit.hitId == vertex.get_primitive_id()) {
				/*for(i32 i = 0; i < sharedVertices; ++i) {
					const auto lodIdx = m_sceneDesc.lodIndices[shadowHit.instanceId];
					record_silhouette_vertex_contribution(importances[lodIdx], sums[lodIdx], edgeIdxFirst[i], importance);
					record_silhouette_vertex_contribution(importances[lodIdx], sums[lodIdx], edgeIdxSecond[i], importance);
				}*/
				return true;
			} else {
				mAssert(thirdHit.hitId.instanceId >= 0);
				// TODO: store a shadow photon?
			}
		}
	}
	return false;
}

float ShadowPhotonVisualizer::query_photon_density(const SpvPathVertex& vertex, const std::size_t lightIndex,
												   ei::Vec3* gradient) const {
	switch(m_params.interpolation) {
		/*case PInterpolate::Values::LINEAR:
			if(m_params.mode == PSpvMode::Values::HASHGRID)
				return m_densityPhotonsHashgrid[lightIndex].get_density_interpolated<false>(vertex.get_position(),
																							vertex.get_normal(), gradient);
			else if(m_params.mode == PSpvMode::Values::OCTREE)
				return m_densityPhotonsOctree[lightIndex].get_density_interpolated<false>(vertex.get_position(),
																						  vertex.get_normal(), gradient);
			else
				return 0.f;
		case PInterpolate::Values::SMOOTHSTEP:
			if(m_params.mode == PSpvMode::Values::HASHGRID)
				return m_densityPhotonsHashgrid[lightIndex].get_density_interpolated<true>(vertex.get_position(),
																						   vertex.get_normal(), gradient);
			else if(m_params.mode == PSpvMode::Values::OCTREE)
				return m_densityPhotonsOctree[lightIndex].get_density_interpolated<true>(vertex.get_position(),
																						 vertex.get_normal(), gradient);
			else
				return 0.f;*/
		default:
			if(m_params.mode == PSpvMode::Values::HASHGRID)
				return m_densityPhotonsHashgrid[lightIndex].get_density(vertex.get_position(),
																		vertex.get_normal());
			else if(m_params.mode == PSpvMode::Values::OCTREE)
				return m_densityPhotonsOctree[lightIndex].get_density(vertex.get_position(),
																	  vertex.get_normal());
			else
				return 0.f;
	}
}

float ShadowPhotonVisualizer::query_shadow_photon_density(const SpvPathVertex& vertex, const std::size_t lightIndex,
														  ei::Vec3* gradient) const {
	switch(m_params.interpolation) {
		/*case PInterpolate::Values::LINEAR:
			if(m_params.mode == PSpvMode::Values::HASHGRID)
				return m_densityShadowPhotonsHashgrid[lightIndex].get_density_interpolated<false>(vertex.get_position(),
																								  vertex.get_normal(), gradient);
			else if(m_params.mode == PSpvMode::Values::OCTREE)
				return m_densityShadowPhotonsOctree[lightIndex].get_density_interpolated<false>(vertex.get_position(),
																								vertex.get_normal(), gradient);
			else
				return 0.f;
		case PInterpolate::Values::SMOOTHSTEP:
			if(m_params.mode == PSpvMode::Values::HASHGRID)
				return m_densityShadowPhotonsHashgrid[lightIndex].get_density_interpolated<true>(vertex.get_position(),
																								 vertex.get_normal(), gradient);
			else if(m_params.mode == PSpvMode::Values::OCTREE)
				return m_densityShadowPhotonsOctree[lightIndex].get_density_interpolated<true>(vertex.get_position(),
																							   vertex.get_normal(), gradient);
			else
				return 0.f;*/
		default:
			if(m_params.mode == PSpvMode::Values::HASHGRID)
				return m_densityShadowPhotonsHashgrid[lightIndex].get_density(vertex.get_position(),
																			  vertex.get_normal());
			else if(m_params.mode == PSpvMode::Values::OCTREE)
				return m_densityShadowPhotonsOctree[lightIndex].get_density(vertex.get_position(),
																			vertex.get_normal());
			else
				return 0.f;
	}
}

void ShadowPhotonVisualizer::trace_photon(const int idx, const int numPhotons, const u64 seed) {
	math::RndSet2_1 rndStart{ m_rngs[idx].next(), m_rngs[idx].next() };
	u32 lightIndex = 0u;
	scene::lights::Emitter p = scene::lights::emit(m_sceneDesc, idx, numPhotons, seed, rndStart, &lightIndex);


	SpvPathVertex vertex[2];
	SpvPathVertex::create_light(&vertex[0], nullptr, p);
	Spectrum throughput;

	// TODO: store path length in octree?
	int pathLen = 0;
	int currentV = 0;
	int otherV = 1;
	PhotonDesc* prevPhoton = nullptr;
	do {
		// Walk
		math::RndSet2_1 rnd{ m_rngs[idx].next(), m_rngs[idx].next() };
		float rndRoulette = math::sample_uniform(u32(m_rngs[idx].next()));
		VertexSample sample;
		if(walk(m_sceneDesc, vertex[currentV], rnd, rndRoulette, true, throughput, vertex[otherV], sample) != WalkResult::HIT)
			break;
		++pathLen;
		currentV = otherV;
		otherV = 1 - currentV;

		const auto closestVertex = get_closest_vertex(vertex[currentV].get_position(), vertex[currentV].get_primitive_id());
		if(closestVertex.has_value()) {
			// Deposit regular photon
			prevPhoton = m_photonMap.insert(vertex[currentV].get_position(), {
				vertex[currentV].get_incident_direction(), pathLen,
				throughput / numPhotons, vertex[currentV].get_geometric_normal(),
				vertex[currentV].get_position(), closestVertex.value(), prevPhoton
			});
		}

		// For density estimates (TODO: do we really need this if we have the photon map?)
		if(m_params.mode == PSpvMode::Values::HASHGRID)
			m_densityPhotonsHashgrid[lightIndex].increase_count(vertex[currentV].get_position());
		else if(m_params.mode == PSpvMode::Values::OCTREE)
			m_densityPhotonsOctree[lightIndex].increase_count(vertex[currentV].get_position(),
															  vertex[currentV].get_geometric_normal());

		// Trace shadow photon
		if(pathLen <= 1) {
			std::optional<ei::Vec3> shadowPos = trace_shadow_photon(vertex[currentV], idx);
			if(shadowPos.has_value()) {
				if(m_params.mode == PSpvMode::Values::HASHGRID)
					m_densityShadowPhotonsHashgrid[lightIndex].increase_count(shadowPos.value());
				else if(m_params.mode == PSpvMode::Values::OCTREE)
					m_densityShadowPhotonsOctree[lightIndex].increase_count(shadowPos.value(),
																			vertex[currentV].get_geometric_normal());
			}
		}
	} while(pathLen < m_params.maxPathLength - 1); // -1 because there is at least one segment on the view path
}

std::optional<ei::Vec3> ShadowPhotonVisualizer::trace_shadow_photon(const SpvPathVertex& vertex, const int idx) {
	constexpr float DIST_EPSILON = 0.001f;
	// We may need to trace for quite some time if there are transparent objects
	int frontFaceCounter = 1;

	ei::Ray ray{ vertex.get_position(), vertex.get_incident_direction() };

	do {
		// Step a bit away from the surface
		ray.origin += ray.direction * DIST_EPSILON;
		scene::accel_struct::RayIntersectionResult nextHit =
			scene::accel_struct::first_intersection(m_sceneDesc, ray, vertex.get_geometric_normal(), scene::MAX_SCENE_SIZE);
		if(nextHit.hitId.instanceId < 0)
			return std::nullopt;

		// We need the normal to distinguish front/back

		if(ei::dot(nextHit.normal, ray.direction) < 0)
			++frontFaceCounter;
		else
			--frontFaceCounter;

		ray.origin = ray.origin + nextHit.distance * ray.direction;

	} while(frontFaceCounter > 0u);

	// Last walk
		// Step a bit away from the surface
	ray.origin += ray.direction * DIST_EPSILON;
	scene::accel_struct::RayIntersectionResult nextHit =
		scene::accel_struct::first_intersection(m_sceneDesc, ray, vertex.get_geometric_normal(), scene::MAX_SCENE_SIZE);
	if(nextHit.hitId.instanceId < 0)
		return std::nullopt;

	return ray.origin + nextHit.distance * ray.direction;
}

std::array<ei::Vec3, 3u> ShadowPhotonVisualizer::get_triangle_vertices(const scene::PrimitiveHandle& hitId) const {
	const auto& lod = m_sceneDesc.lods[m_sceneDesc.lodIndices[hitId.instanceId]];
	const auto& polygon = lod.polygon;
	mAssert(static_cast<u32>(hitId.primId) < polygon.numTriangles);
	const i32 vertexOffset = 3 * hitId.primId;
	const ei::IVec3 indices{
		polygon.vertexIndices[vertexOffset + 0],
		polygon.vertexIndices[vertexOffset + 1],
		polygon.vertexIndices[vertexOffset + 2]
	};
	const auto instToWorld = m_sceneDesc.compute_instance_to_world_transformation(hitId.instanceId);
	return std::array<ei::Vec3, 3u>{{
		ei::transform(polygon.vertices[indices.x], instToWorld),
		ei::transform(polygon.vertices[indices.y], instToWorld),
		ei::transform(polygon.vertices[indices.z], instToWorld)
	}};
}

std::array<ei::Vec3, 4u> ShadowPhotonVisualizer::get_quad_vertices(const scene::PrimitiveHandle& hitId) const {
	const auto& lod = m_sceneDesc.lods[m_sceneDesc.lodIndices[hitId.instanceId]];
	const auto& polygon = lod.polygon;
	mAssert(hitId.primId < lod.numPrimitives && static_cast<u32>(hitId.primId) >= polygon.numTriangles);
	const i32 vertexOffset = 3 * polygon.numTriangles + 4 * (hitId.primId - polygon.numTriangles);
	const ei::IVec4 indices{
		polygon.vertexIndices[vertexOffset + 0],
		polygon.vertexIndices[vertexOffset + 1],
		polygon.vertexIndices[vertexOffset + 2],
		polygon.vertexIndices[vertexOffset + 3]
	};
	const auto instToWorld = m_sceneDesc.compute_instance_to_world_transformation(hitId.instanceId);
	return std::array<ei::Vec3, 4u>{{
		ei::transform(polygon.vertices[indices.x], instToWorld),
		ei::transform(polygon.vertices[indices.y], instToWorld),
		ei::transform(polygon.vertices[indices.z], instToWorld),
		ei::transform(polygon.vertices[indices.w], instToWorld)
	}};
}

std::optional<ei::Vec3> ShadowPhotonVisualizer::get_closest_vertex(const ei::Vec3& hitpoint,
																   const scene::PrimitiveHandle& hitId) const {
	const auto& lod = m_sceneDesc.lods[m_sceneDesc.lodIndices[hitId.instanceId]];
	const auto& polygon = lod.polygon;
	if(static_cast<u32>(hitId.primId) < polygon.numTriangles) {
		// Triangle
		const auto vertices = get_triangle_vertices(hitId);
		return vertices[get_closest_point_index(hitpoint, vertices)];
	} else if(hitId.primId < lod.numPrimitives) {
		// Quad
		const auto vertices = get_quad_vertices(hitId);
		return vertices[get_closest_point_index(hitpoint, vertices)];
	} else {
		return std::nullopt;
	}
}

void ShadowPhotonVisualizer::init_rngs(const int num) {
	m_rngs.resize(num);
	int seed = m_params.seed * (num + 1);
	for(int i = 0; i < num; ++i)
		m_rngs[i] = math::Rng(i + seed);
}

} // namespace mufflon::renderer::decimaters::spm