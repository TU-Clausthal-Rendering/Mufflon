#include "cpu_shadow_photons.hpp"
#include "util/parallel.hpp"
#include "core/renderer/random_walk.hpp"

namespace mufflon::renderer::decimaters::spm {

ShadowPhotonVisualizer::ShadowPhotonVisualizer() {

}

ShadowPhotonVisualizer::~ShadowPhotonVisualizer() {

}

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
				m_densityShadowPhotonsOctree.emplace_back(m_sceneDesc.aabb, 1024 * 1024 * 32, m_params.splitFactor);
				m_densityPhotonsOctree.emplace_back(m_sceneDesc.aabb, 1024 * 1024 * 32, m_params.splitFactor);
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

	if(m_params.mode == PSpvMode::Values::HASHGRID || m_params.mode == PSpvMode::Values::OCTREE) {
		const u64 photonSeed = m_rngs[0].next();
		const int numPhotons = m_outputBuffer.get_num_pixels();
#pragma PARALLEL_FOR
		for(int i = 0; i < numPhotons; ++i) {
			this->trace_photon(i, numPhotons, photonSeed);
		}
	}

	if(m_params.mode == PSpvMode::Values::OCTREE) {
		//m_densityPhotons->balance();
		//m_densityShadowPhotons->balance();
	}

	const int numPixels = m_outputBuffer.get_num_pixels();
#pragma PARALLEL_FOR
	for(int pixel = 0; pixel < numPixels; ++pixel) {
		Pixel coord{ pixel % m_outputBuffer.get_width(), pixel / m_outputBuffer.get_width() };

		// Dummies, we don't care about that
		math::Throughput throughput;
		VertexSample sample;
		SpvPathVertex vertex;
		SpvPathVertex::create_camera(&vertex, nullptr, m_sceneDesc.camera.get(), coord, m_rngs[pixel].next());

		math::RndSet2_1 rnd{ m_rngs[pixel].next(), m_rngs[pixel].next() };
		float rndRoulette = math::sample_uniform(u32(m_rngs[pixel].next()));
		if(walk(m_sceneDesc, vertex, rnd, rndRoulette, true, throughput, vertex, sample) != WalkResult::HIT)
			continue;

		if(m_sceneDesc.lightTree.dirLights.lightCount + m_sceneDesc.lightTree.posLights.lightCount + 1u > 3u)
			logWarning("Too many lights for direct color mapping; only the first three lights will be displaced");

		if(m_params.mode == PSpvMode::Values::HASHGRID || m_params.mode == PSpvMode::Values::OCTREE) {
			display_photon_densities(coord, vertex);
		} else {
			// Find silhouette
			// TODO
		}
	}
}

void ShadowPhotonVisualizer::display_photon_densities(const ei::IVec2& coord, const SpvPathVertex& vertex) {
	ei::Vec3 photonDensities{ 0.f };
	ei::Vec3 shadowPhotonDensities{ 0.f };
	ei::Vec3 ratios{ 0.f };
	ei::Vec3 shadowGradients{ 0.f };
	// TODO: this only works for up to three lights (including background)
	const std::size_t lightCount = 1u + m_sceneDesc.lightTree.dirLights.lightCount + m_sceneDesc.lightTree.posLights.lightCount;
	for(std::size_t i = 0u; i < std::min<std::size_t>(3u, lightCount); ++i) {
		ei::Vec3 currShadowGradient{ 0.f };
		const float photonDensity = query_photon_density(vertex, i);
		const float shadowPhotonDensity = query_shadow_photon_density(vertex, i, &currShadowGradient);
		const float ratio = sdiv(photonDensity, photonDensity + shadowPhotonDensity);
		const float projShadowGradient = ei::len(currShadowGradient - ei::dot(currShadowGradient, vertex.get_normal()) * vertex.get_normal());

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
		if(std::abs(ratio - 0.5f) < 0.25f)
			shadowGradients += colorMask * projShadowGradient;
	}

	//m_outputBuffer.set(coord, RenderTargets::RADIANCE, ei::Vec3{ photonDensity, 0.f, shadowPhotonDensity });
	m_outputBuffer.set(coord, RenderTargets::RADIANCE, shadowGradients);
	m_outputBuffer.set(coord, RenderTargets::POSITION, photonDensities);
	m_outputBuffer.set(coord, RenderTargets::ALBEDO, shadowPhotonDensities);
	m_outputBuffer.set(coord, RenderTargets::NORMAL, ratios);
}

float ShadowPhotonVisualizer::query_photon_density(const SpvPathVertex& vertex, const std::size_t lightIndex,
												   ei::Vec3* gradient) const {
	switch(m_params.interpolation) {
		case PInterpolate::Values::LINEAR:
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
				return 0.f;
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
		case PInterpolate::Values::LINEAR:
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
				return 0.f;
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
	u32 lightIndex;
	scene::lights::Emitter p = scene::lights::emit(m_sceneDesc, idx, numPhotons, seed, rndStart, &lightIndex);
	if(p.type == scene::lights::LightType::DIRECTIONAL_LIGHT)
		lightIndex += 1u; // Offset for envmap light
	else if(p.type != scene::lights::LightType::DIRECTIONAL_LIGHT)
		lightIndex += static_cast<u32>(1u + m_sceneDesc.lightTree.dirLights.lightCount); // Offset for envmap + dirlights


	SpvPathVertex vertex[2];
	SpvPathVertex::create_light(&vertex[0], nullptr, p);
	math::Throughput throughput;

	// TODO: store path length in octree?
	int pathLen = 0;
	int currentV = 0;
	int otherV = 1;
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

		// Deposit regular photon
		if(m_params.mode == PSpvMode::Values::HASHGRID)
			m_densityPhotonsHashgrid[lightIndex].increase_count(vertex[currentV].get_position());
		else if(m_params.mode == PSpvMode::Values::OCTREE)
			m_densityPhotonsOctree[lightIndex].increase_count(vertex[currentV].get_position());

		// Trace shadow photon
		std::optional<ei::Vec3> shadowPos = trace_shadow_photon(vertex[currentV], idx);
		if(shadowPos.has_value()) {
			if(m_params.mode == PSpvMode::Values::HASHGRID)
				m_densityShadowPhotonsHashgrid[lightIndex].increase_count(shadowPos.value());
			else if(m_params.mode == PSpvMode::Values::OCTREE)
				m_densityShadowPhotonsOctree[lightIndex].increase_count(shadowPos.value());
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

void ShadowPhotonVisualizer::init_rngs(const int num) {
	m_rngs.resize(num);
	int seed = m_params.seed * (num + 1);
	for(int i = 0; i < num; ++i)
		m_rngs[i] = math::Rng(i + seed);
}

} // namespace mufflon::renderer::decimaters::spm