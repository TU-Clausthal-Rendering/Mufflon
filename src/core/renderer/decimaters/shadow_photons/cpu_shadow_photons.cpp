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
	if(resetFlags.is_set(ResetEvent::RENDERER_ENABLE)) {
		m_densityShadowPhotons = std::make_unique<data_structs::DmHashGrid>(1024 * 1024 * 32,
																			m_params.mergeRadius * m_sceneDesc.diagSize * 2.0001f);
		m_densityPhotons = std::make_unique<data_structs::DmHashGrid>(1024 * 1024 * 32,
																	  m_params.mergeRadius * m_sceneDesc.diagSize * 2.0001f);
	} else {
		// TODO: proper cell size
		m_densityShadowPhotons->set_cell_size(m_params.mergeRadius * m_sceneDesc.diagSize * 2.0001f);
		m_densityPhotons->set_cell_size(m_params.mergeRadius * m_sceneDesc.diagSize * 2.0001f);
	}
	m_densityShadowPhotons->clear();
	m_densityPhotons->clear();
}

void ShadowPhotonVisualizer::iterate() {
	m_densityShadowPhotons->set_density_scale(1.0f / (m_currentIteration + 1));

	const u64 photonSeed = m_rngs[0].next();
	const int numPhotons = m_outputBuffer.get_num_pixels();
#pragma PARALLEL_FOR
	for(int i = 0; i < numPhotons; ++i) {
		this->trace_photon(i, numPhotons, photonSeed);
	}

	// Fetch densities

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

		const float photonDensity = query_photon_density(vertex);
		const float shadowPhotonDensity = query_shadow_photon_density(vertex);
		const float ratio = photonDensity / std::max(1.f, (photonDensity + shadowPhotonDensity));

		m_outputBuffer.set(coord, RenderTargets::RADIANCE, ei::Vec3{ photonDensity, 0.f, shadowPhotonDensity });
		m_outputBuffer.set(coord, RenderTargets::POSITION, ei::Vec3{ photonDensity });
		m_outputBuffer.set(coord, RenderTargets::ALBEDO, ei::Vec3{ shadowPhotonDensity });
		if(ratio < 0.5f)
			m_outputBuffer.set(coord, RenderTargets::NORMAL, ei::Vec3{ 2.f * ratio });
		else
			m_outputBuffer.set(coord, RenderTargets::NORMAL, ei::Vec3{ 1.f - 2.f * (ratio - 0.5f) });
	}
}

float ShadowPhotonVisualizer::query_photon_density(const SpvPathVertex& vertex) {
	if(m_params.pointSampling)
		return m_densityPhotons->get_density(vertex.get_position(), vertex.get_normal());
	else
		return m_densityPhotons->get_density_interpolated(vertex.get_position(), vertex.get_normal());
}

float ShadowPhotonVisualizer::query_shadow_photon_density(const SpvPathVertex& vertex) {
	if(m_params.pointSampling)
		return m_densityShadowPhotons->get_density(vertex.get_position(), vertex.get_normal());
	else
		return m_densityShadowPhotons->get_density_interpolated(vertex.get_position(), vertex.get_normal());
}

void ShadowPhotonVisualizer::trace_photon(const int idx, const int numPhotons, const u64 seed) {
	math::RndSet2_1 rndStart{ m_rngs[idx].next(), m_rngs[idx].next() };
	scene::lights::Emitter p = scene::lights::emit(m_sceneDesc, idx, numPhotons, seed, rndStart);
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
		m_densityPhotons->increase_count(vertex[currentV].get_position());

		// Trace shadow photon
		std::optional<ei::Vec3> shadowPos = trace_shadow_photon(vertex[currentV], idx);
		if(shadowPos.has_value())
			m_densityShadowPhotons->increase_count(shadowPos.value());
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