#pragma once

#include "core/renderer/renderer_base.hpp"
#include "core/scene/util.hpp"
#include "core/scene/geometry/polygon.hpp"
#include "core/renderer/decimaters/util/octree_manager.hpp"
#include "core/renderer/decimaters/util/octree.inl"
#include "core/renderer/pt/pt_common.hpp"
#include <OpenMesh/Tools/Decimater/DecimaterT.hh>
#include <OpenMesh/Tools/Decimater/ModQuadricT.hh>
#include <random>

namespace mufflon::renderer::decimaters {

struct PGridRes {
	int gridRes{ 16 };
	static constexpr ParamDesc get_desc() noexcept {
		return { "Grid res", ParameterTypes::INT };
	}
};
struct PTargetCount {
	int targetCount{ 2000 };
	static constexpr ParamDesc get_desc() noexcept {
		return { "Target count", ParameterTypes::INT };
	}
};
struct PMaxDensity {
	float maxDensity{ 0.05f};
	static constexpr ParamDesc get_desc() noexcept {
		return { "Max density", ParameterTypes::FLOAT };
	}
};
struct AlibiTarget {
	static constexpr const char NAME[] = "Alibi";
	using PixelType = float;
	static constexpr u32 NUM_CHANNELS = 1u;
};

using UniformClustererParameters = ParameterHandler<PGridRes>;
using OctreeClustererParameters = ParameterHandler<PTargetCount, PMaxDensity>;

class CpuUniformVertexClusterer final : public RendererBase<Device::CPU, TargetList<AlibiTarget>> {
public:
	CpuUniformVertexClusterer(mufflon::scene::WorldContainer& world) :
		RendererBase{ world } {}
	~CpuUniformVertexClusterer() = default;

	void pre_reset() {
		if(get_reset_event().geometry_changed())
			m_done = false;

		if(!m_done) {
			logInfo("[CpuUniformVertexClusterer::pre_reset()] Starting clustering...");
			auto& objects = m_currentScene->get_objects();
			for(auto& obj : objects) {
				const auto lodCount = obj.first->get_lod_slot_count();
				for(std::size_t i = 0u; i < lodCount; ++i) {
					if(!obj.first->has_lod(static_cast<u32>(i)))
						continue;

					logInfo("Object: ", obj.first->get_name());
					obj.first->get_lod(static_cast<u32>(i)).template get_geometry<scene::geometry::Polygons>()
						.cluster(static_cast<std::size_t>(m_params.gridRes), true);
					obj.first->get_lod(0).clear_accel_structure();
				}
			}
			m_currentScene->clear_accel_structure();
			m_done = true;
			logInfo("[CpuUniformVertexClusterer::pre_reset()] Finished clustering");
		}
	}
	void iterate() final {}
	IParameterHandler& get_parameters() final { return m_params; }
	static constexpr StringView get_name_static() noexcept { return "Uniform vertex clusterer"; }
	static constexpr StringView get_short_name_static() noexcept { return "UVC"; }
	StringView get_name() const noexcept final { return get_name_static(); }
	StringView get_short_name() const noexcept final { return get_short_name_static(); }

private:
	UniformClustererParameters m_params{};
	bool m_done = false;
};



class CpuOctreeVertexClusterer final : public RendererBase<Device::CPU, TargetList<AlibiTarget>> {
public:
	CpuOctreeVertexClusterer(mufflon::scene::WorldContainer& world) :
		RendererBase{ world } {}
	~CpuOctreeVertexClusterer() = default;

	void pre_reset() {
		if(get_reset_event().geometry_changed())
			m_done = false;

		if(!m_done) {
			const auto& objects = m_currentScene->get_objects();
			const auto& instances = m_currentScene->get_instances();
			m_octrees = std::make_unique<OctreeManager<FloatOctree>>(16'000'000u, static_cast<u32>(objects.size()));

			for(const auto& obj : objects) {
				// TODO: proper bounding box!
				const auto aabb = obj.first->get_lod(0).get_bounding_box();

				// We have to weight the splitting factor with the average instance scaling.
				// Since we weight importance with baseArea / area, it otherwise happens
				// that, if all instances are scaled up or down, importance isn't properly
				// gathered
				ei::Vec3 scaleSum{ 0.f };
				for(std::size_t i = obj.second.offset; i < (obj.second.offset + obj.second.count); ++i)
					scaleSum += scene::Instance::extract_scale(m_world.get_world_to_instance_transformation(instances[i]));
				scaleSum /= static_cast<float>(obj.second.count);
				const auto splitScale = ei::max(scaleSum) * ei::max(scaleSum);

				m_octrees->create(obj.first->get_lod(0).get_bounding_box(), splitScale * 8.f);
			}
			m_currentScene->clear_accel_structure();
			m_done = true;
			logInfo("[CpuUniformVertexClusterer::pre_reset()] Finished clustering");
		}
	}
	void iterate() final {
		const u32 NUM_PIXELS = m_outputBuffer.get_num_pixels();
		std::vector<math::Rng> rngs;
		rngs.reserve(get_max_thread_num());
		for(int i = 0; i < get_max_thread_num(); ++i)
			rngs.emplace_back(std::random_device{}());

#pragma PARALLEL_FOR
		for(int pixel = 0; pixel < (int)NUM_PIXELS; ++pixel) {
			const Pixel coord{ pixel % m_outputBuffer.get_width(), pixel / m_outputBuffer.get_width() };

			Spectrum throughput{ 1.0f };
			float guideWeight = 1.0f;
			PtPathVertex vertex;
			VertexSample sample;
			// Create a start for the path
			auto& rng = rngs[get_current_thread_idx()];
			PtPathVertex::create_camera(&vertex, nullptr, m_sceneDesc.camera.get(), coord, rng.next());
			math::RndSet2_1 rnd{ rng.next(), rng.next() };
			float rndRoulette = math::sample_uniform(u32(rng.next()));
			if(walk(m_sceneDesc, vertex, rnd, rndRoulette, false, throughput, vertex, sample, guideWeight) == WalkResult::HIT) {
				const auto hitId = vertex.get_primitive_id();
				const auto lodIdx = m_sceneDesc.lodIndices[hitId.instanceId];
				const auto& lod = m_sceneDesc.lods[lodIdx];

				const auto cosAngle = -ei::dot(vertex.get_incident_direction(),
											   vertex.get_normal());
				const auto objSpacePos = ei::transform(vertex.get_position(),
													   m_sceneDesc.worldToInstance[hitId.instanceId]);
				const auto objSpaceNormal = ei::transform(vertex.get_normal(),
														  ei::Mat3x3{ m_sceneDesc.worldToInstance[hitId.instanceId] });
				// Determine the face area
				const auto area = compute_area_instance_transformed(m_sceneDesc, lod.polygon, hitId);
				const auto baseArea = compute_area(m_sceneDesc, lod.polygon, hitId);

				const auto impDensity = (1.f - ei::abs(cosAngle)) * baseArea / area;
				if(!isnan(impDensity)) {
					const auto primId = static_cast<u32>(hitId.primId);
					const auto& polygon = lod.polygon;
					auto& octree = (*m_octrees)[lodIdx];
					if(primId < polygon.numTriangles) {
						const auto tri = scene::get_triangle(polygon, primId);
						float distSum = 0.f;
						for(u32 i = 0u; i < 3u; ++i)
							distSum += ei::len(tri.v(i) - objSpacePos);
						for(u32 i = 0u; i < 3u; ++i) {
							const auto dist = ei::len(tri.v(i) - objSpacePos);
							// TODO: normal!
							octree.add_sample(tri.v(i), ei::Vec3{ 0.f, 1.f, 0.f }, impDensity * dist / distSum);
						}
					} else {
						const auto quad = scene::get_quad(polygon, primId);
						float distSum = 0.f;
						for(u32 i = 0u; i < 4u; ++i)
							distSum += ei::len(quad.v(i) - objSpacePos);
						for(u32 i = 0u; i < 4u; ++i) {
							const auto dist = ei::len(quad.v(i) - objSpacePos);
							// TODO: normal!
							octree.add_sample(quad.v(i), ei::Vec3{ 0.f, 1.f, 0.f }, impDensity * dist / distSum);
						}
					}

				}
			}
		}
	}
	void post_iteration(IOutputHandler& outputBuffer) final {
		using namespace std::chrono;
		// Cluster
		auto& objects = m_currentScene->get_objects();
		std::size_t i = 0u;
		for(auto& obj : objects) {
			auto& lod = obj.first->get_lod(0);
			auto& poly = lod.template get_geometry<scene::geometry::Polygons>();
			auto decimater = poly.create_decimater();
			OpenMesh::Decimater::ModQuadricT<scene::geometry::PolygonMeshType>::Handle modQuadricHandle;
			decimater.add(modQuadricHandle);
			const auto t0 = high_resolution_clock::now();
			poly.cluster_decimate((*m_octrees)[i++], decimater, static_cast<std::size_t>(m_params.targetCount), m_params.maxDensity);
			const auto t1 = high_resolution_clock::now();
			logInfo("Duration: ", duration_cast<milliseconds>(t1 - t0).count(), "ms");
			lod.clear_accel_structure();
		}

		m_currentScene->clear_accel_structure();
	}
	IParameterHandler& get_parameters() final { return m_params; }
	static constexpr StringView get_name_static() noexcept { return "Octree vertex clusterer"; }
	static constexpr StringView get_short_name_static() noexcept { return "OVC"; }
	StringView get_name() const noexcept final { return get_name_static(); }
	StringView get_short_name() const noexcept final { return get_short_name_static(); }

private:
	OctreeClustererParameters m_params{};

	std::unique_ptr<OctreeManager<FloatOctree>> m_octrees;

	bool m_done = false;
};



} // namespace mufflon::renderer::decimaters