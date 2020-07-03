#pragma once

#include "core/renderer/renderer_base.hpp"
#include "core/scene/util.hpp"
#include "core/scene/geometry/polygon.hpp"
#include "core/renderer/decimaters/octree/octree_manager.hpp"
#include "core/renderer/decimaters/octree/octree.inl"
#include "core/scene/clustering/octree_clustering.hpp"
#include "core/scene/clustering/uniform_clustering.hpp"
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
			scene::geometry::PolygonMeshType mesh{};
			std::vector<u32> vertexIndexBuffer{};
			mesh.request_vertex_status();
			mesh.request_face_status();
			mesh.request_edge_status();
			auto& objects = m_currentScene->get_objects();
			for(auto& obj : objects) {
				const auto lodCount = obj.first->get_lod_slot_count();
				for(std::size_t i = 0u; i < lodCount; ++i) {
					using namespace std::chrono;
					const auto t0 = high_resolution_clock::now();
					mesh.clean_keep_reservation();
					if(!m_world.load_lod(*obj.first, static_cast<u32>(i)))
						throw std::runtime_error("Failed to load LoD " + std::to_string(i)
												 + " of object '" + std::string(obj.first->get_name()) + "'");
					auto& lod = obj.first->add_reduced_lod(static_cast<u32>(i));
					auto& polygons = lod.template get_geometry<scene::geometry::Polygons>();
					polygons.create_halfedge_structure(mesh);
					scene::clustering::UniformVertexClusterer clusterer{ ei::UVec3{ static_cast<u32>(m_params.gridRes) } };
					const auto clusterCount = clusterer.cluster(mesh, polygons.get_bounding_box());
					polygons.reconstruct_from_reduced_mesh(mesh, &vertexIndexBuffer);
					lod.clear_accel_structure();
					obj.first->remove_original_lod(i);
					logInfo("Object '", obj.first->get_name(), "': ",
							duration_cast<milliseconds>(high_resolution_clock::now() - t0).count(), "ms");
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

class GpuUniformVertexClusterer final : public RendererBase<Device::CPU, TargetList<AlibiTarget>> {
public:
	GpuUniformVertexClusterer(mufflon::scene::WorldContainer& world) :
		RendererBase{ world } {}
	~GpuUniformVertexClusterer() = default;

	bool uses_device(Device device) const noexcept override { return may_use_device(device); }
	static constexpr bool may_use_device(Device device) noexcept { return device == Device::CUDA; }

	void pre_reset() {
		if(get_reset_event().geometry_changed())
			m_done = false;

		if(!m_done) {
			logInfo("[GpuUniformVertexClusterer::pre_reset()] Starting clustering...");
			auto& objects = m_currentScene->get_objects();
			for(auto& obj : objects) {
				const auto lodCount = obj.first->get_lod_slot_count();
				for(std::size_t i = 0u; i < lodCount; ++i) {
					using namespace std::chrono;
					const auto t0 = high_resolution_clock::now();
					if(!m_world.load_lod(*obj.first, static_cast<u32>(i)))
						throw std::runtime_error("Failed to load LoD " + std::to_string(i)
												 + " of object '" + std::string(obj.first->get_name()) + "'");
					auto& lod = obj.first->add_reduced_lod(static_cast<u32>(i));
					auto& polygons = lod.template get_geometry<scene::geometry::Polygons>();
					polygons.cluster_uniformly(ei::UVec3{ m_params.gridRes });
					lod.clear_accel_structure();
					obj.first->remove_original_lod(i);
					logInfo("Object '", obj.first->get_name(), "': ",
							duration_cast<milliseconds>(high_resolution_clock::now() - t0).count(), "ms");
				}
			}
			m_currentScene->clear_accel_structure();
			m_done = true;
			logInfo("[GpuUniformVertexClusterer::pre_reset()] Finished clustering");
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
			m_octrees = std::make_unique<OctreeManager<FloatOctree>>(16'000'000u, static_cast<u32>(objects.size()), false);

			for(const auto& obj : objects) {
				// TODO: proper bounding box!
				if(!m_world.load_lod(*obj.first, 0u))
					throw std::runtime_error("Failed to load LoD 0 of object '"
											 + std::string(obj.first->get_name()) + "'");
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
			if(walk(m_sceneDesc, vertex, rnd, rndRoulette, false, throughput, vertex, sample, nullptr, guideWeight) == WalkResult::HIT) {
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
		if(!m_done) {
			logInfo("[CpuOctreeVertexClusterer::post_iteration()] Starting clustering...");
			scene::geometry::PolygonMeshType mesh{};
			std::vector<u32> vertexIndexBuffer{};
			mesh.request_vertex_status();
			mesh.request_face_status();
			mesh.request_edge_status();
			auto& objects = m_currentScene->get_objects();
			for(auto& obj : objects) {
				const auto lodCount = obj.first->get_lod_slot_count();
				for(std::size_t i = 0u; i < lodCount; ++i) {
					using namespace std::chrono;
					const auto t0 = high_resolution_clock::now();
					mesh.clean_keep_reservation();
					auto& lod = obj.first->add_reduced_lod(static_cast<u32>(i));
					auto& polygons = lod.template get_geometry<scene::geometry::Polygons>();
					polygons.create_halfedge_structure(mesh);
					scene::clustering::OctreeVertexClusterer clusterer{ (*m_octrees)[i++], static_cast<std::size_t>(m_params.targetCount), m_params.maxDensity };
					const auto clusterCount = clusterer.cluster(mesh, polygons.get_bounding_box());
					polygons.reconstruct_from_reduced_mesh(mesh, &vertexIndexBuffer);
					lod.clear_accel_structure();
					obj.first->remove_original_lod(i);
					logInfo("Object '", obj.first->get_name(), "': ",
							duration_cast<milliseconds>(high_resolution_clock::now() - t0).count(), "ms");
				}
			}
			m_currentScene->clear_accel_structure();
			m_done = true;
			logInfo("[CpuOctreeVertexClusterer::post_iteration()] Finished clustering");
		}
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