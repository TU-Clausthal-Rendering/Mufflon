#include <vector>
#include <unordered_map>
#include <string>
#include <string_view>
#include <iostream>
#include <memory>
#include <algorithm>
#include <random>
#include <chrono>
#include <execution>

#include "core/memory/allocator.hpp"
#include "export/interface.h"

#pragma warning(disable:4251)

/*
void test_renderer() {
	//mufflon::renderer::GpuPathTracer gpu;
	//gpu.run();
}

void test_lighttree() {
	using namespace lights;

	std::mt19937_64 rng(std::random_device{}());
	std::uniform_int_distribution<std::uint64_t> intDist;
	std::uniform_real_distribution<float> floatDist;
	std::vector<Photon> photons(1000000u);
	std::vector<NextEventEstimation> nees(1000000u);
	std::size_t lightCount = 10000u;

	auto testLights = [&rng, &photons, &nees, intDist, floatDist](std::vector<PositionalLights> posLights,
									   std::vector<DirectionalLight> dirLights,
									   const ei::Box& bounds,
									   std::optional<textures::TextureHandle> envLight = std::nullopt)
	{
		LightTreeBuilder tree;
		tree.build(std::move(posLights), std::move(dirLights), bounds, envLight);

		using namespace std::chrono;
		const auto& lightTree = tree.aquire_tree<Device::CPU>();

		auto t0 = high_resolution_clock::now();
#pragma omp parallel for
		for(long long i = 0u; i < static_cast<long long>(photons.size()); ++i) {
			photons[i] = emit(lightTree, i, photons.size(), intDist(rng),
							  bounds, { floatDist(rng), floatDist(rng),
							  floatDist(rng), floatDist(rng) });
		}
		auto t1 = high_resolution_clock::now();

		auto t2 = high_resolution_clock::now();
#pragma omp parallel for
		for(long long i = 0u; i < static_cast<long long>(nees.size()); ++i) {
			nees[i] = connect(lightTree, i, photons.size(), intDist(rng),
							  ei::Vec3{floatDist(rng), floatDist(rng), floatDist(rng)},
							  bounds, { floatDist(rng), floatDist(rng),
										floatDist(rng), floatDist(rng) },
							  [](const ei::Vec3& pos, const ei::Vec3& leftPos, const ei::Vec3& rightPos,
								 const float leftFlux, const float rightFlux) {
									return leftFlux / (leftFlux + rightFlux);
								});
		}
		auto t3 = high_resolution_clock::now();

		Photon posPhotons = std::reduce(std::execution::par_unseq, photons.cbegin(),
										photons.cend(), Photon{ {}, {}, {} },
										 [](Photon init, const Photon& photon) {
			init.pos.position += photon.pos.position;
			return init;
		});
		NextEventEstimation posNees = std::reduce(std::execution::par_unseq, nees.cbegin(),
												  nees.cend(), NextEventEstimation{ {}, {}, {} },
											   [](NextEventEstimation init,
												  const NextEventEstimation& photon) {
			init.pos.position += photon.pos.position;
			return init;
		});

		std::cout << "[" << (posPhotons.pos.position[0] + posNees.pos.position[0]) << "] (" << photons.size() << " photons, "
			<< duration_cast<milliseconds>(t1 - t0).count() << "|"
			<< duration_cast<milliseconds>(t3 - t2).count() << "ms)" << std::endl;
	};
	
	float posScale = 10.f;
	float intensityScale = 20.f;
	ei::Box bounds{ posScale * ei::Vec3{-1, -1, -1}, posScale * ei::Vec3{1, 1, 1} };
	std::vector<PositionalLights> pointLights(lightCount);
	std::vector<PositionalLights> spotLights(lightCount);
	std::vector<PositionalLights> areaTriLights(lightCount);
	std::vector<PositionalLights> areaQuadLights(lightCount);
	std::vector<PositionalLights> areaSphereLights(lightCount);
	std::vector<DirectionalLight> dirLights(lightCount);

	{ // Test point lights
		for(auto& light : pointLights) {
			light = PointLight{
				posScale * ei::Vec3{ floatDist(rng), floatDist(rng), floatDist(rng) },
				intensityScale * ei::Vec3{ floatDist(rng), floatDist(rng), floatDist(rng) },
			};
		}
		std::cout << "Point lights: ";
		testLights(pointLights, {}, bounds);
	}
	{ // Test spot lights
		for(auto& light : spotLights) {
			float angle = floatDist(rng);
			light = SpotLight{
				posScale * ei::Vec3{ floatDist(rng), floatDist(rng), floatDist(rng) },
				ei::packOctahedral32(ei::normalize(ei::Vec3{ floatDist(rng), floatDist(rng), floatDist(rng) })),
				intensityScale * ei::Vec3{ floatDist(rng), floatDist(rng), floatDist(rng) },
				__float2half(floatDist(rng)), __float2half(std::min(angle, floatDist(rng)))
			};
		}
		std::cout << "Spot lights: ";
		testLights(spotLights, {}, bounds);
	}
	{ // Test triangular area lights
		for(auto& light : areaTriLights) {
			light = AreaLightTriangle{
				{ posScale * ei::Vec3{ floatDist(rng), floatDist(rng), floatDist(rng) },
				  posScale * ei::Vec3{ floatDist(rng), floatDist(rng), floatDist(rng) },
				  posScale * ei::Vec3{ floatDist(rng), floatDist(rng), floatDist(rng) } },
				intensityScale * ei::Vec3{ floatDist(rng), floatDist(rng), floatDist(rng) }
			};
		}
		std::cout << "Triangular area lights: ";
		testLights(areaTriLights, {}, bounds);
	}
	{ // Test quad area lights
		for(auto& light : areaQuadLights) {
			light = AreaLightQuad{
				{ posScale * ei::Vec3{ floatDist(rng), floatDist(rng), floatDist(rng) },
				  posScale * ei::Vec3{ floatDist(rng), floatDist(rng), floatDist(rng) },
				  posScale * ei::Vec3{ floatDist(rng), floatDist(rng), floatDist(rng) },
				  posScale * ei::Vec3{ floatDist(rng), floatDist(rng), floatDist(rng) } },
				intensityScale * ei::Vec3{ floatDist(rng), floatDist(rng), floatDist(rng) }
			};
		}
		std::cout << "Quad area lights: ";
		testLights(areaQuadLights, {}, bounds);
	}
	{ // Test spherical area lights
		for(auto& light : areaSphereLights) {
			light = AreaLightSphere{
				posScale * ei::Vec3{ floatDist(rng), floatDist(rng), floatDist(rng) },
				floatDist(rng),
				intensityScale * ei::Vec3{ floatDist(rng), floatDist(rng), floatDist(rng) }
			};
		}
		std::cout << "Spherical lights: ";
		testLights(areaSphereLights, {}, bounds);
	}
	{ // Test directional lights
		for(auto& light : dirLights) {
			light = DirectionalLight{
				ei::normalize(ei::Vec3{ floatDist(rng), floatDist(rng), floatDist(rng) }),
				intensityScale * ei::Vec3{ floatDist(rng), floatDist(rng), floatDist(rng) }
			};
		}
		std::cout << "Directional lights: ";
		testLights({}, dirLights, bounds);
	}
	// TODO: Test envmap light

	// All together
	std::vector<PositionalLights> posLights(pointLights.size() + spotLights.size()
											+ areaTriLights.size() + areaQuadLights.size()
											+ areaSphereLights.size());
	auto insert = posLights.insert(posLights.begin(), pointLights.cbegin(), pointLights.cend()) + pointLights.size();
	insert = posLights.insert(insert, spotLights.cbegin(), spotLights.cend()) + pointLights.size();
	insert = posLights.insert(insert, areaTriLights.cbegin(), areaTriLights.cend()) + spotLights.size();
	insert = posLights.insert(insert, areaQuadLights.cbegin(), areaQuadLights.cend()) + areaTriLights.size();
	posLights.insert(insert, areaSphereLights.cbegin(), areaSphereLights.cend());
	std::cout << "All combined: ";
	testLights(posLights, dirLights, bounds);
}

void test_allocator() {
	std::cout << "Testing custom device allocator." << std::endl;

	mufflon::unique_device_ptr<Device::CPU, CameraParams> p0 =
		mufflon::make_udevptr<Device::CPU, CameraParams>(CameraModel::PINHOLE);
	mAssert(p0 != nullptr);
	mAssert(p0->type == CameraModel::PINHOLE);

	mufflon::unique_device_ptr<Device::CPU, CameraParams[]> p1 =
		mufflon::make_udevptr_array<Device::CPU, CameraParams[]>(2, CameraModel::FOCUS);
	mAssert(p1 != nullptr);
	mAssert(p1[0].type == CameraModel::FOCUS);
	mAssert(p1[1].type == CameraModel::FOCUS);

	mufflon::unique_device_ptr<Device::CUDA, CameraParams> p2 =
		mufflon::make_udevptr<Device::CUDA, CameraParams>(CameraModel::PINHOLE);
	mAssert(p2 != nullptr);

	mufflon::unique_device_ptr<Device::CUDA, CameraParams> p3 =
		mufflon::make_udevptr_array<Device::CUDA, CameraParams>(2, CameraModel::FOCUS);
	mAssert(p3 != nullptr);
}

void test_custom_attributes() {
	std::cout << "Testing custom attributes (on polygon)" << std::endl;


	Polygons poly;
	
	{
		// Empty request->remove
		auto hdl0 = poly.request<Polygons::VertexAttributeHandle<float>>("test0");
		auto hdl1 = poly.request<Polygons::VertexAttributeHandle<short>>("test1");
		auto hdl2 = poly.request<Polygons::FaceAttributeHandle<double>>("test2");
		auto& attr1 = poly.aquire(hdl1);
		mAssert(attr1.get_size() == 0);
		mAssert(attr1.get_elem_size() == sizeof(short));
		poly.remove(hdl0);

		auto v0 = poly.add(Point(0, 0, 0), Normal(1, 0, 0), UvCoordinate(0, 0));
		auto v1 = poly.add(Point(1, 0, 0), Normal(1, 0, 0), UvCoordinate(1, 0));
		auto v2 = poly.add(Point(0, 1, 0), Normal(1, 0, 0), UvCoordinate(0, 1));
		poly.add(v0, v1, v2);
		(*poly.get_mat_indices().aquire<>())[0u] = 3u;

		auto& attr2 = poly.aquire(hdl2);
		(*attr1.aquire())[1] = 5;
		(*attr2.aquire())[0] = 0.4;
		mAssert(attr1.get_size() == 3);
		mAssert(attr2.get_size() == 1);
		mAssert((*attr1.aquireConst())[1] == 5);
		mAssert((*attr2.aquireConst())[0] == 0.4);

		poly.remove(hdl0);
		poly.remove(hdl1);
		poly.remove(hdl2);
	}
}*/

void test_polygon() {
	std::cout << "Testing polygons" << std::endl;

	ObjectHdl obj = world_create_object();
	{
		auto v0 = polygon_add_vertex(obj, { 0.f, 0.f, 0.f }, { 1.f, 0.f, 0.f }, { 0.f, 0.f });
		auto v1 = polygon_add_vertex(obj, { 1.f, 0.f, 0.f }, { 1.f, 0.f, 0.f }, { 1.f, 0.f });
		auto v2 = polygon_add_vertex(obj, { 0.f, 1.f, 0.f }, { 1.f, 0.f, 0.f }, { 0.f, 1.f });
		mAssert(v0 >= 0 && v1 >= 0 && v2 >= 0);
		FaceHdl f0 = polygon_add_triangle_material(obj, {
			static_cast<uint32_t>(v0),
			static_cast<uint32_t>(v1),
			static_cast<uint32_t>(v2)
		}, 2u);
		mAssert(f0 >= 0);
		bool success = polygon_set_material_idx(obj, f0, 3u);
		mAssert(success);

		// TODO: test bulk functions as well
		FILE* pointStream = nullptr;
		FILE* normalStream = nullptr;
		FILE* uvStream = nullptr;
		FILE* matStream = nullptr;
		std::size_t vertexCount = 4u;
		std::size_t faceCount = 1u;
		std::size_t pointsRead, normalsRead, uvsRead;

		VertexHdl bv0 = polygon_add_vertex_bulk(obj, vertexCount, pointStream, normalStream, uvStream,
								&pointsRead, &normalsRead, &uvsRead);
		mAssert(bv0 >= 0);
		FaceHdl f1 = polygon_add_quad(obj, {
			static_cast<uint32_t>(bv0),
			static_cast<uint32_t>(bv0 + 1),
			static_cast<uint32_t>(bv0 + 2),
			static_cast<uint32_t>(bv0 + 3),
		});
		std::size_t count = polygon_set_material_idx_bulk(obj, f1, faceCount,
														  matStream);
		mAssert(count != INVALID_SIZE);
	}
}

/*void test_sphere() {
	std::cout << "Testing spheres" << std::endl;

	Spheres spheres;

	{
		auto impHandle = spheres.request<int>("importance");
		auto s0 = spheres.add(Point(0, 0, 0), 55.f);
		(*spheres.get_mat_indices().aquire<>())[s0] = 13u;

		std::vector<Spheres::Sphere> radPos{ {Point(0,0,1), 5.f}, {Point(1,0,1), 1.f}, {Point(0,1,1), 7.f}, {Point(1,1,1), 3.f} };
		std::vector<MaterialIndex> mats{ 1u, 3u, 27u, 15u };
		VectorStream radPosBuffer(radPos);
		VectorStream matBuffer(mats);
		std::istream radPosStream(&radPosBuffer);
		std::istream matStream(&matBuffer);

		auto bs0 = spheres.add_bulk(radPos.size(), radPosStream);
		spheres.add_bulk(spheres.get_mat_indices(), bs0.handle, mats.size(), matStream);
		spheres.remove(impHandle);
	}

	{
		auto radPos = *spheres.get_spheres().aquireConst();
		auto matIndices = *spheres.get_mat_indices().aquireConst();
		std::cout << "Spheres:" << std::endl;
		for(std::size_t i = 0u; i < spheres.get_spheres().get_size(); ++i)
			std::cout << "  [" << radPos[i].m_radPos.position[0] << '|'
			<< radPos[i].m_radPos.position[1] << '|' << radPos[i].m_radPos.position[2]
			<< "] - " << radPos[i].m_radPos.radius << std::endl;
		std::cout << "Material indices:" << std::endl;
		for(std::size_t i = 0u; i < spheres.get_mat_indices().get_size(); ++i)
			std::cout << "  " << matIndices[i] << std::endl;
	}

	spheres.synchronize<Device::CUDA>();
	spheres.unload<Device::CPU>();
	spheres.synchronize<Device::CPU>();
	spheres.unload<Device::CUDA>();
}

void test_object() {
	
	Object obj;

	// Polygon interface
	{
		std::vector<Point> points{ Point(0, 0, 1), Point(1, 0, 1), Point(0, 1, 1), Point(1, 1, 1) };
		std::vector<Point> normals{ Normal(-1, -1, 0), Normal(1, -1, 0), Normal(-1, 1, 1), Normal(1, 1, 1) };
		std::vector<UvCoordinate> uvs{ UvCoordinate(0, 0), UvCoordinate(1, 0), UvCoordinate(0, 1), UvCoordinate(1, 1) };
		std::vector<MaterialIndex> mats{ 5u };
		VectorStream pointBuffer(points);
		VectorStream normalBuffer(normals);
		VectorStream uvBuffer(uvs);
		VectorStream matBuffer(mats);
		std::istream pointStream(&pointBuffer);
		std::istream normalStream(&normalBuffer);
		std::istream uvStream(&uvBuffer);
		std::istream matStream(&matBuffer);

		auto bv0 = obj.add_bulk<Polygons>(points.size(), pointStream, normalStream, uvStream);
		auto bf0 = obj.add<Polygons>(OpenMesh::VertexHandle(static_cast<Polygons::Index>(bv0.handle.idx())),
							OpenMesh::VertexHandle(static_cast<Polygons::Index>(bv0.handle.idx()) + 1),
							OpenMesh::VertexHandle(static_cast<Polygons::Index>(bv0.handle.idx()) + 2),
							OpenMesh::VertexHandle(static_cast<Polygons::Index>(bv0.handle.idx()) + 3));
		auto matIndexAttr = obj.get_mat_indices<Polygons>();
		obj.add_bulk<Polygons>(matIndexAttr, bf0, mats.size(), matStream);
	}

	// Sphere interface
	{
		std::vector<Spheres::Sphere> radPos{ {Point(0,0,1), 5.f}, {Point(1,0,1), 1.f}, {Point(0,1,1), 7.f}, {Point(1,1,1), 3.f} };
		std::vector<MaterialIndex> mats{ 1u, 3u, 27u, 15u };
		VectorStream radPosBuffer(radPos);
		VectorStream matBuffer(mats);
		std::istream radPosStream(&radPosBuffer);
		std::istream matStream(&matBuffer);

		auto bs0 = obj.add_bulk<Spheres>(radPos.size(), radPosStream);
		obj.add_bulk<Spheres>(obj.get_mat_indices<Spheres>(), bs0.handle, mats.size(), matStream);
	}

	// AABB
	const ei::Box& aabb = obj.get_bounding_box();
	std::cout << "Bounding box: [" << aabb.min[0] << '|' << aabb.min[1]
		<< '|' << aabb.min[2] << "] - [" << aabb.max[0] << '|'
		<< aabb.max[1] << '|' << aabb.max[2] << ']' << std::endl;

	// 
	
	// Syncing
	obj.synchronize<Device::CUDA>();
	obj.unload<Device::CUDA>();
}

void test_scene_creation() {
	WorldContainer container;
	Object *obj = container.create_object();
	
	{
		std::vector<Point> points{ Point(0, 0, 1), Point(1, 0, 1), Point(0, 1, 1), Point(1, 1, 1) };
		std::vector<Point> normals{ Normal(-1, -1, 0), Normal(1, -1, 0), Normal(-1, 1, 1), Normal(1, 1, 1) };
		std::vector<UvCoordinate> uvs{ UvCoordinate(0, 0), UvCoordinate(1, 0), UvCoordinate(0, 1), UvCoordinate(1, 1) };
		std::vector<MaterialIndex> mats{ 5u };
		VectorStream pointBuffer(points);
		VectorStream normalBuffer(normals);
		VectorStream uvBuffer(uvs);
		VectorStream matBuffer(mats);
		std::istream pointStream(&pointBuffer);
		std::istream normalStream(&normalBuffer);
		std::istream uvStream(&uvBuffer);
		std::istream matStream(&matBuffer);

		auto bv0 = obj->add_bulk<Polygons>(points.size(), pointStream, normalStream, uvStream);
		auto bf0 = obj->add<Polygons>(OpenMesh::VertexHandle(static_cast<Polygons::Index>(bv0.handle.idx())),
									 OpenMesh::VertexHandle(static_cast<Polygons::Index>(bv0.handle.idx()) + 1),
									 OpenMesh::VertexHandle(static_cast<Polygons::Index>(bv0.handle.idx()) + 2),
									 OpenMesh::VertexHandle(static_cast<Polygons::Index>(bv0.handle.idx()) + 3));
		auto matIndexAttr = obj->get_mat_indices<Polygons>();
		obj->add_bulk<Polygons>(matIndexAttr, bf0, mats.size(), matStream);
	}

	// Sphere interface
	{
		std::vector<Spheres::Sphere> radPos{ {Point(0,0,1), 5.f}, {Point(1,0,1), 1.f}, {Point(0,1,1), 7.f}, {Point(1,1,1), 3.f} };
		std::vector<MaterialIndex> mats{ 1u, 3u, 27u, 15u };
		VectorStream radPosBuffer(radPos);
		VectorStream matBuffer(mats);
		std::istream radPosStream(&radPosBuffer);
		std::istream matStream(&matBuffer);

		auto bs0 = obj->add_bulk<Spheres>(radPos.size(), radPosStream);
		obj->add_bulk<Spheres>(obj->get_mat_indices<Spheres>(), bs0.handle, mats.size(), matStream);
	}

	Instance* inst = container.create_instance(obj);
	inst->set_transformation_matrix(ei::Matrix<float, 4, 3>{
		1.f,2.f,3.f,
		4.f,5.f,6.f,
		7.f,8.f,9.f,
		10.f,11.f,12.f
	});
	Scenario scenario{ "testScenario", {800, 600}, nullptr };
	auto scenarioHdl = container.add_scenario(std::move(scenario));
	auto* sceneHdl = container.load_scene(scenarioHdl);

	const ei::Box& aabb = sceneHdl->get_bounding_box();
	std::cout << "Bounding box: [" << aabb.min[0] << '|' << aabb.min[1]
		<< '|' << aabb.min[2] << "] - [" << aabb.max[0] << '|'
		<< aabb.max[1] << '|' << aabb.max[2] << ']' << std::endl;
}
*/

int main() {
	test_polygon();
	/*test_allocator();
	test_sphere();
	test_custom_attributes();
	test_object();
	test_scene_creation();
	test_lighttree();
	test_renderer();*/

	std::cout << "All tests successful" << std::endl;
	std::cin.get();
	return 0;
}