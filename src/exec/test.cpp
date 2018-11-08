#include <vector>
#include <unordered_map>
#include <string>
#include <string_view>
#include <iostream>
#include <memory>
#include <algorithm>
#include <random>
#include <chrono>

#include <cuda_runtime.h>
#include "ei/conversions.hpp"
#include "core/scene/geometry/polygon.hpp"
#include "core/scene/geometry/sphere.hpp"
#include "core/scene/object.hpp"
#include "core/scene/world_container.hpp"
#include "core/scene/lights/lights.hpp"
#include "core/scene/lights/light_tree.hpp"
#include "core/scene/materials/material.hpp"
#include "core/scene/lights/light_tree.hpp"
#include "core/cameras/camera.hpp"
#include "core/memory/allocator.hpp"

using mufflon::Device;
using namespace mufflon::scene;
using namespace mufflon::scene::geometry;
using namespace mufflon::cameras;

class VectorStream : public std::basic_streambuf<char, std::char_traits<char>> {
public:
	template < class U >
	VectorStream(std::vector<U>& vec) {
		this->setg(reinterpret_cast<char*>(vec.data()), reinterpret_cast<char*>(vec.data()),
				   reinterpret_cast<char*>(vec.data() + vec.size()));
	}
};

void test_lighttree() {
	using namespace lights;

	std::mt19937_64 rng(std::random_device{}());
	std::uniform_int_distribution<std::uint64_t> intDist;
	std::uniform_real_distribution<float> floatDist;
	std::vector<Photon> photons(1000000u);

	auto testLights = [&rng, &photons, intDist, floatDist](std::vector<PositionalLights> posLights,
									   std::vector<DirectionalLight> dirLights,
									   const ei::Box& bounds,
									   std::optional<textures::TextureHandle> envLight = std::nullopt)
	{
		LightTree tree;
		tree.build(std::move(posLights), std::move(dirLights), bounds, envLight);

		using namespace std::chrono;
		auto& lightTree = tree.aquire_tree<Device::CPU>();
		auto t0 = high_resolution_clock::now();
		for(std::size_t i = 0u; i < photons.size(); ++i) {
			photons[i] = emit(lightTree, i, photons.size(), intDist(rng),
							  bounds, { floatDist(rng), floatDist(rng),
							  floatDist(rng), floatDist(rng) });
		}
		auto t1 = high_resolution_clock::now();

		ei::Vec3 positions{};
		ei::Vec3 directions{};
		for(const auto& p : photons) {
			positions += p.pos.position;
			directions += p.dir.direction;
		}

		positions /= static_cast<float>(photons.size());
		directions /= static_cast<float>(photons.size());

		std::cout << "[" << positions[0]
			<< '|' << positions[1] << '|' << positions[2] << "] ["
			<< directions[0] << '|' << directions[1] << '|'
			<< directions[2] << "] (" << photons.size() << " photons, "
			<< duration_cast<milliseconds>(t1 - t0).count()
			<< "ms)" << std::endl;
	};
	
	float posScale = 10.f;
	float intensityScale = 20.f;
	ei::Box bounds{ posScale * ei::Vec3{-1, -1, -1}, posScale * ei::Vec3{1, 1, 1} };
	std::vector<PositionalLights> posLights(10000);
	std::vector<DirectionalLight> dirLights(1);

	{ // Test point lights
		for(auto& light : posLights) {
			light = PointLight{
				posScale * ei::Vec3{ floatDist(rng), floatDist(rng), floatDist(rng) },
				intensityScale * ei::Vec3{ floatDist(rng), floatDist(rng), floatDist(rng) },
			};
		}
		std::cout << "Point lights: ";
		testLights(posLights, {}, bounds);
	}
	{ // Test spot lights
		for(auto& light : posLights) {
			float angle = floatDist(rng);
			light = SpotLight{
				posScale * ei::Vec3{ floatDist(rng), floatDist(rng), floatDist(rng) },
				ei::packOctahedral32(ei::normalize(ei::Vec3{ floatDist(rng), floatDist(rng), floatDist(rng) })),
				intensityScale * ei::Vec3{ floatDist(rng), floatDist(rng), floatDist(rng) },
				__float2half(floatDist(rng)), __float2half(std::min(angle, floatDist(rng)))
			};
		}
		std::cout << "Spot lights: ";
		testLights(posLights, {}, bounds);
	}
	{ // Test triangular area lights
		for(auto& light : posLights) {
			float angle = floatDist(rng);
			light = AreaLightTriangle{
				{ posScale * ei::Vec3{ floatDist(rng), floatDist(rng), floatDist(rng) },
				  posScale * ei::Vec3{ floatDist(rng), floatDist(rng), floatDist(rng) },
				  posScale * ei::Vec3{ floatDist(rng), floatDist(rng), floatDist(rng) } },
				intensityScale * ei::Vec3{ floatDist(rng), floatDist(rng), floatDist(rng) }
			};
		}
		std::cout << "Triangular area lights: ";
		testLights(posLights, {}, bounds);
	}
	{ // Test quad area lights
		for(auto& light : posLights) {
			float angle = floatDist(rng);
			light = AreaLightQuad{
				{ posScale * ei::Vec3{ floatDist(rng), floatDist(rng), floatDist(rng) },
				  posScale * ei::Vec3{ floatDist(rng), floatDist(rng), floatDist(rng) },
				  posScale * ei::Vec3{ floatDist(rng), floatDist(rng), floatDist(rng) },
				  posScale * ei::Vec3{ floatDist(rng), floatDist(rng), floatDist(rng) } },
				intensityScale * ei::Vec3{ floatDist(rng), floatDist(rng), floatDist(rng) }
			};
		}
		std::cout << "Quad area lights: ";
		testLights(posLights, {}, bounds);
	}
	{ // Test spherical area lights
		for(auto& light : posLights) {
			float angle = floatDist(rng);
			light = AreaLightSphere{
				posScale * ei::Vec3{ floatDist(rng), floatDist(rng), floatDist(rng) },
				floatDist(rng),
				intensityScale * ei::Vec3{ floatDist(rng), floatDist(rng), floatDist(rng) }
			};
		}
		std::cout << "Spherical lights: ";
		testLights(posLights, {}, bounds);
	}
	{ // Test directional lights
		for(auto& light : dirLights) {
			float angle = floatDist(rng);
			light = DirectionalLight{
				ei::normalize(ei::Vec3{ floatDist(rng), floatDist(rng), floatDist(rng) }),
				intensityScale * ei::Vec3{ floatDist(rng), floatDist(rng), floatDist(rng) }
			};
		}
		std::cout << "Directional lights: ";
		testLights({}, dirLights, bounds);
	}
	// TODO: Test envmap light
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
}

void test_polygon() {
	std::cout << "Testing polygons" << std::endl;

	Polygons poly;

	{
		auto v0 = poly.add(Point(0, 0, 0), Normal(1, 0, 0), UvCoordinate(0, 0));
		auto v1 = poly.add(Point(1, 0, 0), Normal(1, 0, 0), UvCoordinate(1, 0));
		auto v2 = poly.add(Point(0, 1, 0), Normal(1, 0, 0), UvCoordinate(0, 1));
		poly.add(v0, v1, v2);
		(*poly.get_mat_indices().aquire<>())[0u] = 2u;

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

		auto bv0 = poly.add_bulk(points.size(), pointStream, normalStream, uvStream);
		auto f1 = poly.add(OpenMesh::VertexHandle(static_cast<Polygons::Index>(bv0.handle.idx())),
							OpenMesh::VertexHandle(static_cast<Polygons::Index>(bv0.handle.idx())+1),
							OpenMesh::VertexHandle(static_cast<Polygons::Index>(bv0.handle.idx())+2),
							OpenMesh::VertexHandle(static_cast<Polygons::Index>(bv0.handle.idx())+3));
		poly.add_bulk<>(poly.get_mat_indices(), f1, mats.size(), matStream);
	}

	{
		auto points = *poly.get_points().aquireConst();
		auto normals = *poly.get_normals().aquireConst();
		auto uvs = *poly.get_uvs().aquireConst();
		auto matIndices = *poly.get_mat_indices().aquireConst();
		std::cout << "Points:" << std::endl;
		for(std::size_t i = 0u; i < poly.get_points().get_size(); ++i)
			std::cout << "  [" << points[i][0] << '|' << points[i][1] << '|' << points[i][2] << ']' << std::endl;
		std::cout << "Normals:" << std::endl;
		for(std::size_t i = 0u; i < poly.get_normals().get_size(); ++i)
			std::cout << "  [" << normals[i][0] << '|' << normals[i][1] << '|' << normals[i][2] << ']' << std::endl;
		std::cout << "Normals:" << std::endl;
		for(std::size_t i = 0u; i < poly.get_uvs().get_size(); ++i)
			std::cout << "  [" << uvs[i][0] << '|' << uvs[i][1] << ']' << std::endl;
		std::cout << "Material indices:" << std::endl;
		for(std::size_t i = 0u; i < poly.get_mat_indices().get_size(); ++i)
			std::cout << "  " << matIndices[i] << std::endl;
	}

	poly.synchronize<Device::CUDA>();
	poly.unload<Device::CUDA>();
}

void test_sphere() {
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
	Object* obj = container.create_object();
	
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
	auto scenarioHdl = container.create_scenario();
	auto* sceneHdl = container.load_scene(scenarioHdl);

	const ei::Box& aabb = sceneHdl->get_bounding_box();
	std::cout << "Bounding box: [" << aabb.min[0] << '|' << aabb.min[1]
		<< '|' << aabb.min[2] << "] - [" << aabb.max[0] << '|'
		<< aabb.max[1] << '|' << aabb.max[2] << ']' << std::endl;
}


int main() {
	test_allocator();
	test_polygon();
	test_sphere();
	test_custom_attributes();
	test_object();
	test_scene_creation();
	test_lighttree();

	std::cout << "All tests successful" << std::endl;
	std::cin.get();
	return 0;
}