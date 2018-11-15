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

#include "export/interface.h"
#include "ei/3dtypes.hpp"
#include "ei/vector.hpp"
#include "core/memory/allocator.hpp"
#include "util/assert.hpp"

#define TEST_STREAM false

namespace {

constexpr const char* ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";

bool is_between(char x, char a, char b) {
	return (x >= a) && (x <= b);
}

void increment_string(std::string& str) {
	if(str.length() == 0u) {
		str = "A";
	} else if(str[str.length() - 1u] < 'A') {
		str[str.length() - 1u] = 'A';
	} else if(str[str.length() - 1u] >= 'z') {
		str[str.length() - 1u] = 'A';
		std::size_t index = str.length() - 1u;
		for(; index > 0u; --index) {
			if(str[index - 1u] < 'z') {
				if(str[index - 1u] < 'Z')
					++str[index - 1u];
				else if(str[index - 1u] < 'a')
					str[index - 1u] = 'a';
				else
					++str[index - 1u];
				return;
			} else {
				str[index - 1u] = 'A';
			}
		}
		// Went to the very beginning -> append new letter
		str += 'A';
	} else if(is_between(str[str.length() - 1u], 'Z'+1, 'a'-1)) {
		str[str.length() - 1u] = 'a';
	} else {
		++str[str.length() - 1u];
	}
}

}

/*
void test_renderer() {
	//mufflon::renderer::GpuPathTracer gpu;
	//gpu.run();
}*/

void test_lights() {
	std::cout << "Testing lights" << std::endl;
	auto errorCheck = [](bool cond) {
		if(!cond)
			throw std::runtime_error("Failed condition");
	};

	std::mt19937_64 rng(std::random_device{}());
	std::uniform_int_distribution<std::uint64_t> intDist;
	std::uniform_real_distribution<float> floatDist;
	std::size_t lightCount = 10000u;
	float posScale = 10.f;
	float intenScale = 20.f;

	std::string pointName = "PointLightA";
	std::string spotName = "SpotLightA";
	std::string dirName = "DirLightA";

	ScenarioHdl scenario = world_create_scenario("TestScenario");

	for(std::size_t i = 0u; i < lightCount; ++i) {
		LightHdl point = world_add_point_light(pointName.c_str(),
											   { 0.f, 0.f, 0.f },
											   { 0.f, 0.f, 0.f });
		errorCheck(point != nullptr);
		LightHdl spot = world_add_spot_light(spotName.c_str(),
											  { 0.f, 0.f, 0.f },
											  { 1.f, 0.f, 0.f },
											  { 0.f, 0.f, 0.f },
											  0.f, 0.f);
		errorCheck(spot != nullptr);
		LightHdl dir = world_add_directional_light(dirName.c_str(),
											   { 1.f, 0.f, 0.f },
											   { 0.f, 0.f, 0.f });
		errorCheck(dir != nullptr);
		float openingAngle = 1.5f * floatDist(rng);
		errorCheck(world_set_point_light_position(point, { posScale*floatDist(rng),
									   posScale*floatDist(rng),
									   posScale*floatDist(rng) }));
		errorCheck(world_set_point_light_intensity(point, { intenScale*floatDist(rng),
									   intenScale*floatDist(rng),
									   intenScale*floatDist(rng) }));
		errorCheck(world_set_spot_light_position(spot, { posScale*floatDist(rng),
									   posScale*floatDist(rng),
									   posScale*floatDist(rng) }));
		errorCheck(world_set_spot_light_direction(spot, { floatDist(rng),
									   floatDist(rng),
									   floatDist(rng) }));
		errorCheck(world_set_spot_light_angle(spot, openingAngle));
		errorCheck(world_set_spot_light_falloff(spot, openingAngle * floatDist(rng)));
		errorCheck(world_set_dir_light_direction(dir, { posScale*floatDist(rng),
									   posScale*floatDist(rng),
									   posScale*floatDist(rng) }));
		errorCheck(world_set_dir_light_radiance(point, { intenScale*floatDist(rng),
									   intenScale*floatDist(rng),
									   intenScale*floatDist(rng) }));

		errorCheck(scenario_add_light(scenario, pointName.c_str()));
		errorCheck(scenario_add_light(scenario, spotName.c_str()));
		errorCheck(scenario_add_light(scenario, dirName.c_str()));
		increment_string(pointName);
		increment_string(spotName);
		increment_string(dirName);
	}
}

void test_polygon() {
	std::cout << "Testing polygons" << std::endl;
	bool success = false;

	ObjectHdl obj = world_create_object();
	mAssert(obj != nullptr);
	{
		auto hdl0 = polygon_request_vertex_attribute(obj, "test0", AttribDesc{ AttributeType::ATTR_FLOAT, 1u });
		mAssert(hdl0.customIndex != INVALID_INDEX);
		auto hdl1 = polygon_request_vertex_attribute(obj, "test1", AttribDesc{ AttributeType::ATTR_USHORT, 3u });
		mAssert(hdl1.customIndex != INVALID_INDEX);
		auto hdl2 = polygon_request_face_attribute(obj, "test2", AttribDesc{ AttributeType::ATTR_DOUBLE, 4u });
		mAssert(hdl2.customIndex != INVALID_INDEX);
		float m0{ 25.f };
		ei::Vec<int, 3u> m1{ 1, 2, 3 };
		ei::Vec<unsigned char, 4u> m2{ 2, 4, 6, 8 };

		auto v0 = polygon_add_vertex(obj, { 0.f, 0.f, 0.f }, { 1.f, 0.f, 0.f }, { 0.f, 0.f });
		auto v1 = polygon_add_vertex(obj, { 1.f, 0.f, 0.f }, { 1.f, 0.f, 0.f }, { 1.f, 0.f });
		auto v2 = polygon_add_vertex(obj, { 0.f, 1.f, 0.f }, { 1.f, 0.f, 0.f }, { 0.f, 1.f });
		mAssert(v0 != INVALID_INDEX && v1 != INVALID_INDEX && v2 != INVALID_INDEX);
		FaceHdl f0 = polygon_add_triangle_material(obj, {
			static_cast<uint32_t>(v0),
			static_cast<uint32_t>(v1),
			static_cast<uint32_t>(v2)
		}, 2u);
		mAssert(f0 >= 0);
		success = polygon_set_vertex_attribute(obj, &hdl0, v0, &m0);
		mAssert(success);
		success = polygon_set_vertex_attribute(obj, &hdl1, v1, &m1);
		mAssert(success);
		success = polygon_set_face_attribute(obj, &hdl2, f0, &m2);
		mAssert(success);
		success = polygon_set_material_idx(obj, f0, 3u);
		mAssert(success);

		// TODO: read back attributes

		success = polygon_remove_vertex_attribute(obj, &hdl0);
		mAssert(success);
		success = polygon_remove_vertex_attribute(obj, &hdl1);
		mAssert(success);
		success = polygon_remove_face_attribute(obj, &hdl2);
		mAssert(success);

		// TODO: test bulk functions as well
		if(TEST_STREAM) {
			FILE* pointStream = nullptr;
			FILE* normalStream = nullptr;
			FILE* uvStream = nullptr;
			FILE* matStream = nullptr;
			std::size_t vertexCount = 4u;
			std::size_t faceCount = 1u;
			std::size_t pointsRead, normalsRead, uvsRead;

			VertexHdl bv0 = polygon_add_vertex_bulk(obj, vertexCount, pointStream, normalStream, uvStream,
													&pointsRead, &normalsRead, &uvsRead);
			mAssert(bv0 != INVALID_INDEX);
			FaceHdl f1 = polygon_add_quad(obj, {
				static_cast<uint32_t>(bv0),
				static_cast<uint32_t>(bv0 + 1),
				static_cast<uint32_t>(bv0 + 2),
				static_cast<uint32_t>(bv0 + 3),
										  });
			mAssert(f1 != INVALID_INDEX);
			std::size_t matCount = polygon_set_material_idx_bulk(obj, f1, faceCount,
																 matStream);
			mAssert(matCount != INVALID_SIZE);
		}

		ei::Box aabb;
		polygon_get_bounding_box(obj, reinterpret_cast<Vec3*>(&aabb.min),
								 reinterpret_cast<Vec3*>(&aabb.max));
		std::cout << "Bounding box: [" << aabb.min[0] << '|' << aabb.min[1]
			<< '|' << aabb.min[2] << "] - [" << aabb.max[0] << '|'
			<< aabb.max[1] << '|' << aabb.max[2] << ']' << std::endl;
	}

	success = world_create_instance(obj);
	mAssert(success);
}

void test_sphere() {
	std::cout << "Testing spheres" << std::endl;
	bool success = false;

	ObjectHdl obj = world_create_object();
	mAssert(obj != nullptr);
	{
		auto hdl0 = spheres_request_attribute(obj, "test0", AttribDesc{ AttributeType::ATTR_INT, 1u });
		mAssert(hdl0.index != INVALID_INDEX);
		auto hdl1 = spheres_request_attribute(obj, "test0", AttribDesc{ AttributeType::ATTR_UCHAR, 4u });
		mAssert(hdl1.index != INVALID_INDEX);
		int m0{ 25 };
		ei::Vec<unsigned char, 4u> m1{ 3u, 6u, 9u, 12u };
		auto s0 = spheres_add_sphere(obj, { 0.f, 0.f, 0.f }, 55.f);
		mAssert(s0 != INVALID_INDEX);

		success = spheres_set_attribute(obj, &hdl0, s0, &m0);
		mAssert(success);
		success = spheres_set_attribute(obj, &hdl1, s0, &m1);
		mAssert(success);
		success = spheres_set_material_idx(obj, s0, 13u);
		mAssert(success);

		if(TEST_STREAM) {
			FILE* sphereStream = nullptr;
			FILE* matStream = nullptr;
			std::size_t sphereCount = 4u;
			std::size_t spheresRead;
			SphereHdl bs0 = spheres_add_sphere_bulk(obj, sphereCount, sphereStream, &spheresRead);
			mAssert(bs0 != INVALID_INDEX);
			std::size_t matsRead = spheres_set_material_idx_bulk(obj, bs0, sphereCount, matStream);
			mAssert(matsRead != INVALID_SIZE);
		}

		ei::Box aabb;
		spheres_get_bounding_box(obj, reinterpret_cast<Vec3*>(&aabb.min),
								 reinterpret_cast<Vec3*>(&aabb.max));
		std::cout << "Bounding box: [" << aabb.min[0] << '|' << aabb.min[1]
			<< '|' << aabb.min[2] << "] - [" << aabb.max[0] << '|'
			<< aabb.max[1] << '|' << aabb.max[2] << ']' << std::endl;
	}

	success = world_create_instance(obj);
	mAssert(success);
}

void test_scene() {
	std::cout << "Testing scene" << std::endl;
	bool success = false;

	ScenarioHdl scenario = world_find_scenario("TestScenario");
	mAssert(scenario != nullptr);
	scenario_set_resolution(scenario, { 800, 600 });
	SceneHdl scene = world_load_scenario(scenario);
	mAssert(scene != nullptr);
	ei::Box aabb;
	success = scene_get_bounding_box(scene, reinterpret_cast<Vec3*>(&aabb.min),
									 reinterpret_cast<Vec3*>(&aabb.max));
	mAssert(success);

	std::cout << "Bounding box: [" << aabb.min[0] << '|' << aabb.min[1]
		<< '|' << aabb.min[2] << "] - [" << aabb.max[0] << '|'
		<< aabb.max[1] << '|' << aabb.max[2] << ']' << std::endl;
}

/*void test_allocator() {
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
}*/

int main() {
	test_polygon();
	test_sphere();
	test_lights();
	test_scene();
	/*test_allocator();
	test_sphere();
	test_custom_attributes();
	test_object();
	test_scene_creation();
	test_renderer();*/

	std::cout << "All tests successful" << std::endl;
	std::cin.get();
	return 0;
}