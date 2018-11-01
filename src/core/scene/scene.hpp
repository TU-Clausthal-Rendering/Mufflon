#pragma once

#include "accel_struct.hpp"
#include "instance.hpp"
#include "materials/material.hpp"
#include <memory>
#include <vector>

namespace mufflon::scene {

// Idea: On device side, Materials, Lights, and Cameras call their own functions in a switch statement

/**
 * Main class on the data side.
 * Holds all data related to a scene: materials, cameras and their paths, geometry etc.
 */
class Scene {
public:
	Scene() = default;
	Scene(const Scene&) = default;
	Scene(Scene&&) = default;
	Scene& operator=(const Scene&) = default;
	Scene& operator=(Scene&&) = default;
	~Scene() = default;

	// Adds a new instance.
	void add_instance(Instance &&instance) {
		m_instances.push_back(std::move(instance));
	}

private:
	// TODO: cameras, lights, materials
	// List of instances and thus objects to-be-rendered
	std::vector<Instance> m_instances;
	// Acceleration structure over all instances
	std::unique_ptr<IAccelerationStructure> m_accel_struct = nullptr;
};

} // namespace mufflon::scene