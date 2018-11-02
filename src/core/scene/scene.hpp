#pragma once

#include "accel_struct.hpp"
#include "instance.hpp"
#include "materials/material.hpp"
#include "ei/3dtypes.hpp"
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
		m_boundingBox = ei::Box(m_boundingBox, m_instances.back().get_bounding_box());
	}

	const ei::Box& get_bounding_box() const noexcept {
		return m_boundingBox;
	}

private:
	// TODO: cameras, lights, materials
	// List of instances and thus objects to-be-rendered
	std::vector<Instance> m_instances;
	// Acceleration structure over all instances
	std::unique_ptr<IAccelerationStructure> m_accel_struct = nullptr;
	ei::Box m_boundingBox;
};

} // namespace mufflon::scene