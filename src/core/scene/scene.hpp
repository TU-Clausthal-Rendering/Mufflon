#pragma once

#include "instance.hpp"
#include "object.hpp"
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
	struct DeviceScene {
		Object::DeviceObject* objects;
		Instance::DeviceInstance* instances;
		// TODO: camera and stuff
	};

	Scene() = default;
	Scene(const Scene&) = default;
	Scene(Scene&&) = default;
	Scene& operator=(const Scene&) = default;
	Scene& operator=(Scene&&) = default;
	~Scene() = default;

	// Creates a new, empty object in the scene.
	Object& create_object();
	// Adds a new instance.
	void add_instance(Instance instance);

private:
	// TODO: cameras, lights, materials
	std::vector<Object> m_objects;
	std::vector<Instance> m_instances;
};

} // namespace mufflon::scene