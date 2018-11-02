#include "world_container.hpp"
#include "util/log.hpp"

namespace mufflon::scene {

material::MaterialHandle WorldContainer::add_material(std::unique_ptr<material::IMaterial> material) {
	m_materials.push_back(move(material));
	return m_materials.back().get();
}

WorldContainer::ObjectHandle WorldContainer::create_object() {
	m_objects.emplace_back();
	return &m_objects.back();
}

WorldContainer::ObjectHandle WorldContainer::add_object(Object&& obj) {
	m_objects.emplace_back(std::move(obj));
	return &m_objects.back();
}

WorldContainer::InstanceHandle WorldContainer::create_instance(ObjectHandle obj) {
	if(obj == nullptr) {
		logError("[WorldContainer::create_instance] Invalid object handle");
		return nullptr;
	}
	m_instances.emplace_back(*obj);
	return &m_instances.back();
}

// Adds a new instance.
WorldContainer::InstanceHandle WorldContainer::add_instance(Instance &&instance) {
	m_instances.emplace_back(std::move(instance));
	return &m_instances.back();
}


WorldContainer::ScenarioHandle WorldContainer::create_scenario() {
	m_scenarios.emplace_back();
	return &m_scenarios.back();
}

WorldContainer::ScenarioHandle WorldContainer::add_scenario(Scenario&& scenario) {
	m_scenarios.emplace_back(std::move(scenario));
	return &m_scenarios.back();
}

WorldContainer::SceneHandle WorldContainer::load_scene(ScenarioHandle hdl) {
	if(hdl == nullptr) {
		logError("[WorldContainer::create_instance] Invalid scenario handle");
		return nullptr;
	}
	Scenario& scenario = *hdl;
	m_scene = std::make_unique<Scene>();
	for(auto& instance : m_instances) {
		if(!scenario.is_masked(&instance.get_object()))
			m_scene->add_instance(&instance);
	}

	// TODO: load the materials
	// TODO: camera, light, etc.

	// Assign the newly created scene and destroy the old one?
	return m_scene.get();
}

} // namespace mufflon::scene