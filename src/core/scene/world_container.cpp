#include "world_container.hpp"
#include "util/log.hpp"
#include "core/cameras/camera.hpp"
#include "core/scene/materials/material.hpp"

namespace mufflon::scene {

ObjectHandle WorldContainer::create_object() {
	m_objects.emplace_back();
	return &m_objects.back();
}

ObjectHandle WorldContainer::add_object(Object&& obj) {
	m_objects.emplace_back(std::move(obj));
	return &m_objects.back();
}

InstanceHandle WorldContainer::create_instance(ObjectHandle obj) {
	if(obj == nullptr) {
		logError("[WorldContainer::create_instance] Invalid object handle");
		return nullptr;
	}
	m_instances.emplace_back(*obj);
	return &m_instances.back();
}

// Adds a new instance.
InstanceHandle WorldContainer::add_instance(Instance &&instance) {
	m_instances.emplace_back(std::move(instance));
	return &m_instances.back();
}

WorldContainer::ScenarioHandle WorldContainer::add_scenario(Scenario&& scenario) {
	return m_scenarios.emplace(scenario.get_name(), std::move(scenario)).first;
}

std::optional<WorldContainer::ScenarioHandle> WorldContainer::get_scenario(const std::string_view& name) {
	auto iter = m_scenarios.find(name);
	if(iter != m_scenarios.end())
		return iter;
	return std::nullopt;
}

MaterialHandle WorldContainer::add_material(std::unique_ptr<materials::IMaterial> material) {
	m_materials.push_back(move(material));
	return m_materials.back().get();
}

CameraHandle WorldContainer::add_camera(std::unique_ptr<cameras::Camera> camera) {
	std::string_view name = camera->get_name();
	CameraHandle handle = camera.get();
	m_cameras.emplace(name, move(camera));
	return handle;
}

CameraHandle WorldContainer::get_camera(std::string_view name) {
	auto it = m_cameras.find(name);
	if(it == m_cameras.end()) {
		logError("[WorldContainer::get_camera] Cannot find a camera with name '", name, "'");
		return nullptr;
	}
	return it->second.get();
}

std::optional<WorldContainer::PointLightHandle> WorldContainer::add_light(std::string name,
																		  lights::PointLight&& light) {
	if(m_pointLights.find(name) != m_pointLights.cend()) {
		logError("[WorldContainer::add_light] Point light with name '", name, "' already exists");
		return std::nullopt;
	}
	return m_pointLights.insert({ std::move(name), std::move(light) }).first;

}

std::optional<WorldContainer::SpotLightHandle> WorldContainer::add_light(std::string name,
																		 lights::SpotLight&& light) {
	if(m_spotLights.find(name) != m_spotLights.cend()) {
		logError("[WorldContainer::add_light] Spot light with name '", name, "' already exists");
		return std::nullopt;
	}
	return m_spotLights.insert({ std::move(name), std::move(light) }).first;

}

std::optional<WorldContainer::DirLightHandle> WorldContainer::add_light(std::string name,
																		lights::DirectionalLight&& light) {
	if(m_dirLights.find(name) != m_dirLights.cend()) {
		logError("[WorldContainer::add_light] Directional light with name '", name, "' already exists");
		return std::nullopt;
	}
	return m_dirLights.insert({ std::move(name), std::move(light) }).first;
}
std::optional<WorldContainer::EnvLightHandle> WorldContainer::add_light(std::string name,
																		textures::TextureHandle env) {
	if(m_envLights.find(name) != m_envLights.cend()) {
		logError("[WorldContainer::add_light] Envmap light with name '", name, "' already exists");
		return std::nullopt;
	}
	return m_envLights.insert({ std::move(name), env }).first;
}

std::optional<WorldContainer::PointLightHandle> WorldContainer::get_point_light(const std::string_view& name) {
	auto light = m_pointLights.find(name);
	if(light == m_pointLights.end())
		return std::nullopt;
	return light;
}

std::optional<WorldContainer::SpotLightHandle> WorldContainer::get_spot_light(const std::string_view& name) {
	auto light = m_spotLights.find(name);
	if(light == m_spotLights.end()) 
		return std::nullopt;
	return light;
}

std::optional<WorldContainer::DirLightHandle> WorldContainer::get_dir_light(const std::string_view& name) {
	auto light = m_dirLights.find(name);
	if(light == m_dirLights.end())
		return std::nullopt;
	return light;
}

std::optional<WorldContainer::EnvLightHandle> WorldContainer::get_env_light(const std::string_view& name) {
	auto light = m_envLights.find(name);
	if(light == m_envLights.end())
		return std::nullopt;
	return light;
}

bool WorldContainer::is_point_light(const std::string_view& name) const {
	return m_pointLights.find(name) != m_pointLights.cend();
}

bool WorldContainer::is_spot_light(const std::string_view& name) const {
	return m_spotLights.find(name) != m_spotLights.cend();
}

bool WorldContainer::is_dir_light(const std::string_view& name) const {
	return m_dirLights.find(name) != m_dirLights.cend();
}

bool WorldContainer::is_env_light(const std::string_view& name) const {
	return m_envLights.find(name) != m_envLights.cend();
}

std::optional<std::string_view> WorldContainer::get_light_name_ref(const std::string_view& name) const noexcept {
	auto pointLight = m_pointLights.find(name);
	if(pointLight != m_pointLights.cend())
		return pointLight->first;
	auto spotLight = m_spotLights.find(name);
	if(spotLight != m_spotLights.cend())
		return spotLight->first;
	auto dirLight = m_dirLights.find(name);
	if(dirLight != m_dirLights.cend())
		return dirLight->first;
	auto envLight = m_envLights.find(name);
	if(envLight != m_envLights.cend())
		return envLight->first;
	logError("[WorldContainer::get_light_name_ref] Unknown light '", name, "'");
	return std::nullopt;
}

SceneHandle WorldContainer::load_scene(const Scenario& scenario) {
	std::vector<lights::PositionalLights> posLights;
	std::vector<lights::DirectionalLight> dirLights;
	std::optional<EnvLightHandle> envLightTex;
	posLights.reserve(m_pointLights.size() + m_spotLights.size());
	dirLights.reserve(m_dirLights.size());

	m_scene = std::make_unique<Scene>();
	for(auto& instance : m_instances) {
		if(!scenario.is_masked(&instance.get_object())) {
			m_scene->add_instance(&instance);
			// TODO: add area light here!
		}
	}

	// Add regular lights
	for(const std::string_view& name : scenario.get_light_names()) {
		if(auto pointLight = get_point_light(name); pointLight.has_value()) {
			posLights.push_back(pointLight.value()->second);
		} else if(auto spotLight = get_spot_light(name); spotLight.has_value()) {
			posLights.push_back(spotLight.value()->second);
		} else if(auto dirLight = get_dir_light(name); dirLight.has_value()) {
			dirLights.push_back(dirLight.value()->second);
		} else if(auto envLight = get_env_light(name); envLight.has_value()) {
			if(envLightTex.has_value())
				logWarning("[WorldContainer::load_scene] Multiple envmap lights are not supported; replacing '",
						   envLightTex.value()->first, "' with '",
						   envLight.value()->first);
			envLightTex = envLight;
		} else {
			logWarning("[WorldContainer::load_scene] Unknown light source '", name, "' in scenario '",
					   scenario.get_name(), "'");
		}
	}

	if(envLightTex.has_value())
		m_scene->set_lights(std::move(posLights), std::move(dirLights),
							envLightTex.value()->second);
	else
		m_scene->set_lights(std::move(posLights), std::move(dirLights));

	// TODO: load the materials (make something resident?)
	// TODO: cameras light, etc.

	// Assign the newly created scene and destroy the old one?
	return m_scene.get();
}

SceneHandle WorldContainer::load_scene(ScenarioHandle hdl) {
	return load_scene(hdl->second);
}

} // namespace mufflon::scene