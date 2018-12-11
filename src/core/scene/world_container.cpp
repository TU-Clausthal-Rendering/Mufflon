#include "world_container.hpp"
#include "util/log.hpp"
#include "core/cameras/camera.hpp"
#include "core/scene/materials/material.hpp"
#include "core/scene/materials/medium.hpp"
#include <iostream>

namespace mufflon::scene {

WorldContainer WorldContainer::s_container{};

void WorldContainer::clear_instance() {
	s_container = WorldContainer();
}

ObjectHandle WorldContainer::create_object(std::string name, ObjectFlags flags) {
	auto hdl = m_objects.emplace(std::move(name), Object{});
	if(!hdl.second)
		return nullptr;
	hdl.first->second.set_name(hdl.first->first);
	hdl.first->second.set_flags(flags);
	return &hdl.first->second;
}

ObjectHandle WorldContainer::get_object(const std::string_view& name) {
	auto iter = m_objects.find(name);
	if(iter != m_objects.end())
		return &iter->second;
	return nullptr;
}

InstanceHandle WorldContainer::create_instance(ObjectHandle obj) {
	if(obj == nullptr) {
		logError("[WorldContainer::create_instance] Invalid object handle");
		return nullptr;
	}
	m_instances.emplace_back(*obj);
	return &m_instances.back();
}

ScenarioHandle WorldContainer::create_scenario(std::string name) {
	// TODO: switch to pointer
	auto hdl = m_scenarios.emplace(std::move(name), Scenario{}).first;
	hdl->second.set_name(hdl->first);
	return &hdl->second;
}

ScenarioHandle WorldContainer::get_scenario(const std::string_view& name) {
	auto iter = m_scenarios.find(name);
	if(iter != m_scenarios.end())
		return &iter->second;
	return nullptr;
}

MaterialHandle WorldContainer::add_material(std::unique_ptr<materials::IMaterial> material) {
	m_materials.push_back(move(material));
	return m_materials.back().get();
}

materials::MediumHandle WorldContainer::add_medium(const materials::Medium& medium) {
	materials::MediumHandle h = 0;
	// Check for duplicates
	for(auto& m : m_media) {
		if(m == medium) return h;
		++h;
	}
	m_media.push_back(medium);
	return h;
}

CameraHandle WorldContainer::add_camera(std::string name, std::unique_ptr<cameras::Camera> camera) {
	auto iter = m_cameras.emplace(name, move(camera));
	if(!iter.second)
		return nullptr;
	iter.first->second->set_name(iter.first->first);
	return iter.first->second.get();
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
	// TODO: switch to pointers
	if(m_pointLights.find(name) != m_pointLights.cend()) {
		logError("[WorldContainer::add_light] Point light with name '", name, "' already exists");
		return std::nullopt;
	}
	return m_pointLights.insert({ std::move(name), std::move(light) }).first;

}

std::optional<WorldContainer::SpotLightHandle> WorldContainer::add_light(std::string name,
																		 lights::SpotLight&& light) {
	// TODO: switch to pointers
	if(m_spotLights.find(name) != m_spotLights.cend()) {
		logError("[WorldContainer::add_light] Spot light with name '", name, "' already exists");
		return std::nullopt;
	}
	return m_spotLights.insert({ std::move(name), std::move(light) }).first;

}

std::optional<WorldContainer::DirLightHandle> WorldContainer::add_light(std::string name,
																		lights::DirectionalLight&& light) {
	// TODO: switch to pointers
	if(m_dirLights.find(name) != m_dirLights.cend()) {
		logError("[WorldContainer::add_light] Directional light with name '", name, "' already exists");
		return std::nullopt;
	}
	return m_dirLights.insert({ std::move(name), std::move(light) }).first;
}
std::optional<WorldContainer::EnvLightHandle> WorldContainer::add_light(std::string name,
																		TextureHandle env) {
	// TODO: switch to pointers
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

bool WorldContainer::has_texture(std::string_view name) const {
	return m_textures.find(name) != m_textures.cend();
}

std::optional<WorldContainer::TexCacheHandle> WorldContainer::find_texture(std::string_view name) {
	auto iter = m_textures.find(name);
	if(iter != m_textures.end())
		return iter;
	return std::nullopt;
}

WorldContainer::TexCacheHandle WorldContainer::add_texture(std::string_view path, u16 width,
											  u16 height, u16 numLayers,
											  textures::Format format, textures::SamplingMode mode,
											  bool sRgb, std::unique_ptr<u8[]> data) {
	// TODO: ensure that we have at least 1x1 pixels?
	return m_textures.emplace(path, textures::Texture{ width, height, numLayers,
							  format, mode, sRgb, move(data) }).first;
}

SceneHandle WorldContainer::load_scene(const Scenario& scenario) {
	std::vector<lights::PositionalLights> posLights;
	std::vector<lights::DirectionalLight> dirLights;
	std::optional<EnvLightHandle> envLightTex;
	posLights.reserve(m_pointLights.size() + m_spotLights.size());
	dirLights.reserve(m_dirLights.size());

	m_scenario = &scenario;
	m_scene = std::make_unique<Scene>(scenario.get_camera(), m_materials);
	u32 instIdx = 0;
	for(auto& instance : m_instances) {
		if(!scenario.is_masked(&instance.get_object())) {
			m_scene->add_instance(&instance);
			// Find all area lights, if the object contains some
			if(instance.get_object().is_emissive()) {
				u32 primIdx = 0;
				// First search in polygons (PrimitiveHandle expects poly before sphere)
				auto& polygons = instance.get_object().get_geometry<geometry::Polygons>();
				const MaterialIndex* materials = polygons.get_mat_indices().aquireConst<Device::CPU>();
				const scene::Point* positions = as<scene::Point>(polygons.get_points().aquireConst<Device::CPU>());
				const scene::UvCoordinate* uvs = as<scene::UvCoordinate>(polygons.get_uvs().aquireConst<Device::CPU>());
				for(const auto& face : polygons.faces()) {
					if(m_materials[materials[primIdx]]->is_emissive()) {
						if(std::distance(face.begin(), face.end()) == 3) {
							lights::AreaLightTriangleDesc al;
							al.radianceTex = m_materials[materials[primIdx]]->get_emissive_texture();
							int i = 0;
							for(auto vHdl : face) {
								al.points[i] = positions[vHdl.idx()];
								al.uv[i] = uvs[vHdl.idx()];
								++i;
							}
							posLights.push_back({al, u64(instIdx) << 32ull | primIdx});
						} else {
							lights::AreaLightQuadDesc al;
							al.radianceTex = m_materials[materials[primIdx]]->get_emissive_texture();
							int i = 0;
							for(auto vHdl : face) {
								al.points[i] = positions[vHdl.idx()];
								al.uv[i] = uvs[vHdl.idx()];
								++i;
							}
							posLights.push_back({al, u64(instIdx) << 32ull | primIdx});
						}
					}
					++primIdx;
				}

				// Then get the sphere lights
				auto& spheres = instance.get_object().get_geometry<geometry::Spheres>();
				materials = spheres.get_mat_indices().aquireConst<Device::CPU>();
				const ei::Sphere* spheresData = spheres.get_spheres().aquireConst<Device::CPU>();
				for(std::size_t i = 0; i < spheres.get_mat_indices().get_size(); ++i) {
					if(m_materials[materials[i]]->is_emissive()) {
						lights::AreaLightSphereDesc al{
							spheresData[i].center,
							spheresData[i].radius,
							m_materials[materials[i]]->get_emissive_texture()
						};
						posLights.push_back({al, u64(instIdx) << 32ull | primIdx});
					}
					++primIdx;
				}
			}
		}
		++instIdx;
	}

	// Check if the resulting scene has issues with size
	if(ei::len(m_scene->get_bounding_box().min) >= SUGGESTED_MAX_SCENE_SIZE
	   || ei::len(m_scene->get_bounding_box().max) >= SUGGESTED_MAX_SCENE_SIZE)
		logWarning("[WorldContainer::load_scene] Scene size is larger than recommended "
				   "(Furthest point of the bounding box should not be further than "
				   "2^20m away)");

	// Add regular lights
	for(const std::string_view& name : scenario.get_light_names()) {
		if(auto pointLight = get_point_light(name); pointLight.has_value()) {
			posLights.push_back({pointLight.value()->second, ~0u});
		} else if(auto spotLight = get_spot_light(name); spotLight.has_value()) {
			posLights.push_back({spotLight.value()->second, ~0u});
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

	// Make media available / resident
	m_scene->load_media(m_media);

	// Assign the newly created scene and destroy the old one?
	return m_scene.get();
}

SceneHandle WorldContainer::load_scene(ConstScenarioHandle hdl) {
	mAssert(hdl != nullptr);
	return load_scene(*hdl);
}

} // namespace mufflon::scene