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

const std::string& WorldContainer::get_scenario_name(std::size_t index) {
	mAssert(index < m_scenarios.size());
	auto iter = m_scenarios.cbegin();
	for(std::size_t i = 0; i < index; ++i)
		++iter;
	return iter->first;
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
	m_cameraHandles.push_back(iter.first);
	return iter.first->second.get();
}

void WorldContainer::remove_camera(CameraHandle hdl) {
	if(hdl == nullptr)
		return;
	auto iter = m_cameras.begin();
	for(std::size_t i = 0u; i < m_cameras.size(); ++i) {
		if(hdl == iter->second.get()) {
			m_cameras.erase(iter);
			m_cameraHandles.erase(m_cameraHandles.begin() + i);
			break;
		}
		++iter;
	}
}

CameraHandle WorldContainer::get_camera(std::string_view name) {
	auto it = m_cameras.find(name);
	if(it == m_cameras.end()) {
		logError("[WorldContainer::get_camera] Cannot find a camera with name '", name, "'");
		return nullptr;
	}
	return it->second.get();
}

CameraHandle WorldContainer::get_camera(std::size_t index) {
	if(index >= m_cameras.size())
		throw std::runtime_error("Camera index out of bounds (" + std::to_string(index)
								 + " >= " + std::to_string(m_cameras.size()));
	return m_cameraHandles[index]->second.get();
}

std::optional<u32> WorldContainer::add_light(std::string name,
											 lights::PointLight&& light) {
	if(m_pointLights.find(name) != nullptr) {
		logError("[WorldContainer::add_light] Point light with name '", name, "' already exists");
		return std::nullopt;
	}
	return m_pointLights.insert(std::move(name), std::move(light));
}

std::optional<u32> WorldContainer::add_light(std::string name,
											 lights::SpotLight&& light) {
	if(m_spotLights.find(name) != nullptr) {
		logError("[WorldContainer::add_light] Spot light with name '", name, "' already exists");
		return std::nullopt;
	}
	return m_spotLights.insert(std::move(name), std::move(light));

}

std::optional<u32> WorldContainer::add_light(std::string name,
											 lights::DirectionalLight&& light) {
	if(m_dirLights.find(name) != nullptr) {
		logError("[WorldContainer::add_light] Directional light with name '", name, "' already exists");
		return std::nullopt;
	}
	return m_dirLights.insert(std::move(name), std::move(light));
}

std::optional<u32> WorldContainer::add_light(std::string name,
											 TextureHandle env) {
	if(m_envLights.find(name) != nullptr) {
		logError("[WorldContainer::add_light] Envmap light with name '", name, "' already exists");
		return std::nullopt;
	}
	return m_envLights.insert(std::move(name), std::move(env));
}

std::optional<std::pair<u32, lights::LightType>> WorldContainer::find_light(const std::string_view& name) {
	if(m_pointLights.find(name) != nullptr)
		return std::make_pair(u32(m_pointLights.get_index(name)), lights::LightType::POINT_LIGHT);
	if(m_spotLights.find(name) != nullptr)
		return std::make_pair(u32(m_spotLights.get_index(name)), lights::LightType::SPOT_LIGHT);
	if(m_dirLights.find(name) != nullptr)
		return std::make_pair(u32(m_dirLights.get_index(name)), lights::LightType::DIRECTIONAL_LIGHT);
	if(m_envLights.find(name) != nullptr)
		return std::make_pair(u32(m_envLights.get_index(name)), lights::LightType::ENVMAP_LIGHT);
	return std::nullopt;
}

lights::PointLight* WorldContainer::get_point_light(u32 index) {
	if(index >= m_pointLights.size())
		throw std::runtime_error("Point light index out of bounds (" + std::to_string(index)
									+ " >= " + std::to_string(m_pointLights.size()));
	return &m_pointLights.get(index);
}

lights::SpotLight* WorldContainer::get_spot_light(u32 index) {
	if(index >= m_spotLights.size())
		throw std::runtime_error("Spot light index out of bounds (" + std::to_string(index)
									+ " >= " + std::to_string(m_spotLights.size()));
	return &m_spotLights.get(index);
}

lights::DirectionalLight* WorldContainer::get_dir_light(u32 index) {
	if(index >= m_dirLights.size())
		throw std::runtime_error("Directional light index out of bounds (" + std::to_string(index)
									+ " >= " + std::to_string(m_dirLights.size()));
	return &m_dirLights.get(index);
}

TextureHandle& WorldContainer::get_env_light(u32 index) {
	if(index >= m_envLights.size())
		throw std::runtime_error("Envmap light index out of bounds (" + std::to_string(index)
									+ " >= " + std::to_string(m_envLights.size()));
	return m_envLights.get(index);
}


void WorldContainer::remove_light(u32 index, lights::LightType type) {
	switch(type) {
		case lights::LightType::POINT_LIGHT: {
			if(index >= m_pointLights.size())
				throw std::runtime_error("Point light index out of bounds (" + std::to_string(index)
										 + " >= " + std::to_string(m_pointLights.size()));
			m_pointLights.erase(index);
		} break;
		case lights::LightType::SPOT_LIGHT: {
			if(index >= m_spotLights.size())
				throw std::runtime_error("Spot light index out of bounds (" + std::to_string(index)
										 + " >= " + std::to_string(m_spotLights.size()));
			m_spotLights.erase(index);
		} break;
		case lights::LightType::DIRECTIONAL_LIGHT: {
			if(index >= m_dirLights.size())
				throw std::runtime_error("Directional light index out of bounds (" + std::to_string(index)
											+ " >= " + std::to_string(m_dirLights.size()));
			m_dirLights.erase(index);
		} break;
		case lights::LightType::ENVMAP_LIGHT: {
			if(index >= m_envLights.size())
				throw std::runtime_error("Envmap light index out of bounds (" + std::to_string(index)
										 + " >= " + std::to_string(m_envLights.size()));
			m_envLights.erase(index);
			// TODO: delete texture?
		} break;
		default:
			throw std::runtime_error("[WorldContainer::remove_light] Invalid light type.");
	}
}

std::string_view WorldContainer::get_light_name(u32 index, lights::LightType type) const {
	switch(type) {
		case lights::LightType::POINT_LIGHT: return m_pointLights.get_key(index);
		case lights::LightType::SPOT_LIGHT: return m_spotLights.get_key(index);
		case lights::LightType::DIRECTIONAL_LIGHT: return m_dirLights.get_key(index);
		case lights::LightType::ENVMAP_LIGHT: return m_envLights.get_key(index);
		default:
			throw std::runtime_error("[WorldContainer::get_light_name] Invalid light type.");
	}
	return "";
}

void WorldContainer::mark_camera_dirty(ConstCameraHandle hdl) {
	// TODO: put the mark-dirty on scenario/scene level (one for add/remove, one for touched)?
	// We need one for the env-map for sure, because that operation takes long
	if(hdl == nullptr)
		return;
	auto iter = m_cameras.begin();
	for(std::size_t i = 0u; i < m_cameras.size(); ++i) {
		if(hdl == iter->second.get()) {
			m_camerasDirty[i] = true;
			break;
		}
	}
}

void WorldContainer::mark_envmap_light_dirty(TextureHandle* hdl) {

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

std::optional<std::string_view> WorldContainer::get_texture_name(TextureHandle hdl) const {
	// Gotta iterate entire map... very expensive operation
	// TODO: use bimap?
	for(auto iter = m_textures.cbegin(); iter != m_textures.cend(); ++iter) {
		if(&iter->second == hdl)
			return iter->first;
	}
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

SceneHandle WorldContainer::load_scene(Scenario& scenario) {
	std::vector<lights::PositionalLights> posLights;
	std::vector<lights::DirectionalLight> dirLights;
	std::optional<EnvLightHandle> envLightTex;
	posLights.reserve(m_pointLights.size() + m_spotLights.size());
	dirLights.reserve(m_dirLights.size());

	m_scenario = &scenario;
	m_scene = std::make_unique<Scene>(scenario.get_camera(), m_materials);
	u32 instIdx = 0;
	for(auto& instance : m_instances) {
		if(!scenario.is_masked(&instance.get_object()))
			m_scene->add_instance(&instance);
	}

	// Check if the resulting scene has issues with size
	if(ei::len(m_scene->get_bounding_box().min) >= SUGGESTED_MAX_SCENE_SIZE
	   || ei::len(m_scene->get_bounding_box().max) >= SUGGESTED_MAX_SCENE_SIZE)
		logWarning("[WorldContainer::load_scene] Scene size is larger than recommended "
				   "(Furthest point of the bounding box should not be further than "
				   "2^20m away)");

	// Load the lights
	this->load_scene_lights();

	// Make media available / resident
	m_scene->load_media(m_media);

	m_scenario->camera_dirty_reset();

	// Assign the newly created scene and destroy the old one?
	return m_scene.get();
}

SceneHandle WorldContainer::load_scene(ScenarioHandle hdl) {
	mAssert(hdl != nullptr);
	return load_scene(*hdl);
}

SceneHandle WorldContainer::reload_scene() {
	// There is always a scenario set
	mAssert(m_scenario != nullptr);

	if(m_scene == nullptr)
		return load_scene(*m_scenario);

	if(m_scenario->lights_dirty_reset()) {
		this->load_scene_lights();
	}

	// TODO: dirty flag for media?
	// TODO: dirty flag for materials?

	if(m_scenario->camera_dirty_reset())
		m_scene->set_camera(m_scenario->get_camera());
	return m_scene.get();
}

void WorldContainer::load_scene_lights() {
	std::vector<lights::PositionalLights> posLights;
	std::vector<lights::DirectionalLight> dirLights;
	TextureHandle envLightTex = nullptr;
	u32 instIdx = 0;
	for(auto& instance : m_instances) {
		if(!m_scenario->is_masked(&instance.get_object())) {
			// Find all area lights, if the object contains some
			if(instance.get_object().is_emissive()) {
				u32 primIdx = 0;
				// First search in polygons (PrimitiveHandle expects poly before sphere)
				auto& polygons = instance.get_object().get_geometry<geometry::Polygons>();
				const MaterialIndex* materials = polygons.acquire_const<Device::CPU, MaterialIndex, true>(polygons.get_material_indices_hdl());
				const scene::Point* positions = polygons.acquire_const<Device::CPU, scene::Point, false>(polygons.get_points_hdl());
				const scene::UvCoordinate* uvs = polygons.acquire_const<Device::CPU, scene::UvCoordinate, false>(polygons.get_uvs_hdl());
				for(const auto& face : polygons.faces()) {
					if(m_materials[materials[primIdx]]->get_properties().is_emissive()) {
						auto emission = m_materials[materials[primIdx]]->get_emission();
						mAssert(emission.texture != nullptr);
						if(std::distance(face.begin(), face.end()) == 3) {
							lights::AreaLightTriangleDesc al;
							al.radianceTex = emission.texture;
							al.scale = ei::packRGB9E5(emission.scale);
							int i = 0;
							for(auto vHdl : face) {
								al.points[i] = positions[vHdl.idx()];
								al.uv[i] = uvs[vHdl.idx()];
								++i;
							}
							posLights.push_back({ al, PrimitiveHandle{ u64(instIdx) << 32ull | primIdx } });
						} else {
							lights::AreaLightQuadDesc al;
							al.radianceTex = emission.texture;
							al.scale = ei::packRGB9E5(emission.scale);
							int i = 0;
							for(auto vHdl : face) {
								al.points[i] = positions[vHdl.idx()];
								al.uv[i] = uvs[vHdl.idx()];
								++i;
							}
							posLights.push_back({ al, PrimitiveHandle{ u64(instIdx) << 32ull | primIdx } });
						}
					}
					++primIdx;
				}

				// Then get the sphere lights
				auto& spheres = instance.get_object().get_geometry<geometry::Spheres>();
				materials = spheres.acquire_const<Device::CPU, MaterialIndex>(spheres.get_material_indices_hdl());
				const ei::Sphere* spheresData = spheres.acquire_const<Device::CPU, ei::Sphere>(spheres.get_spheres_hdl());
				for(std::size_t i = 0; i < spheres.get_sphere_count(); ++i) {
					if(m_materials[materials[i]]->get_properties().is_emissive()) {
						auto emission = m_materials[materials[primIdx]]->get_emission();
						mAssert(emission.texture != nullptr);
						lights::AreaLightSphereDesc al{
							spheresData[i].center,
							spheresData[i].radius,
							emission.texture, ei::packRGB9E5(emission.scale)
						};
						posLights.push_back({ al, u64(instIdx) << 32ull | primIdx });
					}
					++primIdx;
				}
			}
		}
		++instIdx;
	}

	posLights.reserve(posLights.size() + m_pointLights.size() + m_spotLights.size());
	dirLights.reserve(dirLights.size() + m_dirLights.size());

	// Add regular lights
	std::string_view prevEnvName;
	for(const std::string_view& name : m_scenario->get_light_names()) {
		if(auto pointLight = m_pointLights.find(name); pointLight) {
			posLights.push_back({ *pointLight, ~0u });
		} else if(auto spotLight = m_spotLights.find(name); spotLight) {
			posLights.push_back({ *spotLight, ~0u });
		} else if(auto dirLight = m_dirLights.find(name); dirLight) {
			dirLights.push_back(*dirLight);
		} else if(auto envLight = m_envLights.find(name); envLight) {
			if(envLightTex)
				logWarning("[WorldContainer::load_scene] Multiple envmap lights are not supported; replacing '",
						   prevEnvName, "' with '", name);
			envLightTex = *envLight;
			prevEnvName = name;
		} else {
			logWarning("[WorldContainer::load_scene] Unknown light source '", name, "' in scenario '",
					   m_scenario->get_name(), "'");
		}
	}

	if(envLightTex)
		m_scene->set_lights(std::move(posLights), std::move(dirLights),
							envLightTex);
	else
		m_scene->set_lights(std::move(posLights), std::move(dirLights));

	m_scenario->lights_dirty_reset();
}

} // namespace mufflon::scene
