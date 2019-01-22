#include "world_container.hpp"
#include "util/log.hpp"
#include "core/cameras/camera.hpp"
#include "core/scene/lod.hpp"
#include "core/scene/materials/material.hpp"
#include "core/scene/materials/medium.hpp"
#include <iostream>
#include <ei/conversions.hpp>
#include <windows.h>


namespace mufflon::scene {

WorldContainer WorldContainer::s_container{};

WorldContainer::WorldContainer() {
	m_envLights.insert("##DefaultBlack##", lights::Background::black());
}

void WorldContainer::clear_instance() {
	s_container = WorldContainer();
}

WorldContainer::Sanity WorldContainer::is_sane_world() const {
	// Check for objects
	if(m_instances.empty())
		return Sanity::NO_INSTANCES;
	if(m_objects.empty())
		return Sanity::NO_OBJECTS;
	// Check for lights
	if(m_pointLights.empty() && m_spotLights.empty() && m_dirLights.empty() && m_envLights.empty()) {
		// No explicit lights - check if any emitting materials exist
		bool hasEmitters = false;
		for(const auto& mat : m_materials)
			if(mat->get_properties().is_emissive()) {
				hasEmitters = true;
				break;
			}
		if(!hasEmitters)
			return Sanity::NO_LIGHTS;
	}

	// Check for cameras
	if(m_cameras.empty())
		return Sanity::NO_CAMERA;
	return Sanity::SANE;
}


WorldContainer::Sanity WorldContainer::is_sane_scenario(ConstScenarioHandle hdl) const {
	bool hasEmitters = false;
	bool hasObjects = false;
	// Check for objects (and check for emitters as well)
	for(const auto& instance : m_instances) {
		if(!hdl->is_masked(instance.second.get())) {
			hasObjects = true;
			const Object& object = instance.second->get_object();
			const u32 lodLevel = hdl->get_effective_lod(instance.second.get());
			if(object.has_lod_available(lodLevel)) {
				const Lod& lod = object.get_lod(lodLevel);
				if(lod.is_emissive(*hdl)) {
					hasEmitters = true;
					break;
				}
			} else {
				// TODO: how could we possibly check this?
				hasEmitters = true;
				break;
			}
		}
	}
	if(!hasObjects)
		return Sanity::NO_OBJECTS;
	// Check for lights
	if(!hasEmitters && hdl->get_point_lights().empty() && hdl->get_spot_lights().empty()
	   && hdl->get_dir_lights().empty() && (m_envLights.get(hdl->get_background()).get_type() == lights::BackgroundType::COLORED
											&& ei::prod(m_envLights.get(hdl->get_background()).get_color()) == 0))
		return Sanity::NO_LIGHTS;
	// Check for camera
	if(hdl->get_camera() == nullptr)
		return Sanity::NO_CAMERA;
	return Sanity::SANE;
}

ObjectHandle WorldContainer::create_object(std::string name, ObjectFlags flags) {
	auto hdl = m_objects.emplace(std::move(name), Object{static_cast<u32>(m_objects.size())});
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

InstanceHandle WorldContainer::get_instance(const std::string_view& name) {
	auto iter = m_instances.find(name);
	if(iter != m_instances.end())
		return iter->second.get();
	return nullptr;
}

InstanceHandle WorldContainer::create_instance(std::string name, ObjectHandle obj) {
	if(obj == nullptr) {
		logError("[WorldContainer::create_instance] Invalid object handle");
		return nullptr;
	}
	auto instance = std::make_unique<Instance>(move(name), *obj);
	std::string_view nameRef = instance->get_name();
	return m_instances.emplace(nameRef, std::move(instance)).first->second.get();
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

ScenarioHandle WorldContainer::get_scenario(std::size_t index) {
	mAssert(index < m_scenarios.size());
	// TODO: use index string map?
	auto iter = m_scenarios.begin();
	for(std::size_t i = 0; i < index; ++i)
		++iter;
	return &iter->second;
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
	m_camerasDirty.emplace(camera.get(), true);
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
	if(env) {
		mAssertMsg(env->get_sampling_mode() == textures::SamplingMode::NEAREST,
			"Sampling mode must be nearest, otherwise the importance sampling is biased.");
		return m_envLights.insert(std::move(name), lights::Background::envmap(env));
	}
	return m_envLights.insert(std::move(name), lights::Background::black());
}

void WorldContainer::replace_envlight_texture(u32 index, TextureHandle replacement) {
	if(index >= m_envLights.size())
		throw std::runtime_error("Envmap light index out of bounds (" + std::to_string(index)
								 + " >= " + std::to_string(m_envLights.size()) + ")");
	if(replacement) mAssertMsg(replacement->get_sampling_mode() == textures::SamplingMode::NEAREST,
		"Sampling mode must be nearest, otherwise the importance sampling is biased.");
	lights::Background& desc = m_envLights.get(index);
	if(replacement != desc.get_envmap()) {
		this->unref_texture(desc.get_envmap());
		desc = lights::Background::envmap(replacement);
	}
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

lights::Background* WorldContainer::get_background(u32 index) {
	if(index >= m_envLights.size())
		throw std::runtime_error("Envmap light index out of bounds (" + std::to_string(index)
									+ " >= " + std::to_string(m_envLights.size()));
	return &m_envLights.get(index);
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
			this->unref_texture(m_envLights.get(index).get_envmap());
			m_envLights.erase(index);
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

void WorldContainer::set_light_name(u32 index, lights::LightType type, std::string_view name) {
	switch(type) {
		case lights::LightType::POINT_LIGHT: {
			m_pointLights.change_key(index, std::string(name));
		} break;
		case lights::LightType::SPOT_LIGHT: {
			m_spotLights.change_key(index, std::string(name));
		} break;
		case lights::LightType::DIRECTIONAL_LIGHT: {
			m_dirLights.change_key(index, std::string(name));
		} break;
		case lights::LightType::ENVMAP_LIGHT: {
			m_envLights.change_key(index, std::string(name));
		} break;
		default:
			throw std::runtime_error("[WorldContainer::get_light_name] Invalid light type.");
	}
}

void WorldContainer::mark_camera_dirty(ConstCameraHandle hdl) {
	// TODO: put the mark-dirty on scenario/scene level (one for add/remove, one for touched)?
	// We need one for the env-map for sure, because that operation takes long
	if(hdl == nullptr)
		return;
	m_camerasDirty[hdl] = true;
}

void WorldContainer::mark_light_dirty(u32 index, lights::LightType type) {
	if(m_scenario != nullptr) {
		switch(type) {
			case lights::LightType::POINT_LIGHT:
				// Check if the light is part of the active scenario/scene
				if(std::find(m_scenario->get_point_lights().cbegin(), m_scenario->get_point_lights().cend(), index)
				   != m_scenario->get_point_lights().cend())
					m_lightsDirty = true; // Doesn't matter what light, we need to rebuild the light tree
				break;
			case lights::LightType::SPOT_LIGHT:
				// Check if the light is part of the active scenario/scene
				if(std::find(m_scenario->get_spot_lights().cbegin(), m_scenario->get_spot_lights().cend(), index)
				   != m_scenario->get_spot_lights().cend())
					m_lightsDirty = true; // Doesn't matter what light, we need to rebuild the light tree
				break;
			case lights::LightType::DIRECTIONAL_LIGHT:
				// Check if the light is part of the active scenario/scene
				if(std::find(m_scenario->get_dir_lights().cbegin(), m_scenario->get_dir_lights().cend(), index)
				   != m_scenario->get_dir_lights().cend())
					m_lightsDirty = true; // Doesn't matter what light, we need to rebuild the light tree
				break;
			case lights::LightType::ENVMAP_LIGHT:
				// Check if the envmap is the current one
				const lights::Background& background = m_envLights.get(m_scenario->get_background());
				if(&background == &m_envLights.get(index))
					m_envLightDirty = true;
				break;
		}
	}
}

bool WorldContainer::has_texture(std::string_view name) const {
	return m_textures.find(name) != m_textures.cend();
}

TextureHandle WorldContainer::find_texture(std::string_view name) {
	auto iter = m_textures.find(name);
	if(iter != m_textures.end())
		return &iter->second;
	return nullptr;
}

std::optional<std::string_view> WorldContainer::get_texture_name(ConstTextureHandle hdl) const {
	// Gotta iterate entire map... very expensive operation
	// TODO: use bimap?
	for(auto iter = m_textures.cbegin(); iter != m_textures.cend(); ++iter) {
		if(&iter->second == hdl)
			return iter->first;
	}
	return std::nullopt;
}

TextureHandle WorldContainer::add_texture(std::string_view path, u16 width,
										  u16 height, u16 numLayers,
										  textures::Format format, textures::SamplingMode mode,
										  bool sRgb, std::unique_ptr<u8[]> data) {
	mAssertMsg(m_textures.find(path) == m_textures.end(), "Duplicate texture entry");
	// TODO: ensure that we have at least 1x1 pixels?
	TextureHandle texHdl = &m_textures.emplace(path, textures::Texture{ width, height, numLayers,
							  format, mode, sRgb, move(data) }).first->second;
	m_texRefCount[texHdl] = 1u;
	return texHdl;
}

void WorldContainer::ref_texture(TextureHandle hdl) {
	auto iter = m_texRefCount.find(hdl);
	if(iter != m_texRefCount.end())
		++iter->second;
}

void WorldContainer::unref_texture(TextureHandle hdl) {
	auto iter = m_texRefCount.find(hdl);
	if(iter != m_texRefCount.end()) {
		if(iter->second != 0u) {
			if(--iter->second == 0u) {
				// No more references, delete the texture
				// This unfortunately means linear search
				for(auto texIter = m_textures.begin(); texIter != m_textures.end(); ++texIter) {
					if(&texIter->second == hdl) {
						m_textures.erase(texIter);
						break;
					}
				}
				m_texRefCount.erase(iter);
			}
		}
	}
}

SceneHandle WorldContainer::load_scene(Scenario& scenario) {
	logInfo("[WorldContainer::load_scene] Loading scenario ", scenario.get_name());
	m_scenario = &scenario;
	m_scene = std::make_unique<Scene>(scenario);
	u32 instIdx = 0;
	// TODO: unload LoDs that are not needed anymore?
	for(auto& instance : m_instances) {
		Instance& inst = *instance.second;
		Object& obj = inst.get_object();
		if(!scenario.is_masked(&obj) && !scenario.is_masked(&inst)) {
			const u32 lod = scenario.get_effective_lod(&inst);
			if(!obj.has_lod_available(lod)) {
				if(!m_load_lod(&obj, lod))
					throw std::runtime_error("Failed to after-load LoD");
			}
			m_scene->add_instance(instance.second.get());
		}
	}

	// Check if the resulting scene has issues with size
	if(ei::len(m_scene->get_bounding_box().min) >= SUGGESTED_MAX_SCENE_SIZE
	   || ei::len(m_scene->get_bounding_box().max) >= SUGGESTED_MAX_SCENE_SIZE)
		logWarning("[WorldContainer::load_scene] Scene size is larger than recommended "
				   "(Furthest point of the bounding box should not be further than "
				   "2^20m away)");

	// Everything is dirty if we load a new scene
	m_lightsDirty = true;
	m_envLightDirty = true;

	// Load the lights
	this->load_scene_lights();
	// Make media available / resident
	m_scene->load_media(m_media);
	m_scene->set_camera(m_scenario->get_camera());

	m_scenario->camera_dirty_reset();

	// Assign the newly created scene and destroy the old one?
	return m_scene.get();
}

SceneHandle WorldContainer::load_scene(ScenarioHandle hdl) {
	mAssert(hdl != nullptr);
	return load_scene(*hdl);
}

bool WorldContainer::reload_scene() {
	// There is always a scenario set
	mAssert(m_scenario != nullptr);

	if(m_scene == nullptr) {
		(void) load_scene(*m_scenario);
		return true;
	}
	bool reloaded = this->load_scene_lights();

	// TODO: dirty flag for media?
	// TODO: dirty flag for materials?

	// TODO: re-enable dirty flag for camera, but also pay attention to modifications
	if(m_scenario->camera_dirty_reset() || m_camerasDirty[m_scenario->get_camera()]) {
		m_scene->set_camera(m_scenario->get_camera());
		m_camerasDirty[m_scenario->get_camera()] = false;
		reloaded = true;
	}
	return reloaded;
}

bool WorldContainer::load_scene_lights() {
	/* Possibilities:
	 * 1. Lights have been added/removed from the scenario -> rebuild light tree
	 * 2. Lights have been changed -> rebuild tree, but not envmap
	 * 3. Only envmap light has been changed -> only replace envmap
	 */

	bool reloaded = false;
	if(m_lightsDirty || m_scenario->lights_dirty_reset() || !m_scene->get_light_tree_builder().is_resident<Device::CPU>()) {
		reloaded = true;
		std::vector<lights::PositionalLights> posLights;
		std::vector<lights::DirectionalLight> dirLights;
		i32 instIdx = 0;
		for(auto& obj : m_scene->get_objects()) {
			// Object contains area lights.
			// Create one light source per polygone and instance
			for(auto& inst : obj.second) {
				mAssertMsg(obj.first->has_lod_available(m_scenario->get_effective_lod(inst)), "Instance references LoD that doesn't exist");
				Lod& lod = obj.first->get_lod(m_scenario->get_effective_lod(inst));
				if(!lod.is_emissive(*m_scenario)) {
					++instIdx;
					continue;
				}
				i32 primIdx = 0;
				// First search in polygons (PrimitiveHandle expects poly before sphere)
				auto& polygons = lod.get_geometry<geometry::Polygons>();
				const MaterialIndex* materials = polygons.acquire_const<Device::CPU, MaterialIndex>(polygons.get_material_indices_hdl());
				const scene::Point* positions = polygons.acquire_const<Device::CPU, scene::Point>(polygons.get_points_hdl());
				const scene::UvCoordinate* uvs = polygons.acquire_const<Device::CPU, scene::UvCoordinate>(polygons.get_uvs_hdl());
				for(const auto& face : polygons.faces()) {
					ConstMaterialHandle mat = m_scenario->get_assigned_material(materials[primIdx]);
					if(mat->get_properties().is_emissive()) {
						auto emission = mat->get_emission();
						mAssert(emission.texture != nullptr);
						if(std::distance(face.begin(), face.end()) == 3) {
							lights::AreaLightTriangleDesc al;
							al.radianceTex = emission.texture;
							al.scale = ei::packRGB9E5(emission.scale);
							int i = 0;
							for(auto vHdl : face) {
								al.points[i] = inst->get_transformation_matrix() * ei::Vec4{inst->get_scale() * positions[vHdl.idx()], 1.0f};
								al.uv[i] = uvs[vHdl.idx()];
								++i;
							}
							posLights.push_back(lights::PositionalLights{ al, { instIdx, primIdx } });
						} else {
							lights::AreaLightQuadDesc al;
							al.radianceTex = emission.texture;
							al.scale = ei::packRGB9E5(emission.scale);
							int i = 0;
							for(auto vHdl : face) {
								al.points[i] = inst->get_transformation_matrix() * ei::Vec4{inst->get_scale() * positions[vHdl.idx()], 1.0f};
								al.uv[i] = uvs[vHdl.idx()];
								++i;
							}
							posLights.push_back(lights::PositionalLights{ al, { instIdx, primIdx } });
						}
					}
					++primIdx;
				}

				// Then get the sphere lights
				primIdx = (u32)lod.get_geometry<geometry::Polygons>().get_face_count();
				auto& spheres = lod.get_geometry<geometry::Spheres>();
				materials = spheres.acquire_const<Device::CPU, MaterialIndex>(spheres.get_material_indices_hdl());
				const ei::Sphere* spheresData = spheres.acquire_const<Device::CPU, ei::Sphere>(spheres.get_spheres_hdl());
				for(std::size_t i = 0; i < spheres.get_sphere_count(); ++i) {
					ConstMaterialHandle mat = m_scenario->get_assigned_material(materials[i]);
					if(mat->get_properties().is_emissive()) {
						auto emission = mat->get_emission();
						mAssert(emission.texture != nullptr);
						lights::AreaLightSphereDesc al{
							inst->get_transformation_matrix() * ei::Vec4{inst->get_scale() * spheresData[i].center, 1.0f},
							inst->get_scale() * spheresData[i].radius,
							emission.texture, ei::packRGB9E5(emission.scale)
						};
						posLights.push_back({ al, PrimitiveHandle{instIdx, primIdx} });
					}
					++primIdx;
				}
				++instIdx;
			}
		}

		posLights.reserve(posLights.size() + m_pointLights.size() + m_spotLights.size());
		dirLights.reserve(dirLights.size() + m_dirLights.size());

		// Add regular lights
		for(u32 lightIndex : m_scenario->get_point_lights()) {
			mAssert(lightIndex < m_pointLights.size());
			posLights.push_back(lights::PositionalLights{ m_pointLights.get(lightIndex), PrimitiveHandle{} });
		}
		for(u32 lightIndex : m_scenario->get_spot_lights()) {
			mAssert(lightIndex < m_spotLights.size());
			posLights.push_back(lights::PositionalLights{ m_spotLights.get(lightIndex), PrimitiveHandle{} });
		}
		for(u32 lightIndex : m_scenario->get_dir_lights()) {
			mAssert(lightIndex < m_dirLights.size());
			dirLights.push_back(m_dirLights.get(lightIndex));
		}

		m_scene->set_lights(std::move(posLights), std::move(dirLights));
		m_lightsDirty = false;

		// Detect whether an (or THE envmap has been added/removed)
	}

	if(m_envLightDirty || m_scenario->envmap_lights_dirty_reset()) {
		reloaded = true;

		// Find out what the active envmap light is
		lights::Background& background = m_envLights.get(m_scenario->get_background());
		m_scene->set_background(background);

		m_envLightDirty = false;
	}

	m_scenario->lights_dirty_reset();
	m_scenario->envmap_lights_dirty_reset();
	return reloaded;
}

} // namespace mufflon::scene
