#include "world_container.hpp"
#include "util/log.hpp"
#include "core/cameras/camera.hpp"
#include "core/scene/lod.hpp"
#include "core/scene/materials/material.hpp"
#include "core/scene/materials/medium.hpp"
#include "core/scene/tessellation/cam_dist.hpp"
#include <iostream>
#include <ei/conversions.hpp>
#include <windows.h>

namespace mufflon::scene {

WorldContainer WorldContainer::s_container{};

WorldContainer::WorldContainer()
{
	m_envLights.insert("##DefaultBlack##", lights::Background::black());
}

WorldContainer& WorldContainer::instance() {
	return s_container;
}

void WorldContainer::clear_instance() {
	s_container = WorldContainer();
}

bool WorldContainer::set_frame_current(const u32 frameCurrent) {
	const u32 newFrame = std::min(m_frameEnd, std::max(m_frameStart, frameCurrent));
	if(newFrame != m_frameCurrent) {
		m_frameCurrent = newFrame;
		// Delete the current scene to make it clear to everyone that it needs to be refetched
		m_sceneValid = false;
		return true;
	}
	return false;
}

WorldContainer::Sanity WorldContainer::finalize_world() const {
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


WorldContainer::Sanity WorldContainer::finalize_scenario(ConstScenarioHandle hdl) {
	bool hasEmitters = false;
	bool hasObjects = false;

	// First update the flags for each LoD
	std::unordered_set<MaterialIndex> uniqueMatIndices;
	uniqueMatIndices.reserve(m_materials.size());

	for(auto& obj : m_objects) {
		for(u32 l = 0u; l < static_cast<u32>(obj.second.get_lod_slot_count()); ++l) {
			if(!obj.second.has_lod_available(l))
				continue;
			auto& lod = obj.second.get_lod(l);
			lod.update_flags(*hdl, uniqueMatIndices);
		}
	}

	// Check for objects (and check for emitters as well)
	// Gotta check both animated and stationary instances
	for(const auto& instance : m_instances) {
		if(hdl->is_masked(&instance))
			continue;

		hasObjects = true;
		const Object& object = instance.get_object();
		const u32 lodLevel = hdl->get_effective_lod(&instance);
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

	if(!hasObjects)
		return Sanity::NO_OBJECTS;
	// Check for lights
	if(!hasEmitters && hdl->get_point_lights().empty() && hdl->get_spot_lights().empty()
	   && hdl->get_dir_lights().empty() && (m_envLights.get(hdl->get_background()).get_type() == lights::BackgroundType::COLORED
											&& ei::prod(m_envLights.get(hdl->get_background()).get_monochrom_color()) == 0))
		return Sanity::NO_LIGHTS;
	// Check for camera
	if(hdl->get_camera() == nullptr)
		return Sanity::NO_CAMERA;
	return Sanity::SANE;
}

void WorldContainer::reserve(const u32 objects, const u32 instances) {
	if(m_objects.size() > 0u || m_instances.size() > 0u)
		throw std::runtime_error("This method may only be called on a clean world state!");
	m_objects = util::FixedHashMap<StringView, Object>{ objects };
	m_instances.reserve(instances);
	m_instanceTransformations.reserve(instances);
	// Set the name pool size (only objects have names anyway)
	if(!m_namePool.empty())
		throw std::runtime_error("Trying to set the size for a non-empty name pool; "
								 "this invalidates all already stored names and shouldn't happen!");
	// Use a heuristic based on the object count (assumed name length ~7)
	const auto pages = 1u + 7lu * objects / util::StringPool::PAGE_SIZE;
	m_namePool = util::StringPool{ pages };
}

void WorldContainer::reserve(const u32 scenarios) {
	if(m_scenarios.size() > 0u)
		throw std::runtime_error("This method may only be called on a clean scenario state!");
	if(scenarios > 32u)
		throw std::runtime_error("Too many scenarios; we are currently limited to 32 at a time");
	m_scenarios = util::FixedHashMap<StringView, Scenario>{ scenarios };
}

ObjectHandle WorldContainer::create_object(const StringView name, ObjectFlags flags) {
	const auto pooledName = m_namePool.insert(name);
	if(m_objects.find(pooledName) != m_objects.cend())
		throw std::runtime_error("Object with the name already exists!");
	auto& hdl = m_objects.emplace(pooledName, Object{static_cast<u32>(m_objects.size())});
	hdl.set_name(pooledName);
	hdl.set_flags(flags);
	return &hdl;
}

ObjectHandle WorldContainer::get_object(const StringView name) {
	auto iter = m_objects.find(name);
	if(iter != m_objects.end())
		return &iter->second;
	return nullptr;
}

ObjectHandle WorldContainer::duplicate_object(ObjectHandle hdl, const StringView newName) {
	const auto pooledName = m_namePool.insert(newName);
	ObjectHandle newHdl = create_object(pooledName, hdl->get_flags());
	newHdl->copy_lods_from(*hdl);
	return newHdl;
}

void WorldContainer::apply_transformation(InstanceHandle hdl) {
	const ei::Mat3x4& transMat = get_instance_transformation(hdl);
	ObjectHandle objectHandle = &hdl->get_object();
	if(objectHandle->get_instance_counter() > 1) {
		static thread_local std::string newName{};
		newName.clear();
		newName.append(objectHandle->get_name().data());
		newName.append("###TRANSFORMED_INSTANCE");
		newName.append(std::to_string(objectHandle->get_instance_counter()));
		objectHandle = duplicate_object(objectHandle, newName);
		hdl->set_object(*objectHandle);
	}
	for(size_t i = 0; i < objectHandle->get_lod_slot_count(); i++) {
		if(objectHandle->has_lod_available(u32(i))) {
			Lod& lod = objectHandle->get_lod(u32(i));
			auto& polygons = lod.get_geometry<geometry::Polygons>();
			polygons.transform(transMat);
			auto& spheres = lod.get_geometry<geometry::Spheres>();
			spheres.transform(transMat);
		}
	}
	set_instance_transformation(hdl, ei::Mat3x4{ ei::identity4x4() });
}

InstanceHandle WorldContainer::create_instance(ObjectHandle obj, const u32 animationFrame) {
	if(obj == nullptr) {
		logError("[WorldContainer::create_instance] Invalid object handle");
		return nullptr;
	}
	if(m_instances.capacity() <= m_instances.size())
		throw std::runtime_error("Instance created outside of reserved bounds; since this "
								 "may lead to invalid instance handles, this is disallowed. "
								 "Reserve the right number of instances beforehand");

	if(animationFrame == Instance::NO_ANIMATION_FRAME) {
		if(!m_frameInstanceIndices.empty())
			throw std::runtime_error("Non-animated instances must be added before animated ones!");
	} else {
		// Check if it's the first instance ever and set start+end frames accordingly
		if(m_frameInstanceIndices.size() == 0u) {
			m_frameCurrent = animationFrame;
			m_frameStart = animationFrame;
			m_frameEnd = animationFrame;
		} else {
			m_frameStart = std::min(m_frameStart, animationFrame);
			m_frameEnd = std::max(m_frameEnd, animationFrame);
		}
		const auto index = animationFrame - m_frameStart;
		// Check for out-of-order insert
		if(m_frameInstanceIndices.size() == index) {
			// New frame added
			m_frameInstanceIndices.emplace_back(static_cast<u32>(m_instances.size()), 1u);
		} else if(m_frameInstanceIndices.size() == (index + 1u)) {
			// Additional instance for current frame
			++m_frameInstanceIndices.back().second;
		} else {
			throw std::runtime_error("Animated instances must be added in order of their frame");
		}
	}

	m_instanceTransformations.push_back(ei::Mat3x4{ ei::identity4x4() });
	return &m_instances.emplace_back(*obj, static_cast<u32>(m_instances.size()));
}

InstanceHandle WorldContainer::get_instance(std::size_t index, const u32 animationFrame) {
	if(animationFrame == Instance::NO_ANIMATION_FRAME) {
		// TODO: use index string map?
		auto iter = m_instances.begin();
		for(std::size_t i = 0; iter != m_instances.end() && i < index; ++i)
			++iter;
		if(iter != m_instances.end())
			return &*iter;
	} else if(animationFrame < m_frameInstanceIndices.size()) {
		if(m_frameInstanceIndices[animationFrame].second > index)
			return &m_instances[m_frameInstanceIndices[animationFrame].first + index];
	}
	return nullptr;
}

const ei::Mat3x4& WorldContainer::get_instance_transformation(ConstInstanceHandle instance) const {
	return m_instanceTransformations[instance->get_index()];
}

void WorldContainer::set_instance_transformation(ConstInstanceHandle instance, const ei::Mat3x4& mat) {
	m_instanceTransformations[instance->get_index()] = mat;
}

std::size_t WorldContainer::get_highest_instance_frame() const noexcept { 
	return m_frameInstanceIndices.size();
}

std::size_t WorldContainer::get_instance_count(const u32 frame) const noexcept {
	if(frame == Instance::NO_ANIMATION_FRAME)
		return m_instances.size();
	else if(frame >= m_frameInstanceIndices.size())
		return 0u;
	else
		return m_frameInstanceIndices[frame].second;
};

ScenarioHandle WorldContainer::create_scenario(const StringView name) {
	if(m_scenarios.size() >= 32u)
		throw std::runtime_error("Too many scenarios already exist - limit is 32 (for now)");
	const auto pooledName = m_namePool.insert(name);
	// TODO: switch to pointer
	auto& hdl = m_scenarios.emplace(pooledName, Scenario{ static_cast<u32>(m_scenarios.size()), m_namePool });
	hdl.set_name(pooledName);
	return &hdl;
}

ScenarioHandle WorldContainer::get_scenario(const StringView name) {
	auto iter = m_scenarios.find(name);
	if(iter != m_scenarios.end())
		return &iter->second;
	return nullptr;
}

ScenarioHandle WorldContainer::get_current_scenario() const noexcept {
	return m_scenario;
}

SceneHandle WorldContainer::get_current_scene() {
	return m_scene.get();
}

bool WorldContainer::is_current_scene_valid() const noexcept {
	return m_sceneValid && m_scene != nullptr;
}

ScenarioHandle WorldContainer::get_scenario(std::size_t index) {
	mAssert(index < m_scenarios.size());
	// TODO: use index string map?
	auto iter = m_scenarios.begin();
	for(std::size_t i = 0; i < index; ++i)
		++iter;
	return &iter->second;
}

std::size_t WorldContainer::get_scenario_count() const noexcept {
	return m_scenarios.size();
}

MaterialHandle WorldContainer::add_material(std::unique_ptr<materials::IMaterial> material) {
	m_materials.push_back(move(material));
	return m_materials.back().get();
}

std::size_t WorldContainer::get_material_count() const noexcept {
	return m_materials.size();
}

MaterialHandle WorldContainer::get_material(u32 index) {
	return m_materials.at(index).get();
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

const materials::Medium& WorldContainer::get_medium(materials::MediumHandle hdl) const {
	return m_media.at(hdl);
}

CameraHandle WorldContainer::add_camera(const StringView name, std::unique_ptr<cameras::Camera> camera) {
	const auto pooledName = m_namePool.insert(name);
	auto iter = m_cameras.emplace(pooledName, move(camera));
	if(!iter.second)
		return nullptr;
	iter.first->second->set_name(pooledName);
	m_cameraHandles.push_back(iter.first);
	m_frameEnd = std::max(m_frameEnd, m_frameStart + iter.first->second->get_path_segment_count() - 1u);
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

CameraHandle WorldContainer::get_camera(StringView name) {
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

std::optional<u32> WorldContainer::add_light(std::string name, const lights::PointLight& light,
											 const u32 count) {
	if(m_pointLights.find(name) != nullptr) {
		logError("[WorldContainer::add_light] Point light with name '", name, "' already exists");
		return std::nullopt;
	}
	if(m_frameStart + count > m_frameEnd + 1u)
		m_frameEnd = m_frameStart + count - 1u;
	return m_pointLights.insert(std::move(name), std::vector<lights::PointLight>(count, light));
}

std::optional<u32> WorldContainer::add_light(std::string name,
											 const lights::SpotLight& light,
											 const u32 count) {
	if(m_spotLights.find(name) != nullptr) {
		logError("[WorldContainer::add_light] Spot light with name '", name, "' already exists");
		return std::nullopt;
	}
	if(m_frameStart + count > m_frameEnd + 1u)
		m_frameEnd = m_frameStart + count - 1u;
	return m_spotLights.insert(std::move(name), std::vector<lights::SpotLight>(count, light));

}

std::optional<u32> WorldContainer::add_light(std::string name,
											 const lights::DirectionalLight& light,
											 const u32 count) {
	if(m_dirLights.find(name) != nullptr) {
		logError("[WorldContainer::add_light] Directional light with name '", name, "' already exists");
		return std::nullopt;
	}
	if(m_frameStart + count > m_frameEnd + 1u)
		m_frameEnd = m_frameStart + count - 1u;
	return m_dirLights.insert(std::move(name), std::vector<lights::DirectionalLight>(count, light));
}

std::optional<u32> WorldContainer::add_light(std::string name,
											 lights::BackgroundType type) {
	if(m_envLights.find(name) != nullptr) {
		logError("[WorldContainer::add_light] Background light with name '", name, "' already exists");
		return std::nullopt;
	}

	return m_envLights.insert(std::move(name), lights::Background{ type });
}

void WorldContainer::replace_envlight_texture(u32 index, TextureHandle replacement) {
	if(index >= m_envLights.size())
		throw std::runtime_error("Envmap light index out of bounds (" + std::to_string(index)
								 + " >= " + std::to_string(m_envLights.size()) + ")");
	if(replacement) mAssertMsg(replacement->get_sampling_mode() == textures::SamplingMode::NEAREST,
		"Sampling mode must be nearest, otherwise the importance sampling is biased.");
	lights::Background& desc = m_envLights.get(index);
	if(desc.get_type() != lights::BackgroundType::ENVMAP)
		throw std::runtime_error("Background light is not of type 'envmap'");
	if(replacement != desc.get_envmap()) {
		this->unref_texture(desc.get_envmap());
		desc = lights::Background::envmap(replacement);
	}
}

std::optional<std::pair<u32, lights::LightType>> WorldContainer::find_light(const StringView& name) {
	if(m_pointLights.find(name) != nullptr)
		return std::make_pair(u32(m_pointLights.get_index(name)), lights::LightType::POINT_LIGHT);
	if(m_spotLights.find(name) != nullptr)
		return std::make_pair(u32(m_spotLights.get_index(name)), lights::LightType::SPOT_LIGHT);
	if(m_dirLights.find(name) != nullptr)
		return std::make_pair(u32(m_dirLights.get_index(name)), lights::LightType::DIRECTIONAL_LIGHT);
	if(const auto& light = m_envLights.find(name); light != nullptr) {
		return std::make_pair(u32(m_envLights.get_index(name)), lights::LightType::ENVMAP_LIGHT);
	}
	return std::nullopt;
}

std::size_t WorldContainer::get_point_light_segment_count(u32 index) {
	if(index >= m_pointLights.size())
		throw std::runtime_error("Point light index out of bounds (" + std::to_string(index)
								 + " >= " + std::to_string(m_pointLights.size()));
	return m_pointLights.get(index).size();
}

std::size_t WorldContainer::get_spot_light_segment_count(u32 index) {
	if(index >= m_spotLights.size())
		throw std::runtime_error("Spot light index out of bounds (" + std::to_string(index)
								 + " >= " + std::to_string(m_spotLights.size()));
	return m_spotLights.get(index).size();
}

std::size_t WorldContainer::get_dir_light_segment_count(u32 index) {
	if(index >= m_dirLights.size())
		throw std::runtime_error("Directional light index out of bounds (" + std::to_string(index)
								 + " >= " + std::to_string(m_dirLights.size()));
	return m_dirLights.get(index).size();
}

lights::PointLight* WorldContainer::get_point_light(u32 index, const u32 frame) {
	if(index >= m_pointLights.size())
		throw std::runtime_error("Point light index out of bounds (" + std::to_string(index)
								 + " >= " + std::to_string(m_pointLights.size()));
	auto& animLight = m_pointLights.get(index);
	return &animLight[std::min(frame, static_cast<u32>(animLight.size()) - 1u)];
}

lights::SpotLight* WorldContainer::get_spot_light(u32 index, const u32 frame) {
	if(index >= m_spotLights.size())
		throw std::runtime_error("Spot light index out of bounds (" + std::to_string(index)
								 + " >= " + std::to_string(m_spotLights.size()));
	auto& animLight = m_spotLights.get(index);
	return &animLight[std::min(frame, static_cast<u32>(animLight.size()) - 1u)];
}

lights::DirectionalLight* WorldContainer::get_dir_light(u32 index, const u32 frame) {
	if(index >= m_dirLights.size())
		throw std::runtime_error("Directional light index out of bounds (" + std::to_string(index)
									+ " >= " + std::to_string(m_dirLights.size()));
	auto& animLight = m_dirLights.get(index);
	return &animLight[std::min(frame, static_cast<u32>(animLight.size()) - 1u)];
}

lights::Background* WorldContainer::get_background(u32 index) {
	if(index >= m_envLights.size())
		throw std::runtime_error("Envmap light index out of bounds (" + std::to_string(index)
									+ " >= " + std::to_string(m_envLights.size()));
	return &m_envLights.get(index);
}

lights::Background& WorldContainer::get_default_background() {
	static lights::Background defaultBackground = lights::Background::black();
	return defaultBackground;
}

void WorldContainer::remove_light(u32 index, lights::LightType type) {
	switch(type) {
		case lights::LightType::POINT_LIGHT:
		{
			if(index >= m_pointLights.size())
				throw std::runtime_error("Point light index out of bounds (" + std::to_string(index)
											+ " >= " + std::to_string(m_pointLights.size()));
			m_pointLights.erase(index);
			for(auto& scenario : m_scenarios)
				scenario.second.remove_point_light(index);
		} break;
		case lights::LightType::SPOT_LIGHT:
		{
			if(index >= m_spotLights.size())
				throw std::runtime_error("Spot light index out of bounds (" + std::to_string(index)
											+ " >= " + std::to_string(m_spotLights.size()));
			m_spotLights.erase(index);
			for(auto& scenario : m_scenarios)
				scenario.second.remove_spot_light(index);
		} break;
		case lights::LightType::DIRECTIONAL_LIGHT:
		{
			if(index >= m_dirLights.size())
				throw std::runtime_error("Directional light index out of bounds (" + std::to_string(index)
											+ " >= " + std::to_string(m_dirLights.size()));
			m_dirLights.erase(index);
			for(auto& scenario : m_scenarios)
				scenario.second.remove_dir_light(index);
		} break;
		case lights::LightType::ENVMAP_LIGHT:
		{
			if(index >= m_envLights.size())
				throw std::runtime_error("Envmap light index out of bounds (" + std::to_string(index)
											+ " >= " + std::to_string(m_envLights.size()));
			this->unref_texture(m_envLights.get(index).get_envmap());
			m_envLights.erase(index);
			for(auto& scenario : m_scenarios) {
				if(scenario.second.get_background() == index)
					scenario.second.remove_background();
			}
		} break;
		default:
			throw std::runtime_error("[WorldContainer::remove_light] Invalid light type.");
	}
}

StringView WorldContainer::get_light_name(u32 index, lights::LightType type) const {
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

void WorldContainer::set_light_name(u32 index, lights::LightType type, StringView name) {
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

bool WorldContainer::mark_light_dirty(u32 index, lights::LightType type) {
	if(m_scenario != nullptr && m_sceneValid) {
		switch(type) {
			case lights::LightType::POINT_LIGHT:
				// Check if the light is part of the active scenario/scene
				if(m_scene != nullptr && std::find(m_scenario->get_point_lights().cbegin(), m_scenario->get_point_lights().cend(), index)
				   != m_scenario->get_point_lights().cend()) {
					m_scene->mark_lights_dirty(); // Doesn't matter what light, we need to rebuild the light tree
					return true;
				}
				break;
			case lights::LightType::SPOT_LIGHT:
				// Check if the light is part of the active scenario/scene
				if(std::find(m_scenario->get_spot_lights().cbegin(), m_scenario->get_spot_lights().cend(), index)
				   != m_scenario->get_spot_lights().cend()) {
					m_scene->mark_lights_dirty(); // Doesn't matter what light, we need to rebuild the light tree
					return true;
				}
				break;
			case lights::LightType::DIRECTIONAL_LIGHT:
				// Check if the light is part of the active scenario/scene
				if(std::find(m_scenario->get_dir_lights().cbegin(), m_scenario->get_dir_lights().cend(), index)
				   != m_scenario->get_dir_lights().cend()) {
					m_scene->mark_lights_dirty(); // Doesn't matter what light, we need to rebuild the light tree
					return true;
				}
				break;
			case lights::LightType::ENVMAP_LIGHT:
				if(index >= m_envLights.size()) {
					logError("[WorldContainer::mark_light_dirty]: Background light index out of bounds");
					return false;
				}
				// We have only one background light
				m_scene->set_background(m_envLights.get(index));
				return true;
			default:
				logWarning("[WorldContainer::mark_light_dirty]: Ignoring unknown light type");
		}
	}
	return false;
}

u32 WorldContainer::get_frame_start() const noexcept {
	return m_frameStart;
}
u32 WorldContainer::get_frame_end() const noexcept {
	return m_frameEnd;
}
u32 WorldContainer::get_frame_current() const noexcept {
	return m_frameCurrent;
}

bool WorldContainer::has_texture(StringView name) const {
	return m_textures.find(name) != m_textures.cend();
}

TextureHandle WorldContainer::find_texture(StringView name) {
	auto iter = m_textures.find(name);
	if(iter != m_textures.end())
		return iter->second.get();
	return nullptr;
}

TextureHandle WorldContainer::add_texture(std::unique_ptr<textures::Texture> texture) {
	// TODO: ensure that we have at least 1x1 pixels?
	StringView nameRef = texture->get_name();
	mAssertMsg(m_textures.find(nameRef) == m_textures.end(), "Duplicate texture entry");
	TextureHandle texHdl = texture.get();
	m_textures.emplace(nameRef, move(texture));
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
					if(texIter->second.get() == hdl) {
						m_textures.erase(texIter);
						break;
					}
				}
				m_texRefCount.erase(iter);
			}
		}
	}
}

bool WorldContainer::load_lod(Object& obj, const u32 lodIndex) {
	if(!obj.has_lod_available(lodIndex)) {
		if(!m_load_lod(&obj, lodIndex))
			return false;
		// Update the flags for this LoD
		std::unordered_set<MaterialIndex> uniqueMatIndices;
		uniqueMatIndices.reserve(m_materials.size());
		for(const auto& scenario : m_scenarios)
			obj.get_lod(lodIndex).update_flags(scenario.second, uniqueMatIndices);
	}
	return true;
}

bool WorldContainer::unload_lod(Object& obj, const u32 lodIndex) {
	if(obj.has_lod_available(lodIndex))
		obj.remove_lod(lodIndex);
	return true;
}

SceneHandle WorldContainer::load_scene(Scenario& scenario, renderer::IRenderer* renderer) {
	logInfo("[WorldContainer::load_scene] Loading scenario ", scenario.get_name());
	if(renderer != nullptr)
		renderer->on_scenario_changing();
	m_scenario = &scenario;

	const auto frameIndex = m_frameCurrent - m_frameStart;
	const bool hasAnimatedInsts = m_frameInstanceIndices.size() > frameIndex;
	const u32 animatedInstCount = hasAnimatedInsts ? m_frameInstanceIndices[frameIndex].second : 0u;
	const u32 animatedInstOffset = hasAnimatedInsts ? m_frameInstanceIndices[frameIndex].first : 0u;
	const std::size_t regInstCount = m_frameInstanceIndices.empty() ? m_instances.size() : m_frameInstanceIndices.front().first;

	// We need a dictionary of all instances for a given object participating in the scene
	util::FixedHashMap<ObjectHandle, Scene::InstanceRef> objInstRef{ m_objects.size() };

	// First add all participating objects and count their actual instances
	for(std::size_t i = 0u; i < regInstCount; ++i) {
		// Discard masked instances and objects
		if(m_scenario->is_masked(m_instances.data() + i))
			continue;
		auto& obj = m_instances[i].get_object();
		if(m_scenario->is_masked(&obj))
			continue;

		if(const auto iter = objInstRef.find(&obj); iter == objInstRef.cend())
			objInstRef.emplace(&obj, Scene::InstanceRef{ 0u, 1u });
		else
			++(iter->second.count);
	}
	for(std::size_t i = animatedInstOffset; i < animatedInstCount; ++i) {
		// Discard masked instances and objects
		if(m_scenario->is_masked(m_instances.data() + i))
			continue;
		auto& obj = m_instances[i].get_object();
		if(m_scenario->is_masked(&obj))
			continue;

		if(const auto iter = objInstRef.find(&obj); iter == objInstRef.cend())
			objInstRef.emplace(&obj, Scene::InstanceRef{ 0u, 1u });
		else
			++(iter->second.count);
	}

	std::vector<InstanceHandle> instanceHandles;
	{	// Now we can adjust the offsets for the instances
		u32 currOffset = 0u;
		for(auto& ref : objInstRef) {
			ref.second.offset = currOffset;
			currOffset += ref.second.count;
			ref.second.count = 0u;
		}
		if(currOffset == 0u)
			throw std::runtime_error("A scene needs at least one instance!");
		instanceHandles.resize(currOffset);
	}

	// Next we can build the vector of instances and compute the bounding box
	ei::Box aabb{};
	aabb.min = ei::Vec3{ std::numeric_limits<float>::max() };
	aabb.max = ei::Vec3{ -std::numeric_limits<float>::max() };
	for(std::size_t i = 0u; i < regInstCount; ++i) {
		auto& inst = m_instances[i];
		// Discard masked instances and objects
		if(m_scenario->is_masked(&inst))
			continue;
		auto& obj = inst.get_object();
		if(m_scenario->is_masked(&obj))
			continue;
		// Write the instance handles to the proper (sorted by object) place
		const auto iter = objInstRef.find(&obj);
		const auto index = iter->second.offset + iter->second.count;
		instanceHandles[index] = &inst;
		++iter->second.count;
		aabb = ei::Box{ aabb, inst.get_bounding_box(m_scenario->get_effective_lod(&inst),
													get_instance_transformation(&inst)) };
	}
	for(std::size_t i = animatedInstOffset; i < animatedInstCount; ++i) {
		auto& inst = m_instances[i];
		// Discard masked instances and objects
		if(m_scenario->is_masked(&inst))
			continue;
		auto& obj = inst.get_object();
		if(m_scenario->is_masked(&obj))
			continue;
		// Write the instance handles to the proper (sorted by object) place
		const auto iter = objInstRef.find(&obj);
		const auto index = iter->second.offset + iter->second.count;
		instanceHandles[index] = &inst;
		++iter->second.count;
		aabb = ei::Box{ aabb, inst.get_bounding_box(m_scenario->get_effective_lod(&inst),
													get_instance_transformation(&inst)) };
	}

	m_scene = std::make_unique<Scene>(scenario, m_frameCurrent - m_frameStart,
									  std::move(objInstRef), std::move(instanceHandles),
									  m_instanceTransformations, aabb);

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
	m_scene->set_camera(m_scenario->get_camera());

	m_scenario->camera_dirty_reset();

	if(renderer != nullptr)
		renderer->on_scenario_changed(&scenario);

	// Assign the newly created scene and destroy the old one?
	m_sceneValid = true;
	return m_scene.get();
}

SceneHandle WorldContainer::load_scene(ScenarioHandle hdl, renderer::IRenderer* renderer) {
	mAssert(hdl != nullptr);
	return load_scene(*hdl, renderer);
}

void WorldContainer::reload_scene(renderer::IRenderer* renderer) {
	// There is always a scenario set
	mAssert(m_scenario != nullptr);

	// TODO: better differentiate what needs to be reloaded on animation change
	if(m_scene == nullptr || !m_sceneValid) {
		(void) load_scene(*m_scenario, renderer);
		return;
	}

	// TODO: dirty flag for media?
	// TODO: dirty flag for materials?

	// TODO: re-enable dirty flag for camera, but also pay attention to modifications
	if(m_scenario->camera_dirty_reset() || m_scenario->get_camera()->is_dirty()) {
		renderer->on_camera_changing();
		m_scene->set_camera(m_scenario->get_camera());
		m_scenario->get_camera()->mark_clean();
		renderer->on_camera_changed();
	}
	
	this->load_scene_lights();
}

bool WorldContainer::load_scene_lights() {
	/* Possibilities:
	 * 1. Lights have been added/removed from the scenario -> rebuild light tree
	 * 2. Lights have been changed -> rebuild tree, but not envmap
	 * 3. Only envmap light has been changed -> only replace envmap
	 */

	bool reloaded = false;

	// This needs to come first since setting the lights rebuilds the tree and marks the envmap as non-dirty
	if(m_scene->get_light_tree_builder().is_background_dirty()
	   || m_scenario->envmap_lights_dirty_reset()) {
		reloaded = true;

		// Find out what the active envmap light is
		lights::Background& background = m_envLights.get(m_scenario->get_background());
		m_scene->set_background(background);
	}

	if(!m_scene->get_light_tree_builder().is_resident<Device::CPU>() || m_scenario->lights_dirty_reset()
	   || !m_scene->get_light_tree_builder().is_resident<Device::CPU>()) {
		reloaded = true;
		std::vector<lights::PositionalLights> posLights;
		std::vector<lights::DirectionalLight> dirLights;
		for(auto& obj : m_scene->get_objects()) {
			// Object contains area lights.
			// Create one light source per polygone and instance
			const auto& instances = m_scene->get_instances();
			const auto endIndex = obj.second.offset + obj.second.count;
			for(std::size_t idx = obj.second.offset; idx < endIndex; ++idx) {
				InstanceHandle inst = instances[idx];
				mAssertMsg(obj.first->has_lod_available(m_scenario->get_effective_lod(inst)), "Instance references LoD that doesn't exist");
				Lod& lod = obj.first->get_lod(m_scenario->get_effective_lod(inst));
				if(!lod.is_emissive(*m_scenario))
					continue;

				i32 primIdx = 0;
				// First search in polygons (PrimitiveHandle expects poly before sphere)
				auto& polygons = lod.get_geometry<geometry::Polygons>();
				const MaterialIndex* materials = polygons.acquire_const<Device::CPU, MaterialIndex>(polygons.get_material_indices_hdl());
				const scene::Point* positions = polygons.acquire_const<Device::CPU, scene::Point>(polygons.get_points_hdl());
				const scene::UvCoordinate* uvs = polygons.acquire_const<Device::CPU, scene::UvCoordinate>(polygons.get_uvs_hdl());
				const ei::Mat3x4& instanceTransformation = get_instance_transformation(inst);
				bool isMirroring = determinant(ei::Mat3x3{instanceTransformation}) < 0.0f;
				for(const auto& face : polygons.faces()) {
					ConstMaterialHandle mat = m_scenario->get_assigned_material(materials[primIdx]);
					if(mat->get_properties().is_emissive()) {
						if(std::distance(face.begin(), face.end()) == 3) {
							lights::AreaLightTriangleDesc al;
							al.material = materials[primIdx];
							int i = 0;
							for(auto vHdl : face) {
								al.points[i] = ei::transform(positions[vHdl.idx()], instanceTransformation);
								al.uv[i] = uvs[vHdl.idx()];
								++i;
							}
							if(isMirroring) {
								std::swap(al.points[1], al.points[2]);
								std::swap(al.uv[1], al.uv[2]);
							}
							posLights.push_back(lights::PositionalLights{ al, { (i32)inst->get_index(), primIdx } });
						} else {
							lights::AreaLightQuadDesc al;
							al.material = materials[primIdx];
							int i = 0;
							for(auto vHdl : face) {
								al.points[i] = ei::transform(positions[vHdl.idx()], instanceTransformation);
								al.uv[i] = uvs[vHdl.idx()];
								++i;
							}
							if(isMirroring) {
								std::swap(al.points[1], al.points[3]);
								std::swap(al.uv[1], al.uv[3]);
							}
							posLights.push_back(lights::PositionalLights{ al, { (i32)inst->get_index(), primIdx } });
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
						const auto scale = Instance::extract_scale(instanceTransformation);
						mAssert(ei::approx(scale.x, scale.y) && ei::approx(scale.x, scale.z));
						lights::AreaLightSphereDesc al{
							transform(spheresData[i].center, instanceTransformation),
							scale.x * spheresData[i].radius,
							materials[i]
						};
						posLights.push_back({ al, PrimitiveHandle{ (i32)inst->get_index(), primIdx } });
					}
					++primIdx;
				}
			}
		}

		posLights.reserve(posLights.size() + m_pointLights.size() + m_spotLights.size());
		dirLights.reserve(dirLights.size() + m_dirLights.size());

		// Add regular lights
		for(u32 lightIndex : m_scenario->get_point_lights()) {
			mAssert(lightIndex < m_pointLights.size());
			const auto& light = m_pointLights.get(lightIndex);
			posLights.push_back(lights::PositionalLights{
				light[std::min(m_frameCurrent - m_frameStart, static_cast<u32>(light.size()) - 1u)],
				PrimitiveHandle{}
			});
		}
		for(u32 lightIndex : m_scenario->get_spot_lights()) {
			mAssert(lightIndex < m_spotLights.size());
			const auto& light = m_spotLights.get(lightIndex);
			posLights.push_back(lights::PositionalLights{
				light[std::min(m_frameCurrent - m_frameStart, static_cast<u32>(light.size()) - 1u)],
				PrimitiveHandle{}
			});
		}
		for(u32 lightIndex : m_scenario->get_dir_lights()) {
			mAssert(lightIndex < m_dirLights.size());
			const auto& light = m_dirLights.get(lightIndex);
			dirLights.push_back(light[std::min(m_frameCurrent - m_frameStart, static_cast<u32>(light.size()) - 1u)]);
		}

		m_scene->set_lights(std::move(posLights), std::move(dirLights));
		// Detect whether an (or THE envmap has been added/removed)
	}

	m_scenario->lights_dirty_reset();
	m_scenario->envmap_lights_dirty_reset();
	return reloaded;
}

void WorldContainer::set_lod_loader_function(bool (CDECL*func)(ObjectHandle, u32)) {
	m_load_lod = func;
}

void WorldContainer::retessellate() {
	// Basically we need to find out which LoDs are part of the scene, re-tessellate them, and then
	// rebuild the light tree
	if(m_scene == nullptr || m_scenario == nullptr || !m_sceneValid)
		return;

	if(m_scene->retessellate(m_tessLevel)) {
		// Gotta rebuild the light tree
		m_scene->mark_lights_dirty();
		(void)this->load_scene_lights();
	}
}

} // namespace mufflon::scene
