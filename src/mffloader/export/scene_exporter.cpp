#include "scene_exporter.hpp"
#include "rapidjson/stringbuffer.h"
#include "rapidjson/prettywriter.h"

#include "core/export/interface.h"

#include <fstream> 
#include "util/degrad.hpp"

namespace mff_loader::exprt {

bool SceneExporter::save_scene() const
{
	// TODO Open old Json

	rapidjson::Document document;
	document.SetObject();

	document.AddMember("version", FILE_VERSION, document.GetAllocator());
	document.AddMember("binary", store_in_string_relative_to_destination_path(m_mffPath, document), document.GetAllocator());

	// JSON
	if(!save_cameras(document))
		return false;
	if (!save_lights(document))
		return false;
	if (!save_materials(document))
		return false;
	if (!save_scenarios(document))
		return false;

	rapidjson::StringBuffer strbuf;
	rapidjson::PrettyWriter<rapidjson::StringBuffer> writer(strbuf);
	document.Accept(writer);

	std::string json = std::string(strbuf.GetString());
	std::ofstream ofs(m_fileDestinationPath);

	ofs << json;

	ofs.close();


	// Binary


	return true;
}

bool SceneExporter::save_cameras(rapidjson::Document& document) const
{
	rapidjson::Value cameras;
	cameras.SetObject();
	size_t cameraCount = world_get_camera_count();

	for (size_t i = 0; i < cameraCount; i++)
	{
		rapidjson::Value camera;
		camera.SetObject();
		CameraHdl cameraHandle = world_get_camera_by_index(i);
		CameraType cameraType = world_get_camera_type(cameraHandle);

		switch (cameraType)
		{
		case CAM_PINHOLE:
			camera.AddMember("type", "pinhole", document.GetAllocator());
			float vFov;
			world_get_pinhole_camera_fov(cameraHandle, &vFov);
			vFov = static_cast<float>(mufflon::Degrees(mufflon::Radians(vFov))); // Convert Radian to Degree
			camera.AddMember("fov", vFov, document.GetAllocator());
			break;
		case CAM_FOCUS:
			camera.AddMember("type", "focus", document.GetAllocator());
			float focalLength;
			world_get_focus_camera_focal_length(cameraHandle, &focalLength);
			camera.AddMember("focalLength", focalLength, document.GetAllocator());
			float chipHeight;
			world_get_focus_camera_sensor_height(cameraHandle, &chipHeight);
			camera.AddMember("chipHeight", chipHeight, document.GetAllocator());
			float focusDistance;
			world_get_focus_camera_focus_distance(cameraHandle, &focusDistance);
			camera.AddMember("focusDistance", focusDistance, document.GetAllocator());
			float aperture;
			world_get_focus_camera_aperture(cameraHandle, &aperture);
			camera.AddMember("focalLength", aperture, document.GetAllocator());
			break;
			// TODO ORTHO (not implemented yet)
		default:
			// TODO Exception?
			break;
		}


		Vec3 position; 
		world_get_camera_position(cameraHandle, &position);
		camera.AddMember("path", store_in_array(position, document), document.GetAllocator());

		Vec3 viewDirection;
		world_get_camera_direction(cameraHandle, &viewDirection);
		camera.AddMember("viewDir", store_in_array(viewDirection, document), document.GetAllocator());

		Vec3 upDirection;
		world_get_camera_direction(cameraHandle, &upDirection);
		camera.AddMember("up", store_in_array(upDirection, document), document.GetAllocator());

		rapidjson::Value cameraName;
		cameraName.SetString(rapidjson::StringRef(world_get_camera_name(cameraHandle)));
		cameras.AddMember(cameraName, camera, document.GetAllocator());
	}
	document.AddMember("cameras", cameras, document.GetAllocator());
	return true;
}

bool SceneExporter::save_lights(rapidjson::Document& document) const
{
	rapidjson::Value lights;
	lights.SetObject();

	size_t pointLightCount = world_get_point_light_count();

	for (size_t i = 0; i < pointLightCount; i++)
	{
		rapidjson::Value light;
		light.SetObject();

		LightHdl lightHandle = world_get_light_handle(i, LIGHT_POINT);

		light.AddMember("type", "point", document.GetAllocator());

		Vec3 position;
		world_get_point_light_position(lightHandle, &position);
		light.AddMember("position", store_in_array(position, document), document.GetAllocator());

		Vec3 intensity;
		world_get_point_light_intensity(lightHandle, &intensity);
		light.AddMember("intensity", store_in_array(intensity, document), document.GetAllocator());

		light.AddMember("scale", 1.0f, document.GetAllocator());

		rapidjson::Value lightName;
		lightName.SetString(rapidjson::StringRef(world_get_light_name(lightHandle)));
		lights.AddMember(lightName, light, document.GetAllocator());
	}

	size_t dirLightCount = world_get_dir_light_count();

	for (size_t i = 0; i < dirLightCount; i++)
	{
		rapidjson::Value light;
		light.SetObject();

		LightHdl lightHandle = world_get_light_handle(i, LIGHT_DIRECTIONAL);

		light.AddMember("type", "directional", document.GetAllocator());

		Vec3 direction;
		world_get_dir_light_direction(lightHandle, &direction);
		light.AddMember("direction", store_in_array(direction, document), document.GetAllocator());

		Vec3 radiance;
		world_get_dir_light_irradiance(lightHandle, &radiance);
		light.AddMember("radiance", store_in_array(radiance, document), document.GetAllocator());

		light.AddMember("scale", 1.0f, document.GetAllocator());

		rapidjson::Value lightName;
		lightName.SetString(rapidjson::StringRef(world_get_light_name(lightHandle)));
		lights.AddMember(lightName, light, document.GetAllocator());
	}

	size_t spotLightCount = world_get_spot_light_count();

	for (size_t i = 0; i < spotLightCount; i++)
	{
		rapidjson::Value light;
		light.SetObject();

		LightHdl lightHandle = world_get_light_handle(i, LIGHT_SPOT);

		light.AddMember("type", "spot", document.GetAllocator());

		Vec3 position;
		world_get_spot_light_position(lightHandle, &position);
		light.AddMember("position", store_in_array(position, document), document.GetAllocator());

		Vec3 direction;
		world_get_spot_light_direction(lightHandle, &direction);
		light.AddMember("direction", store_in_array(direction, document), document.GetAllocator());

		Vec3 intensity;
		world_get_spot_light_intensity(lightHandle, &intensity);
		light.AddMember("intensity", store_in_array(intensity, document), document.GetAllocator());

		light.AddMember("scale", 1.0f, document.GetAllocator());

		float width;
		world_get_spot_light_angle(lightHandle, &width);
		light.AddMember("width", width, document.GetAllocator());

		float falloffStart;
		world_get_spot_light_falloff(lightHandle, &falloffStart);
		light.AddMember("falloffStart", falloffStart, document.GetAllocator());

		rapidjson::Value lightName;
		lightName.SetString(rapidjson::StringRef(world_get_light_name(lightHandle)));
		lights.AddMember(lightName, light, document.GetAllocator());
	}

	size_t envLightCount = world_get_env_light_count();

	for (size_t i = 0; i < envLightCount; i++)
	{
		rapidjson::Value light;
		light.SetObject();

		LightHdl lightHandle = world_get_light_handle(i, LIGHT_ENVMAP);

		light.AddMember("type", "envmap", document.GetAllocator());

		fs::path mapPath(world_get_env_light_map(lightHandle));

		light.AddMember("map", store_in_string_relative_to_destination_path(mapPath, document), document.GetAllocator());

		Vec3 scale;
		world_get_env_light_scale(lightHandle, &scale);
		light.AddMember("scale", store_in_array(scale, document), document.GetAllocator());

		rapidjson::Value lightName;
		lightName.SetString(rapidjson::StringRef(world_get_light_name(lightHandle)));
		lights.AddMember(lightName, light, document.GetAllocator());
	}

	// TODO GONIOMETRIC (not implemented yet)

	document.AddMember("lights", lights, document.GetAllocator());

	return true;
}

bool SceneExporter::save_materials(rapidjson::Document& document) const
{
	rapidjson::Value materials;
	materials.SetObject();
	size_t materialCount = world_get_material_count();

	for (size_t i = 0; i < materialCount; i++)
	{
		MaterialHdl materialHandle = world_get_material(IndexType(i));

		MaterialParams materialParams;
		world_get_material_data(materialHandle, &materialParams);

		rapidjson::Value materialName;
		materialName.SetString(std::to_string(i).c_str(), std::to_string(i).length(), document.GetAllocator()); // TODO get right material name (not implemented yet)

		materials.AddMember(materialName, save_material(materialParams, document), document.GetAllocator());
	}

	document.AddMember("materials", materials, document.GetAllocator());
	return true;
}

rapidjson::Value SceneExporter::save_material(MaterialParams materialParams, rapidjson::Document& document) const
{
	MaterialParamType matType = materialParams.innerType;

	rapidjson::Value material;
	material.SetObject();

	switch (matType)
	{
	case (MaterialParamType::MATERIAL_BLEND):
		material.AddMember("type", "blend", document.GetAllocator());
		material.AddMember("layerA", save_material(*materialParams.inner.blend.a.mat, document), document.GetAllocator());
		material.AddMember("layerB", save_material(*materialParams.inner.blend.b.mat, document), document.GetAllocator());
		material.AddMember("factorA", materialParams.inner.blend.a.factor, document.GetAllocator());
		material.AddMember("factorB", materialParams.inner.blend.b.factor, document.GetAllocator());
		break;
	case (MaterialParamType::MATERIAL_EMISSIVE):
		material.AddMember("type", "emissive", document.GetAllocator());
		// TextureHdl radianceTextureHandle = materialParams.inner.emissive.radiance;
		// TODO Extract radiance(path) from texture handle (not Implemented yet)
		material.AddMember("radiance", "C++ Scene Exporter: Not implemented yet :(", document.GetAllocator());
		material.AddMember("scale", store_in_array(materialParams.inner.emissive.scale, document), document.GetAllocator());
		break;
	case(MATERIAL_FRESNEL):
		material.AddMember("type", "fresnel", document.GetAllocator());
		material.AddMember("layerReflection", save_material(*materialParams.inner.fresnel.a, document), document.GetAllocator());
		material.AddMember("layerRefraction", save_material(*materialParams.inner.fresnel.b, document), document.GetAllocator());
		material.AddMember("refractionIndex", store_in_array(materialParams.inner.fresnel.refractionIndex, document), document.GetAllocator());
		break;
	case(MATERIAL_LAMBERT):
		material.AddMember("type", "lambert", document.GetAllocator());
		// TextureHdl albedoTextureHandle = materialParams.inner.lambert.albedo
		// TODO Extract albedo(path) from texture handle (not Implemented yet)
		material.AddMember("albedo", "C++ Scene Exporter: Not implemented yet :(", document.GetAllocator());
		break;
	case(MATERIAL_ORENNAYAR):
		material.AddMember("type", "orennayar", document.GetAllocator());
		// TextureHdl albedoTextureHandle = materialParams.inner.orennayar.albedo
		// TODO Extract albedo(path) from texture handle (not Implemented yet)
		material.AddMember("albedo", "C++ Scene Exporter: Not implemented yet :(", document.GetAllocator());
		material.AddMember("roughness", materialParams.inner.orennayar.roughness, document.GetAllocator());
		break;
	case(MATERIAL_TORRANCE):
		material.AddMember("type", "torrance", document.GetAllocator());
		// TextureHdl albedoTextureHandle = materialParams.inner.torrance.roughness
		// TODO Extract roughness(path) from texture handle (not Implemented yet)
		material.AddMember("roughness", "C++ Scene Exporter: Not implemented yet :(", document.GetAllocator());
		material.AddMember("ndf", materialParams.inner.torrance.ndf, document.GetAllocator());
		// TextureHdl albedoTextureHandle = materialParams.inner.torrance.albedo
		// TODO Extract albedo(path) from texture handle (not Implemented yet)
		material.AddMember("albedo", "C++ Scene Exporter: Not implemented yet :(", document.GetAllocator());
		break;
	case(MATERIAL_WALTER):
		material.AddMember("type", "walter", document.GetAllocator());
		// TextureHdl albedoTextureHandle = materialParams.inner.walter.roughness
		// TODO Extract roughness(path) from texture handle (not Implemented yet)
		material.AddMember("roughness", "C++ Scene Exporter: Not implemented yet :(", document.GetAllocator());
		material.AddMember("ndf", materialParams.inner.walter.ndf, document.GetAllocator());
		material.AddMember("absorption", store_in_array(materialParams.inner.walter.absorption, document), document.GetAllocator());
	default:
		// TODO Exception?
		break;
	}

	return material;
}

	bool SceneExporter::save_scenarios(rapidjson::Document& document) const
{
	rapidjson::Value scenarios;
	scenarios.SetObject();
	size_t scenarioCount = world_get_scenario_count();
	for (size_t i = 0; i < scenarioCount; i++)
	{
		rapidjson::Value scenario;
		scenario.SetObject();
		ScenarioHdl scenarioHandle = world_get_scenario_by_index(uint32_t(i));

		rapidjson::Value camera;
		camera.SetString(rapidjson::StringRef(world_get_camera_name(scenario_get_camera(scenarioHandle))));
		scenario.AddMember("camera", camera, document.GetAllocator());

		uint32_t  resWidth;
		uint32_t  resHeight;

		scenario_get_resolution(scenarioHandle, &resWidth, &resHeight);
		rapidjson::Value resolution;
		resolution.SetArray();
		resolution.PushBack(resWidth, document.GetAllocator());
		resolution.PushBack(resHeight, document.GetAllocator());
		scenario.AddMember("resolution", resolution, document.GetAllocator());

		rapidjson::Value lights;
		lights.SetArray();
		if (scenario_has_envmap_light(scenarioHandle))
		{
			LightHdl lightHandle = scenario_get_light_handle(scenarioHandle, 0, LIGHT_ENVMAP);
			lights.PushBack(rapidjson::StringRef(world_get_light_name(lightHandle)), document.GetAllocator());
		}
		size_t pointLightCount = scenario_get_point_light_count(scenarioHandle);

		for (size_t j = 0; j < pointLightCount; j++)
		{
			LightHdl lightHandle = scenario_get_light_handle(scenarioHandle, IndexType(j), LIGHT_POINT);
			lights.PushBack(rapidjson::StringRef(world_get_light_name(lightHandle)), document.GetAllocator());
		}

		size_t dirLightCount = scenario_get_dir_light_count(scenarioHandle);

		for (size_t j = 0; j < dirLightCount; j++)
		{
			LightHdl lightHandle = scenario_get_light_handle(scenarioHandle, IndexType(j), LIGHT_DIRECTIONAL);
			lights.PushBack(rapidjson::StringRef(world_get_light_name(lightHandle)), document.GetAllocator());
		}
		size_t spotLightCount = scenario_get_spot_light_count(scenarioHandle);

		for (size_t j = 0; j < spotLightCount; j++)
		{
			LightHdl lightHandle = scenario_get_light_handle(scenarioHandle, IndexType(j), LIGHT_SPOT);
			lights.PushBack(rapidjson::StringRef(world_get_light_name(lightHandle)), document.GetAllocator());
		}

		// TODO GONIOMETRIC (not implemented yet)

		scenario.AddMember("lights", lights, document.GetAllocator());

		scenarios.AddMember("lod", scenario_get_global_lod_level(scenarioHandle), document.GetAllocator());


		rapidjson::Value materialAssignments;
		materialAssignments.SetArray();

		// TODO Material Assignments (not implemented yet)

		scenario.AddMember("materialAssignments", materialAssignments, document.GetAllocator());

		rapidjson::Value objectProperties;
		objectProperties.SetArray();

		// TODO Object Properties (not implemented yet)

		scenario.AddMember("objectProperties", objectProperties, document.GetAllocator());

		rapidjson::Value name;
		name.SetString(rapidjson::StringRef(scenario_get_name(scenarioHandle)));
		scenarios.AddMember(name, scenario, document.GetAllocator());
		scenario_get_name(scenarioHandle);
	}
	document.AddMember("scenarios", scenarios, document.GetAllocator());
	return true;
}


rapidjson::Value SceneExporter::store_in_array(Vec3 value, rapidjson::Document& document) const
{
	rapidjson::Value out;
	out.SetArray();
	out.PushBack(value.x, document.GetAllocator());
	out.PushBack(value.y, document.GetAllocator());
	out.PushBack(value.z, document.GetAllocator());

	return out;
}

rapidjson::Value SceneExporter::store_in_array(Vec2 value, rapidjson::Document& document) const
{
	rapidjson::Value out;
	out.SetArray();
	out.PushBack(value.x, document.GetAllocator());
	out.PushBack(value.y, document.GetAllocator());

	return out;
}

	rapidjson::Value SceneExporter::store_in_string_relative_to_destination_path(const fs::path& path, rapidjson::Document& document) const
{
	fs::path copy = m_fileDestinationPath;
	rapidjson::Value out;
	std::string s = fs::relative(path, copy.remove_filename()).string();
	out.SetString(s.c_str(), s.length(), document.GetAllocator());
	return out;
}
} // namespace mff_loader::exprt