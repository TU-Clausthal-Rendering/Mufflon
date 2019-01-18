#include "scene_exporter.hpp"
#include "rapidjson/stringbuffer.h"
#include "rapidjson/prettywriter.h"

#include "core/export/interface.h"

#include <fstream> 

namespace mff_loader::exprt {

bool SceneExporter::save_scene()
{
	// TODO Open old Json

	fs::path tempPath = m_filePath;

	tempPath.replace_extension("json");

	rapidjson::Document document;
	document.SetObject();

	document.AddMember("version", FILE_VERSION, document.GetAllocator());
	std::string binaryPathString = m_filePath.string(); // TODO RelPath
	binaryPathString.append(".mff");
	rapidjson::Value binaryPath;
	binaryPath.SetString(binaryPathString.c_str(), rapidjson::SizeType(binaryPathString.length()));
	document.AddMember("binary", binaryPath, document.GetAllocator());

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
	std::ofstream ofs(tempPath);

	ofs << json;

	ofs.close();


	// Binary


	return true;
}

bool SceneExporter::save_cameras(rapidjson::Document& document)
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
		// TODO path viewDir up
		rapidjson::Value cameraName;
		cameraName.SetString(rapidjson::StringRef(world_get_camera_name(cameraHandle)));
		cameras.AddMember(cameraName, camera, document.GetAllocator());
	}

	document.AddMember("cameras", cameras, document.GetAllocator());
	return true;
}

bool SceneExporter::save_lights(rapidjson::Document& document)
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

		Vec3 position; // TODO Replace Vec3 with something we can export
		world_get_point_light_position(lightHandle, &position);
		rapidjson::Value pos;
		pos.SetArray();
		pos.PushBack(position.x, document.GetAllocator());
		pos.PushBack(position.y, document.GetAllocator());
		pos.PushBack(position.z, document.GetAllocator());
		light.AddMember("position", pos, document.GetAllocator());

		Vec3 intensity;
		world_get_point_light_intensity(lightHandle, &intensity);
		rapidjson::Value intens;
		intens.SetArray();
		intens.PushBack(intensity.x, document.GetAllocator());
		intens.PushBack(intensity.y, document.GetAllocator());
		intens.PushBack(intensity.z, document.GetAllocator());
		light.AddMember("intensity", intens, document.GetAllocator());

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
		rapidjson::Value dir;
		dir.SetArray();
		dir.PushBack(direction.x, document.GetAllocator());
		dir.PushBack(direction.y, document.GetAllocator());
		dir.PushBack(direction.z, document.GetAllocator());
		light.AddMember("direction", dir, document.GetAllocator());

		Vec3 radiance;
		world_get_dir_light_irradiance(lightHandle, &radiance);
		rapidjson::Value rad;
		rad.SetArray();
		rad.PushBack(radiance.x, document.GetAllocator());
		rad.PushBack(radiance.y, document.GetAllocator());
		rad.PushBack(radiance.z, document.GetAllocator());
		light.AddMember("radiance", rad, document.GetAllocator());

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
		rapidjson::Value pos;
		pos.SetArray();
		pos.PushBack(position.x, document.GetAllocator());
		pos.PushBack(position.y, document.GetAllocator());
		pos.PushBack(position.z, document.GetAllocator());
		light.AddMember("position", pos, document.GetAllocator());

		Vec3 direction;
		world_get_spot_light_direction(lightHandle, &direction);
		rapidjson::Value dir;
		dir.SetArray();
		dir.PushBack(direction.x, document.GetAllocator());
		dir.PushBack(direction.y, document.GetAllocator());
		dir.PushBack(direction.z, document.GetAllocator());
		light.AddMember("direction", dir, document.GetAllocator());

		Vec3 intensity;
		world_get_spot_light_intensity(lightHandle, &intensity);
		rapidjson::Value intens;
		intens.SetArray();
		intens.PushBack(intensity.x, document.GetAllocator());
		intens.PushBack(intensity.y, document.GetAllocator());
		intens.PushBack(intensity.z, document.GetAllocator());
		light.AddMember("intensity", intens, document.GetAllocator());

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


		rapidjson::Value map;
		map.SetString(rapidjson::StringRef(world_get_env_light_map(lightHandle)));
		light.AddMember("map", map, document.GetAllocator());

		Vec3 scale;
		world_get_env_light_scale(lightHandle, &scale);
		rapidjson::Value sca;
		sca.SetArray();
		sca.PushBack(scale.x, document.GetAllocator());
		sca.PushBack(scale.y, document.GetAllocator());
		sca.PushBack(scale.z, document.GetAllocator());
		light.AddMember("scale", sca, document.GetAllocator());

		rapidjson::Value lightName;
		lightName.SetString(rapidjson::StringRef(world_get_light_name(lightHandle)));
		lights.AddMember(lightName, light, document.GetAllocator());
	}

	// TODO GONIOMETRIC (not implemented yet)

	document.AddMember("lights", lights, document.GetAllocator());

	return true;
}

bool SceneExporter::save_materials(rapidjson::Document& document)
{
	rapidjson::Value materials;
	materials.SetObject();
	//size_t materialCount = ;// TODO Wait for implementation

	return true;
}

bool SceneExporter::save_scenarios(rapidjson::Document& document)
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

		for (size_t j = 0; j < pointLightCount; i++)
		{
			LightHdl lightHandle = scenario_get_light_handle(scenarioHandle, j, LIGHT_POINT);
			lights.PushBack(rapidjson::StringRef(world_get_light_name(lightHandle)), document.GetAllocator());
		}

		size_t dirLightCount = scenario_get_dir_light_count(scenarioHandle);

		for (size_t j = 0; j < dirLightCount; i++)
		{
			LightHdl lightHandle = scenario_get_light_handle(scenarioHandle, j, LIGHT_DIRECTIONAL);
			lights.PushBack(rapidjson::StringRef(world_get_light_name(lightHandle)), document.GetAllocator());
		}
		size_t spotLightCount = scenario_get_spot_light_count(scenarioHandle);

		for (size_t j = 0; j < spotLightCount; i++)
		{
			LightHdl lightHandle = scenario_get_light_handle(scenarioHandle, j, LIGHT_SPOT);
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
} // namespace mff_loader::exprt