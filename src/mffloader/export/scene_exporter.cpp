#include "scene_exporter.hpp"
#include "rapidjson/stringbuffer.h"
#include "rapidjson/prettywriter.h"

#include <fstream> 
#include "util/degrad.hpp"
#include "util/log.hpp"
#include "profiler/cpu_profiler.hpp"
#include <sstream>
#include "util/assert.hpp"

namespace mff_loader::exprt {

namespace {

// Utility for comparing Vec3s
bool operator!=(const Vec3& a, const Vec3& b) {
	return (a.x != b.x) && (a.y != b.y) && (a.z != b.z);
}

// Reads a file completely and returns the string containing all bytes
std::string read_file(fs::path path) {
	auto scope = mufflon::Profiler::loader().start<mufflon::CpuProfileState>("JSON read_file", mufflon::ProfileLevel::HIGH);
	mufflon::logPedantic("[read_file] Loading JSON file '", path.string(), "' into RAM");
	const std::uintmax_t fileSize = fs::file_size(path);
	std::string fileString;
	fileString.resize(fileSize);

	std::ifstream file(path, std::ios::binary);
	file.read(&fileString[0u], fileSize);
	if(file.gcount() != static_cast<std::streamsize>(fileSize))
		mufflon::logWarning("[read_file] File '", path.string(), "'not read completely");
	// Finalize the string
	fileString[file.gcount()] = '\0';
	return fileString;
}
class CutomPrettyWriter : public rapidjson::PrettyWriter<rapidjson::StringBuffer>
{
public:
	CutomPrettyWriter(rapidjson::StringBuffer& sBuf)
		: rapidjson::PrettyWriter<rapidjson::StringBuffer>(sBuf)
	{}
	bool Double(double d) { Prefix(rapidjson::kNumberType); return EndValue(WriteDouble(d)); }
	bool WriteDouble(double d) {
		if(rapidjson::internal::Double(d).isnanOrInf()) {
			// Note: This code path can only be reached if (RAPIDJSON_WRITE_DEFAULT_FLAGS & kWriteNanAndInfFlag).
			if(!(rapidjson::kWriteDefaultFlags & rapidjson::kWriteNanAndInfFlag))
				return false;
			if(rapidjson::internal::Double(d).isnan()) {
				PutReserve(*os_, 3);
				PutUnsafe(*os_, 'N'); PutUnsafe(*os_, 'a'); PutUnsafe(*os_, 'N');
				return true;
			}
			if(rapidjson::internal::Double(d).Sign()) {
				PutReserve(*os_, 9);
				PutUnsafe(*os_, '-');
			}
			else
				PutReserve(*os_, 8);
			PutUnsafe(*os_, 'I'); PutUnsafe(*os_, 'n'); PutUnsafe(*os_, 'f');
			PutUnsafe(*os_, 'i'); PutUnsafe(*os_, 'n'); PutUnsafe(*os_, 'i'); PutUnsafe(*os_, 't'); PutUnsafe(*os_, 'y');
			return true;
		}

		char *buffer = os_->Push(25);

		/*double dCopy = d;
		int i = 0;
		if (d != 0.0)
		{
			for (i = -1; abs(dCopy) < 1.0; i++)
			{
				dCopy *= 10;
			}
		}
		char* end = rapidjson::internal::dtoa(d, buffer, std::min(maxDecimalPlaces_ + i, 324));*/

		int end = sprintf(buffer, "%.3g", d);

		os_->Pop(static_cast<size_t>(25 - end));
		return true;
	}
};
}

bool SceneExporter::save_scene() const {
	rapidjson::Document document;
	document.SetObject();

	if(fs::is_regular_file(m_fileDestinationPath)) {
		std::string oldJson = read_file(m_fileDestinationPath);
		//document.Parse(oldJson.c_str());
		// TODO Load oldJson 
	}

	document.AddMember("version", FILE_VERSION, document.GetAllocator());
	document.AddMember("binary", store_in_string_relative_to_destination_path(m_mffPath, document), document.GetAllocator());

	// JSON
	if(!save_cameras(document))
		return false;
	if(!save_lights(document))
		return false;
	if(!save_materials(document))
		return false;
	if(!save_scenarios(document))
		return false;

	rapidjson::StringBuffer strbuf;
	CutomPrettyWriter writer(strbuf); // Overwritten to take relevant decimal places
	writer.SetFormatOptions(rapidjson::kFormatSingleLineArray);
	writer.SetMaxDecimalPlaces(3); // Overwritten to take relevant decimal places
	document.Accept(writer);

	std::string json = std::string(strbuf.GetString());
	std::ofstream ofs(m_fileDestinationPath);

	ofs << json;

	ofs.close();

	// Binary

	return true;
}

bool SceneExporter::save_cameras(rapidjson::Document& document) const {
	rapidjson::Value cameras;
	cameras.SetObject();
	size_t cameraCount = world_get_camera_count(m_mffInstHdl);

	std::uint32_t frameCount;
	if(!world_get_frame_count(m_mffInstHdl, &frameCount))
		throw std::runtime_error("Failed to acquire animation frame count.");

	for(size_t i = 0; i < cameraCount; i++) {
		rapidjson::Value camera;
		camera.SetObject();
		CameraHdl cameraHandle = world_get_camera_by_index(m_mffInstHdl, i);
		CameraType cameraType = world_get_camera_type(cameraHandle);

		switch(cameraType) {
		case CAM_PINHOLE: {
				camera.AddMember("type", "pinhole", document.GetAllocator());
				float vFov;
				world_get_pinhole_camera_fov(cameraHandle, &vFov);
				vFov = static_cast<float>(mufflon::Degrees(mufflon::Radians(vFov))); // Convert Radian to Degree
				camera.AddMember("fov", vFov, document.GetAllocator());
				break;
			}
		case CAM_FOCUS: {
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
			}
			// TODO ORTHO (not implemented yet)
		default:
			mAssert(false);
			// TODO Exception?
			break;
		}
		rapidjson::Value v = rapidjson::Value();

		// Export the paths (possibly animated)
		bool animated = false;
		std::vector<Vec3> positions, directions, ups;
		for(std::uint32_t frame = 0; frame < frameCount; ++frame) {
			Vec3 position;
			world_get_camera_position(cameraHandle, &position, frame);

			Vec3 viewDirection;
			world_get_camera_direction(cameraHandle, &viewDirection, frame);

			Vec3 upDirection;
			world_get_camera_direction(cameraHandle, &upDirection, frame);

			// Check if there are differences
			if(!animated && positions.size() > 1u && (positions.back() != position
				|| directions.back() != viewDirection || ups.back() != upDirection))
				animated = true;
		
			positions.push_back(position);
			directions.push_back(viewDirection);
			ups.push_back(upDirection);
		}

		// If we're not animated only export one frame
		for(std::uint32_t frame = 0; frame < (animated ? frameCount : 1); ++frame) {
			rapidjson::Value positionPath;
			positionPath.SetArray();
			positionPath.PushBack(store_in_array(positions[frame], document), document.GetAllocator());
			camera.AddMember("path", positionPath, document.GetAllocator());

			rapidjson::Value  viewDirectionPath;
			viewDirectionPath.SetArray();
			viewDirectionPath.PushBack(store_in_array(directions[frame], document), document.GetAllocator());
			camera.AddMember("viewDir", viewDirectionPath, document.GetAllocator());

			rapidjson::Value upDirectionPath;
			upDirectionPath.SetArray();
			upDirectionPath.PushBack(store_in_array(ups[frame], document), document.GetAllocator());
			camera.AddMember("up", upDirectionPath, document.GetAllocator());
		}

		rapidjson::Value cameraName;
		cameraName.SetString(rapidjson::StringRef(world_get_camera_name(cameraHandle)));
		cameras.AddMember(cameraName, camera, document.GetAllocator());
	}
	document.AddMember("cameras", cameras, document.GetAllocator());
	return true;
}

bool SceneExporter::save_lights(rapidjson::Document& document) const {
	rapidjson::Value lights;
	lights.SetObject();

	size_t pointLightCount = world_get_point_light_count(m_mffInstHdl);

	for(size_t i = 0; i < pointLightCount; i++) {
		rapidjson::Value light;
		light.SetObject();

		LightHdl lightHandle = world_get_light_handle(i, LIGHT_POINT);

		light.AddMember("type", "point", document.GetAllocator());

		// TODO: animation
		Vec3 position;
		world_get_point_light_position(m_mffInstHdl, lightHandle, &position, 0u);
		light.AddMember("position", store_in_array(position, document), document.GetAllocator());

		Vec3 intensity;
		world_get_point_light_intensity(m_mffInstHdl, lightHandle, &intensity, 0u);
		light.AddMember("intensity", store_in_array(intensity, document), document.GetAllocator());

		light.AddMember("scale", 1.0f, document.GetAllocator());

		rapidjson::Value lightName;
		lightName.SetString(rapidjson::StringRef(world_get_light_name(m_mffInstHdl, lightHandle)));
		lights.AddMember(lightName, light, document.GetAllocator());
	}

	size_t dirLightCount = world_get_dir_light_count(m_mffInstHdl);

	for(size_t i = 0; i < dirLightCount; i++) {
		rapidjson::Value light;
		light.SetObject();

		LightHdl lightHandle = world_get_light_handle(i, LIGHT_DIRECTIONAL);

		light.AddMember("type", "directional", document.GetAllocator());

		// TODO: animation
		Vec3 direction;
		world_get_dir_light_direction(m_mffInstHdl, lightHandle, &direction, 0u);
		light.AddMember("direction", store_in_array(direction, document), document.GetAllocator());

		Vec3 radiance;
		world_get_dir_light_irradiance(m_mffInstHdl, lightHandle, &radiance, 0u);
		light.AddMember("radiance", store_in_array(radiance, document), document.GetAllocator());

		light.AddMember("scale", 1.0f, document.GetAllocator());

		rapidjson::Value lightName;
		lightName.SetString(rapidjson::StringRef(world_get_light_name(m_mffInstHdl, lightHandle)));
		lights.AddMember(lightName, light, document.GetAllocator());
	}

	size_t spotLightCount = world_get_spot_light_count(m_mffInstHdl);

	for(size_t i = 0; i < spotLightCount; i++) {
		rapidjson::Value light;
		light.SetObject();

		LightHdl lightHandle = world_get_light_handle(i, LIGHT_SPOT);

		light.AddMember("type", "spot", document.GetAllocator());

		// TODO: animation
		Vec3 position;
		world_get_spot_light_position(m_mffInstHdl, lightHandle, &position, 0u);
		light.AddMember("position", store_in_array(position, document), document.GetAllocator());

		Vec3 direction;
		world_get_spot_light_direction(m_mffInstHdl, lightHandle, &direction, 0u);
		light.AddMember("direction", store_in_array(direction, document), document.GetAllocator());

		Vec3 intensity;
		world_get_spot_light_intensity(m_mffInstHdl, lightHandle, &intensity, 0u);
		light.AddMember("intensity", store_in_array(intensity, document), document.GetAllocator());

		light.AddMember("scale", 1.0f, document.GetAllocator());

		float width;
		world_get_spot_light_angle(m_mffInstHdl, lightHandle, &width, 0u);
		light.AddMember("width", width, document.GetAllocator());

		float falloffStart;
		world_get_spot_light_falloff(m_mffInstHdl, lightHandle, &falloffStart, 0u);
		light.AddMember("falloffStart", falloffStart, document.GetAllocator());

		rapidjson::Value lightName;
		lightName.SetString(rapidjson::StringRef(world_get_light_name(m_mffInstHdl, lightHandle)));
		lights.AddMember(lightName, light, document.GetAllocator());
	}

	size_t envLightCount = world_get_env_light_count(m_mffInstHdl);

	for(size_t i = 0; i < envLightCount; i++) {
		rapidjson::Value light;
		light.SetObject();

		LightHdl lightHandle = world_get_light_handle(i, LIGHT_ENVMAP);

		light.AddMember("type", "envmap", document.GetAllocator());

		fs::path mapPath(world_get_env_light_map(m_mffInstHdl, lightHandle));

		light.AddMember("map", store_in_string_relative_to_destination_path(mapPath, document), document.GetAllocator());

		Vec3 scale;
		world_get_env_light_scale(m_mffInstHdl, lightHandle, &scale);
		light.AddMember("scale", store_in_array(scale, document), document.GetAllocator());

		rapidjson::Value lightName;
		lightName.SetString(rapidjson::StringRef(world_get_light_name(m_mffInstHdl, lightHandle)));
		lights.AddMember(lightName, light, document.GetAllocator());
	}

	// TODO GONIOMETRIC (not implemented yet)

	document.AddMember("lights", lights, document.GetAllocator());

	return true;
}

bool SceneExporter::save_materials(rapidjson::Document& document) const {
	rapidjson::Value materials;
	materials.SetObject();
	size_t materialCount = world_get_material_count(m_mffInstHdl);

	std::vector<std::byte> buffer;
	for(size_t i = 0; i < materialCount; i++) {
		MaterialHdl materialHandle = world_get_material(m_mffInstHdl, IndexType(i));

		buffer.resize(world_get_material_size(materialHandle));

		MaterialParams* materialParams = reinterpret_cast<MaterialParams*>(buffer.data());

		world_get_material_data(m_mffInstHdl, materialHandle, materialParams);

		rapidjson::Value materialName;
		materialName.SetString(world_get_material_name(materialHandle), document.GetAllocator());

		materials.AddMember(materialName, save_material(*materialParams, document), document.GetAllocator());
	}

	document.AddMember("materials", materials, document.GetAllocator());
	return true;
}

rapidjson::Value SceneExporter::save_material(const MaterialParams& materialParams, rapidjson::Document& document) const {
	MaterialParamType matType = materialParams.innerType;

	rapidjson::Value material;
	material.SetObject();


	switch (matType) {
	case MATERIAL_BLEND: {
			material.AddMember("type", "blend", document.GetAllocator());
			material.AddMember("layerA", save_material(*materialParams.inner.blend.a.mat, document), document.GetAllocator());
			material.AddMember("layerB", save_material(*materialParams.inner.blend.b.mat, document), document.GetAllocator());
			material.AddMember("factorA", materialParams.inner.blend.a.factor, document.GetAllocator());
			material.AddMember("factorB", materialParams.inner.blend.b.factor, document.GetAllocator());
			break;
		}
	case MATERIAL_EMISSIVE: {
			material.AddMember("type", "emissive", document.GetAllocator());
			TextureHdl radianceTextureHandle = materialParams.inner.emissive.radiance;
			add_member_from_texture_handle(radianceTextureHandle, "radiance", material, document);
			material.AddMember("scale", store_in_array(materialParams.inner.emissive.scale, document), document.GetAllocator());
			break;
		}
	case MATERIAL_FRESNEL: {
			material.AddMember("type", "fresnel", document.GetAllocator());
			material.AddMember("layerReflection", save_material(*materialParams.inner.fresnel.a, document), document.GetAllocator());
			material.AddMember("layerRefraction", save_material(*materialParams.inner.fresnel.b, document), document.GetAllocator());
			material.AddMember("refractionIndex", store_in_array(materialParams.inner.fresnel.refractionIndex, document), document.GetAllocator());
			break;
		}
	case MATERIAL_LAMBERT: {
			material.AddMember("type", "lambert", document.GetAllocator());
			TextureHdl albedoTextureHandle = materialParams.inner.lambert.albedo;
			add_member_from_texture_handle(albedoTextureHandle, "albedo", material, document);
			break;
		}
	case MATERIAL_ORENNAYAR: {
			material.AddMember("type", "orennayar", document.GetAllocator());
			TextureHdl albedoTextureHandle = materialParams.inner.orennayar.albedo;
			add_member_from_texture_handle(albedoTextureHandle, "albedo", material, document);
			material.AddMember("roughness", materialParams.inner.orennayar.roughness, document.GetAllocator());
			break;
		}
	case MATERIAL_TORRANCE: {
			material.AddMember("type", "torrance", document.GetAllocator());
			TextureHdl roughnessTextureHandle = materialParams.inner.torrance.roughness;
			add_member_from_texture_handle(roughnessTextureHandle, "roughness", material, document);
			material.AddMember("ndf", materialParams.inner.torrance.ndf, document.GetAllocator());
			TextureHdl albedoTextureHandle = materialParams.inner.torrance.albedo;
			add_member_from_texture_handle(albedoTextureHandle, "albedo", material, document);
			material.AddMember("albedo", "C++ Scene Exporter: Not implemented yet :(", document.GetAllocator());
			break;
		}
	case MATERIAL_WALTER: {
			material.AddMember("type", "walter", document.GetAllocator());
			TextureHdl roughnessTextureHandle = materialParams.inner.walter.roughness;
			add_member_from_texture_handle(roughnessTextureHandle, "roughness", material, document);
			material.AddMember("roughness", "C++ Scene Exporter: Not implemented yet :(", document.GetAllocator());
			material.AddMember("ndf", materialParams.inner.walter.ndf, document.GetAllocator());
			material.AddMember("absorption", store_in_array(materialParams.inner.walter.absorption, document), document.GetAllocator());
			break;
		}
	default:
		mAssert(false);
		// TODO Exception?
		break;
	}

	return material;
}

	bool SceneExporter::save_scenarios(rapidjson::Document& document) const {
	rapidjson::Value scenarios;
	scenarios.SetObject();
	size_t scenarioCount = world_get_scenario_count(m_mffInstHdl);
	for(size_t i = 0; i < scenarioCount; i++) {
		rapidjson::Value scenario;
		scenario.SetObject();
		ScenarioHdl scenarioHandle = world_get_scenario_by_index(m_mffInstHdl, uint32_t(i));

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
		if(scenario_has_envmap_light(m_mffInstHdl, scenarioHandle)) {
			LightHdl lightHandle = scenario_get_light_handle(scenarioHandle, 0, LIGHT_ENVMAP);
			lights.PushBack(rapidjson::StringRef(world_get_light_name(m_mffInstHdl, lightHandle)), document.GetAllocator());
		}
		size_t pointLightCount = scenario_get_point_light_count(scenarioHandle);

		for(size_t j = 0; j < pointLightCount; j++) {
			LightHdl lightHandle = scenario_get_light_handle(scenarioHandle, IndexType(j), LIGHT_POINT);
			lights.PushBack(rapidjson::StringRef(world_get_light_name(m_mffInstHdl, lightHandle)), document.GetAllocator());
		}

		size_t dirLightCount = scenario_get_dir_light_count(scenarioHandle);

		for(size_t j = 0; j < dirLightCount; j++) {
			LightHdl lightHandle = scenario_get_light_handle(scenarioHandle, IndexType(j), LIGHT_DIRECTIONAL);
			lights.PushBack(rapidjson::StringRef(world_get_light_name(m_mffInstHdl, lightHandle)), document.GetAllocator());
		}
		size_t spotLightCount = scenario_get_spot_light_count(scenarioHandle);

		for (size_t j = 0; j < spotLightCount; j++) {
			LightHdl lightHandle = scenario_get_light_handle(scenarioHandle, IndexType(j), LIGHT_SPOT);
			lights.PushBack(rapidjson::StringRef(world_get_light_name(m_mffInstHdl, lightHandle)), document.GetAllocator());
		}

		// TODO GONIOMETRIC (not implemented yet)

		scenario.AddMember("lights", lights, document.GetAllocator());

		scenario.AddMember("lod", scenario_get_global_lod_level(scenarioHandle), document.GetAllocator());


		rapidjson::Value materialAssignments;
		materialAssignments.SetObject();

		size_t materialSlotCount = scenario_get_material_slot_count(scenarioHandle);
		for(size_t j = 0; j < materialSlotCount; j++) {
			const char* materialSlotName = scenario_get_material_slot_name(scenarioHandle, MatIdx(j));
			MaterialHdl materialHandle = scenario_get_assigned_material(scenarioHandle, MatIdx(j));
			const char* materialName = world_get_material_name(materialHandle);

			rapidjson::Value matSlotName;
			matSlotName.SetString(materialSlotName, rapidjson::SizeType(strlen(materialSlotName)));

			rapidjson::Value matName;
			matName.SetString(materialName, rapidjson::SizeType(strlen(materialName)));
			materialAssignments.AddMember(matSlotName, matName, document.GetAllocator());
		}
		
		scenario.AddMember("materialAssignments", materialAssignments, document.GetAllocator());

		rapidjson::Value objectProperties;
		objectProperties.SetObject();

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

bool SceneExporter::add_member_from_texture_handle(const TextureHdl& textureHdl, const std::string& memberName, rapidjson::Value& saveIn,
	rapidjson::Document& document) const {
	if(textureHdl == nullptr)
		return false;
	try	{
		std::string textureName = world_get_texture_name(textureHdl);
		rapidjson::Value textureNameRj;
		textureNameRj.SetString(textureName.c_str(), rapidjson::SizeType(textureName.length()), document.GetAllocator());


		IVec2 texSize;
		world_get_texture_size(textureHdl, &texSize);

		rapidjson::Value name;
		name.SetString(memberName.c_str(), rapidjson::SizeType(memberName.length()), document.GetAllocator());

		if (texSize.x == 1 && texSize.y == 1)
			saveIn.AddMember(name, store_in_array_from_float_string(textureName, document), document.GetAllocator());
		else
			saveIn.AddMember(name, textureNameRj, document.GetAllocator());
	}
	catch(std::ios_base::failure&) {
		return false;
	}
	return true;
}


	rapidjson::Value SceneExporter::store_in_array(Vec3 value, rapidjson::Document& document) const {
	rapidjson::Value out;
	out.SetArray();
	out.PushBack(value.x, document.GetAllocator());
	out.PushBack(value.y, document.GetAllocator());
	out.PushBack(value.z, document.GetAllocator());

	return out;
}

template<typename T>
rapidjson::Value SceneExporter::store_in_array(std::vector<T> value, rapidjson::Document& document) const {
	rapidjson::Value out;
	out.SetArray();
	for (auto& v : value)
		out.PushBack(v, document.GetAllocator());
	return out;
}

rapidjson::Value SceneExporter::store_in_array(Vec2 value, rapidjson::Document& document) const {
	rapidjson::Value out;
	out.SetArray();
	out.PushBack(value.x, document.GetAllocator());
	out.PushBack(value.y, document.GetAllocator());

	return out;
}
rapidjson::Value SceneExporter::store_in_array_from_float_string(std::string floatString, rapidjson::Document & document) const {
	std::vector<float> values;
	std::stringstream ss(floatString);
	try {
		for (int i = 0; i < 4; i++) {
			float f;
			ss >> f;
			values.push_back(f);
			if(ss.eof())
				break;
		}
	}
	catch (std::ios_base::failure&) {
		throw;
	}
	return store_in_array(values, document);
}
	rapidjson::Value SceneExporter::store_in_string_relative_to_destination_path(const fs::path& path, rapidjson::Document& document) const {
	fs::path copy = m_fileDestinationPath;
	rapidjson::Value out;
	std::string s = fs::relative(path, copy.remove_filename()).string();
	out.SetString(s.c_str(), rapidjson::SizeType(s.length()), document.GetAllocator());
	return out;
}
} // namespace mff_loader::exprt
