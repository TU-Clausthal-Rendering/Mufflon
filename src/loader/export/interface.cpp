#include "interface.h"
#include "binary.hpp"
#include "filesystem.hpp"
#include "util/assert.hpp"
#include "util/log.hpp"
#include "util/punning.hpp"
#include "util/degrad.hpp"
#include "loader/json/json.hpp"
#include "core/export/interface.h"
#include <rapidjson/document.h>
#include <rapidjson/rapidjson.h>
#include <rapidjson/error/en.h>
#include <ei/vector.hpp>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>

#define FUNCTION_NAME __func__

inline constexpr const char FILE_VERSION[] = "1.0";

using namespace mufflon;
using namespace loader;
using namespace loader::json;

namespace {

// Reads a file completely and returns the string containing all bytes
std::string read_file(fs::path path) {
	const std::uintmax_t fileSize = fs::file_size(path);
	std::string fileString;
	fileString.resize(fileSize);

	std::ifstream file(path, std::ios::binary);
	file.read(&fileString[0u], fileSize);
	if(file.gcount() != fileSize)
		logWarning("[", FUNCTION_NAME, "] File '", path.string(), "'not read completely");
	// Finalize the string
	fileString[file.gcount()] = '\0';
	return fileString;
}

void parse_cameras(ParserState& state, const rapidjson::Value& cameras) {
	using namespace rapidjson;
	assertObject(state, cameras);
	state.current = ParserState::Level::CAMERAS;


	for(auto cameraIter = cameras.MemberBegin(); cameraIter != cameras.MemberEnd(); ++cameraIter) {
		const Value& camera = cameraIter->value;
		assertObject(state, camera);
		state.objectNames.push_back(cameraIter->name.GetString());

		// Read common values
		// Placeholder values, because we don't know the scene size yet
		// TODO: parse binary before JSON!
		const float near = read_opt<float>(state, camera, "near", std::numeric_limits<float>::max());
		const float far = read_opt<float>(state, camera, "near", std::numeric_limits<float>::max());
		std::string_view type = read<const char*>(state, get(state, camera, "type"));
		std::vector<ei::Vec3> camPath;
		std::vector<ei::Vec3> camViewDir;
		std::vector<ei::Vec3> camUp;
		read(state, get(state, camera, "path"), camPath);
		read(state, get(state, camera, "viewDir"), camViewDir, camPath.size());
		auto upIter = get(state, camera, "up", false);
		if(upIter != camera.MemberEnd()) {
			read(state, get(state, camera, "up"), camUp, camPath.size());
		} else {
			camUp.push_back(ei::Vec3{ 0, 1, 0 });
		}

		// Per-camera-model values
		if(type.compare("pinhole") == 0) {
			// Pinhole camera
			const float fovDegree = read_opt<float>(state, camera, "fov", 25.f);
			// TODO: add entire path!
			if(camPath.size() > 1u)
				logWarning("[", FUNCTION_NAME, "] Scene file: camera paths are not supported yet");
			if(world_add_pinhole_camera(cameraIter->name.GetString(), util::pun<Vec3>(camPath[0u]),
										util::pun<Vec3>(camViewDir[0u]),
										util::pun<Vec3>(camUp[0u]), near, far,
										static_cast<Radians>(Degrees(fovDegree))) == nullptr)
				throw std::runtime_error("Failed to add camera '"
										 + std::string(cameraIter->name.GetString()) + "'");
		} else if(type.compare("focus") == 0 == 0) {
			// TODO: Focus camera
			logWarning("[", FUNCTION_NAME, "] Scene file: Focus cameras are not supported yet");
		} else if(type.compare("ortho") == 0 == 0) {
			// TODO: Orthogonal camera
			logWarning("[", FUNCTION_NAME, "] Scene file: Focus cameras are not supported yet");
		} else {
			logWarning("[", FUNCTION_NAME, "] Scene file: camera object '",
					   cameraIter->name.GetString(), "' has unknown type '", type, "'");
		}

		state.objectNames.pop_back();
	}
}

void parse_lights(ParserState& state, const rapidjson::Value& lights) {
	using namespace rapidjson;
	assertObject(state, lights);
	state.current = ParserState::Level::LIGHTS;

	for(auto lightIter = lights.MemberBegin(); lightIter != lights.MemberEnd(); ++lightIter) {
		const Value& light = lightIter->value;
		assertObject(state, light);
		state.objectNames.push_back(lightIter->name.GetString());

		// Read common values (aka the type only)
		std::string_view type = read<const char*>(state, get(state, light, "type"));
		if(type.compare("point") == 0) {
			// Point light
			const ei::Vec3 position = read<ei::Vec3>(state, get(state, light, "position"));
			ei::Vec3 intensity;
			auto intensityIter = get(state, light, "intensity", false);
			if(intensityIter != light.MemberEnd())
				intensity = read<ei::Vec3>(state, intensityIter);
			else
				intensity = read<ei::Vec3>(state, get(state, light, "flux")) * 4.f * ei::PI;
			intensity *= read_opt<float>(state, light, "scale", 1.f);

			if(world_add_point_light(lightIter->name.GetString(), util::pun<Vec3>(position),
									 util::pun<Vec3>(intensity)) == nullptr)
				throw std::runtime_error("Failed to add point light '"
										 + std::string(lightIter->name.GetString()) + "'");
		} else if(type.compare("spot") == 0) {
			// Spot light
			const ei::Vec3 position = read<ei::Vec3>(state, get(state, light, "position"));
			const ei::Vec3 direction = read<ei::Vec3>(state, get(state, light, "direction"));
			const ei::Vec3 intensity = read<ei::Vec3>(state, get(state, light, "intensity"))
								* read_opt<float>(state, light, "scale", 1.f);
			Radians angle;
			Radians falloffStart;
			auto angleIter = get(state, light, "cosWidth", false);
			if(angleIter != light.MemberEnd())
				angle = std::acos(static_cast<Radians>(Degrees(read<float>(state, angleIter))));
			else
				angle = static_cast<Radians>(Degrees(read<float>(state, get(state, light, "width"))));
			auto falloffIter = get(state, light, "cosFalloffStart", false);
			if(falloffIter != light.MemberEnd())
				falloffStart = std::acos(read<float>(state, falloffIter));
			else
				falloffStart = static_cast<Radians>(Degrees(read_opt<float>(state, light, "falloffWidth",
								static_cast<Radians>(Degrees(angle)))));

			if(world_add_spot_light(lightIter->name.GetString(), util::pun<Vec3>(position),
									util::pun<Vec3>(direction), util::pun<Vec3>(intensity),
									angle, falloffStart) == nullptr)
				throw std::runtime_error("Failed to add spot light '"
										 + std::string(lightIter->name.GetString()) + "'");
		} else if(type.compare("directional") == 0) {
			// Directional light
			const ei::Vec3 direction = read<ei::Vec3>(state, get(state, light, "direction"));
			const ei::Vec3 radiance = read<ei::Vec3>(state, get(state, light, "radiance"))
								* read_opt<float>(state, light, "scale", 1.f);

			if(world_add_directional_light(lightIter->name.GetString(), util::pun<Vec3>(direction),
										   util::pun<Vec3>(radiance)) == nullptr)
				throw std::runtime_error("Failed to add directional light '"
										 + std::string(lightIter->name.GetString()) + "'");
		} else if(type.compare("envmap") == 0) {
			// Environment-mapped light
			const char* texPath = read<const char*>(state, get(state, light, "map"));
			const float scale = read_opt<float>(state, light, "scale", 1.f);
				// TODO: load the texture
			TextureHdl texture = world_add_texture(texPath, 0u, 0u, 0u, TextureFormat::FORMAT_R8U,
												   TextureSampling::SAMPLING_NEAREST, false, nullptr);
			if(texture == nullptr)
				throw std::runtime_error("Failed to load texture for envmap light '"
										 + std::string(lightIter->name.GetString()) + "'");
			// TODO: incorporate scale

			if(world_add_envmap_light(lightIter->name.GetString(), texture) == nullptr)
				throw std::runtime_error("Failed to add directional light '"
										 + std::string(lightIter->name.GetString()) + "'");
		} else if(type.compare("goniometric") == 0) {
			// TODO: Goniometric light
			const ei::Vec3 position = read<ei::Vec3>(state, get(state, light, "position"));
			const char* texPath = read<const char*>(state, get(state, light, "map"));
			const float scale = read_opt<float>(state, light, "scale", 1.f);
				// TODO: load the texture
			TextureHdl texture = world_add_texture(texPath, 0u, 0u, 0u, TextureFormat::FORMAT_R8U,
												   TextureSampling::SAMPLING_NEAREST, false, nullptr);
			if(texture == nullptr)
				throw std::runtime_error("Failed to load texture for goniometric light '"
										 + std::string(lightIter->name.GetString()) + "'");
			// TODO: incorporate scale

			logWarning("[", FUNCTION_NAME, "] Scene file: Goniometric lights are not supported yet");
		} else {
			logWarning("[", FUNCTION_NAME, "] Scene file: light object '",
					   lightIter->name.GetString(), "' has unknown type '", type, "'");
		}

		state.objectNames.pop_back();
	}
}

void parse_materials(ParserState& state, const rapidjson::Value& materials) {
	using namespace rapidjson;
	assertObject(state, materials);
	state.current = ParserState::Level::MATERIALS;

	for(auto matIter = materials.MemberBegin(); matIter != materials.MemberEnd(); ++matIter) {
	}
}

void parse_scenarios(ParserState& state, const rapidjson::Value& scenarios) {
	using namespace rapidjson;
	assertObject(state, scenarios);
	state.current = ParserState::Level::SCENARIOS;

	for(auto scenarioIter = scenarios.MemberBegin(); scenarioIter != scenarios.MemberEnd(); ++scenarioIter) {
		const Value& scenario = scenarioIter->value;
		assertObject(state, scenario);
		state.objectNames.push_back(scenarioIter->name.GetString());

		const char* camera = read<const char*>(state, get(state, scenario, "camera"));
		ei::IVec2 resolution = read<ei::IVec2>(state, get(state, scenario, "resolution"));
		std::vector<const char*> lights;
		auto lightIter = get(state, scenario, "lights", false);
		std::size_t lod = read_opt<std::size_t>(state, scenario, "lod", 0u);
		
		CameraHdl camHdl = world_get_camera(camera);
		if(camHdl == nullptr)
			throw std::runtime_error("Camera '" + std::string(camera) + "' does not exist");
		ScenarioHdl scenarioHdl = world_create_scenario(scenarioIter->name.GetString());
		if(scenarioHdl == nullptr)
			throw std::runtime_error("Failed to create scenario '" + std::string(scenarioIter->name.GetString()) + "'");

		if(!scenario_set_camera(scenarioHdl, camHdl))
			throw std::runtime_error("Failed to set camera '" + std::string(camera) + "' for scenario '"
									 + std::string(scenarioIter->name.GetString()) + "'");
		if(!scenario_set_resolution(scenarioHdl, resolution.x, resolution.y))
			throw std::runtime_error("Failed to set resolution '" + std::to_string(resolution.x) + "x"
									 + std::to_string(resolution.y) + " for scenario '"
									 + std::string(scenarioIter->name.GetString()) + "'");
		if(!scenario_set_global_lod_level(scenarioHdl, lod))
			throw std::runtime_error("Failed to set LoD " + std::to_string(lod) + " for scenario '"
									 + std::string(scenarioIter->name.GetString()) + "'");

		// Add lights
		if(lightIter != scenario.MemberEnd()) {
			assertArray(state, lightIter);
			state.objectNames.push_back("lights");
			for(SizeType i = 0u; i < lightIter->value.Size(); ++i) {
				const char* lightName = read<const char*>(state, lightIter->value[i]);
				if(!scenario_add_light(scenarioHdl, lightName))
					throw std::runtime_error("Failed to add light '" + std::string(lightName) + "' to scenario '"
											 + std::string(scenarioIter->name.GetString()) + "'");
			}
		}

		// Add objects
		auto objectsIter = get(state, scenario, "objectProperties", false);
		if(objectsIter != scenario.MemberEnd()) {
			state.objectNames.push_back(objectsIter->name.GetString());
			assertObject(state, objectsIter->value);
			for(auto objIter = objectsIter->value.MemberBegin(); objIter != objectsIter->value.MemberEnd(); ++objIter) {
				std::string_view objectName = objIter->name.GetString();
				state.objectNames.push_back(&objectName[0u]);
				const Value& object = objIter->value;
				assertObject(state, object);
				// Check for object name meta-tag
				if(std::strncmp(&objectName[0u], "[obj:", 5u) != 0)
					continue;
				std::string subName{ objectName.substr(5u, objectName.length() - 6u) };
				ObjectHdl objHdl = world_get_object(&objectName[0u]);
				if(objHdl == nullptr)
					throw std::runtime_error("Failed to find object '" + subName + "' from scenario '"
											 + std::string(scenarioIter->name.GetString()) + "'");
				// Check for LoD and masked
				auto lodIter = get(state, object, "lod", false);
				if(lodIter != object.MemberEnd())
					if(!scenario_set_object_lod(scenarioHdl, objHdl, read<std::size_t>(state, lodIter)))
						throw std::runtime_error("Failed to set LoD level of object '" + subName
												 + "' for scenario '" + std::string(scenarioIter->name.GetString())
												 + "'");
				if(object.HasMember("masked"))
					if(!scenario_mask_object(scenarioHdl, objHdl))
						throw std::runtime_error("Failed to set mask for object '" + subName
												 + "' for scenario '" + std::string(scenarioIter->name.GetString())
												 + "'");

				state.objectNames.pop_back();
			}
			state.objectNames.pop_back();
		}

		auto materialsIter = get(state, scenario, "materialAssignments");
		state.objectNames.push_back(materialsIter->name.GetString());
		assertObject(state, materialsIter->value);
		for(auto matIter = materialsIter->value.MemberBegin(); matIter != materialsIter->value.MemberEnd(); ++matIter) {
			std::string_view matName = matIter->name.GetString();
			state.objectNames.push_back(&matName[0u]);
			// Check for object name meta-tag
			if(std::strncmp(&matName[0u], "[mat:", 5u) != 0)
				continue;
			std::string subName{ matName.substr(5u, matName.length() - 6u) };
			const std::string_view inScenName = read<const char*>(state, matIter->value);
			// TODO: create material association
			// TODO: check if all materials were associated
		}
	}
}

void parse_scene_file(fs::path filePath) {
	using namespace rapidjson;

	// Make the file path absolute (easier to deal with)
	if(filePath.is_relative())
		filePath = fs::canonical(filePath);

	// Track the state of the parser for error messages
	ParserState state;
	state.current = ParserState::Level::ROOT;

	logInfo("[", FUNCTION_NAME, "] Parsing scene file '", filePath.string(), "'");

	// JSON text
	std::string jsonString = read_file(filePath);
	
	Document document;
	// Parse and check for errors
	ParseResult res = document.Parse(jsonString.c_str());
	if(res.IsError()) {
		// Walk the string and determine line number
		std::stringstream ss(jsonString);
		std::string line;
		const std::size_t offset = res.Offset();
		std::size_t currOffset = 0u;
		std::size_t currLine = 0u;
		while(std::getline(ss, line)) {
			if(offset >= currOffset && offset <= (currOffset + line.length()))
				break;
			// Extra byte for newline
			++currLine;
			currOffset += line.size() + 1u;
		}
		throw std::runtime_error("Parser error: " + std::string(GetParseError_En(res.Code()))
								 + " at offset " + std::to_string(res.Offset())
								 + " (line " + std::to_string(currLine) + ')');
	}

	// Parse our file specification
	assertObject(state, document);
	// Version
	std::string_view version;
	auto versionIter = get(state, document, "version", false);
	if(versionIter == document.MemberEnd()) {
		logWarning("[", FUNCTION_NAME, "] Scene file: no version specified (current one assumed)");
	} else {
		std::string_view version = read<const char*>(state, versionIter);
		if(version.compare(FILE_VERSION) != 0)
			logWarning("[", FUNCTION_NAME, "] Scene file: version mismatch (",
					   version, "(file) vs ", FILE_VERSION, "(current))");
	}
	// Binary file path
	fs::path binaryFile = read<const char*>(state, get(state, document, "binary"));
	
	if(binaryFile.empty()) {
		logError("[", FUNCTION_NAME, "] Scene file: has an empty binary file path");
		return;
	}
	// Make the file path absolute
	if(binaryFile.is_relative())
		binaryFile = fs::canonical(filePath.parent_path() / binaryFile);
	if(!fs::exists(binaryFile)) {
		logError("[", FUNCTION_NAME, "] Scene file: specifies a binary file that doesn't exist ('",
				 binaryFile.string(), "'");
		return;
	}
	// First parse binary file
	binary::parse_file(binaryFile);

	// Cameras
	state.current = ParserState::Level::ROOT;
	parse_cameras(state, get(state, document, "cameras")->value);
	// Lights
	state.current = ParserState::Level::ROOT;
	parse_lights(state, get(state, document, "lights")->value);
	// Materials
	state.current = ParserState::Level::ROOT;
	parse_materials(state, get(state, document, "lights")->value);
	// Scenarios
	state.current = ParserState::Level::ROOT;
	parse_scenarios(state, get(state, document, "scenarios")->value);

	// TODO: parse binary file
}

} // namespace

bool load_scene_file(const char* path) {
	fs::path filePath(path);

	// Perform some error checking
	if (!fs::exists(filePath)) {
		logError("[", FUNCTION_NAME, "] File '", fs::canonical(filePath).string(), "' does not exist");
		return false;
	}
	if (fs::is_directory(filePath)) {
		logError("[", FUNCTION_NAME, "] Path '", fs::canonical(filePath).string(), "' is a directory, not a file");
		return false;
	}
	if (filePath.extension() != ".json")
		logWarning("[", FUNCTION_NAME, "] Scene file does not end with '.json'; attempting to parse it anyway");

	try {
		parse_scene_file(filePath);
	} catch(const std::exception& e) {
		logError("[", FUNCTION_NAME, "] ", e.what());
		return false;
	}

	return true;
}