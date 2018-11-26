#include "json_loader.hpp"
#include "binary.hpp"
#include "util/log.hpp"
#include "util/punning.hpp"
#include "util/int_types.hpp"
#include "util/degrad.hpp"
#include "core/export/interface.h"
#include <rapidjson/document.h>
#include <rapidjson/error/en.h>
#include <fstream>
#include <sstream>

#define FUNCTION_NAME __func__

namespace loader::json {

using namespace mufflon;

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

} // namespace

void JsonLoader::clear_state() {
	m_jsonString.clear();
	m_state.reset();
	m_binaryFile.clear();
}

void JsonLoader::load_cameras() {
	using namespace rapidjson;
	const Value& cameras = m_cameras->value;
	assertObject(m_state, cameras);
	m_state.current = ParserState::Level::CAMERAS;


	for(auto cameraIter = cameras.MemberBegin(); cameraIter != cameras.MemberEnd(); ++cameraIter) {
		const Value& camera = cameraIter->value;
		assertObject(m_state, camera);
		m_state.objectNames.push_back(cameraIter->name.GetString());

		// Read common values
		// Placeholder values, because we don't know the scene size yet
		// TODO: parse binary before JSON!
		const float near = read_opt<float>(m_state, camera, "near", std::numeric_limits<float>::max());
		const float far = read_opt<float>(m_state, camera, "near", std::numeric_limits<float>::max());
		std::string_view type = read<const char*>(m_state, get(m_state, camera, "type"));
		std::vector<ei::Vec3> camPath;
		std::vector<ei::Vec3> camViewDir;
		std::vector<ei::Vec3> camUp;
		read(m_state, get(m_state, camera, "path"), camPath);
		read(m_state, get(m_state, camera, "viewDir"), camViewDir, camPath.size());
		auto upIter = get(m_state, camera, "up", false);
		if(upIter != camera.MemberEnd()) {
			read(m_state, get(m_state, camera, "up"), camUp, camPath.size());
		} else {
			camUp.push_back(ei::Vec3{ 0, 1, 0 });
		}

		// Per-camera-model values
		if(type.compare("pinhole") == 0) {
			// Pinhole camera
			const float fovDegree = read_opt<float>(m_state, camera, "fov", 25.f);
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

		m_state.objectNames.pop_back();
	}
}

void JsonLoader::load_lights() {
	using namespace rapidjson;
	const Value& lights = m_lights->value;
	assertObject(m_state, lights);
	m_state.current = ParserState::Level::LIGHTS;

	for(auto lightIter = lights.MemberBegin(); lightIter != lights.MemberEnd(); ++lightIter) {
		const Value& light = lightIter->value;
		assertObject(m_state, light);
		m_state.objectNames.push_back(lightIter->name.GetString());

		// Read common values (aka the type only)
		std::string_view type = read<const char*>(m_state, get(m_state, light, "type"));
		if(type.compare("point") == 0) {
			// Point light
			const ei::Vec3 position = read<ei::Vec3>(m_state, get(m_state, light, "position"));
			ei::Vec3 intensity;
			auto intensityIter = get(m_state, light, "intensity", false);
			if(intensityIter != light.MemberEnd())
				intensity = read<ei::Vec3>(m_state, intensityIter);
			else
				intensity = read<ei::Vec3>(m_state, get(m_state, light, "flux")) * 4.f * ei::PI;
			intensity *= read_opt<float>(m_state, light, "scale", 1.f);

			if(world_add_point_light(lightIter->name.GetString(), util::pun<Vec3>(position),
									 util::pun<Vec3>(intensity)) == nullptr)
				throw std::runtime_error("Failed to add point light '"
										 + std::string(lightIter->name.GetString()) + "'");
		} else if(type.compare("spot") == 0) {
			// Spot light
			const ei::Vec3 position = read<ei::Vec3>(m_state, get(m_state, light, "position"));
			const ei::Vec3 direction = read<ei::Vec3>(m_state, get(m_state, light, "direction"));
			const ei::Vec3 intensity = read<ei::Vec3>(m_state, get(m_state, light, "intensity"))
				* read_opt<float>(m_state, light, "scale", 1.f);
			Radians angle;
			Radians falloffStart;
			auto angleIter = get(m_state, light, "cosWidth", false);
			if(angleIter != light.MemberEnd())
				angle = std::acos(static_cast<Radians>(Degrees(read<float>(m_state, angleIter))));
			else
				angle = static_cast<Radians>(Degrees(read<float>(m_state, get(m_state, light, "width"))));
			auto falloffIter = get(m_state, light, "cosFalloffStart", false);
			if(falloffIter != light.MemberEnd())
				falloffStart = std::acos(read<float>(m_state, falloffIter));
			else
				falloffStart = static_cast<Radians>(Degrees(read_opt<float>(m_state, light, "falloffWidth",
																			static_cast<Radians>(Degrees(angle)))));

			if(world_add_spot_light(lightIter->name.GetString(), util::pun<Vec3>(position),
									util::pun<Vec3>(direction), util::pun<Vec3>(intensity),
									angle, falloffStart) == nullptr)
				throw std::runtime_error("Failed to add spot light '"
										 + std::string(lightIter->name.GetString()) + "'");
		} else if(type.compare("directional") == 0) {
			// Directional light
			const ei::Vec3 direction = read<ei::Vec3>(m_state, get(m_state, light, "direction"));
			const ei::Vec3 radiance = read<ei::Vec3>(m_state, get(m_state, light, "radiance"))
				* read_opt<float>(m_state, light, "scale", 1.f);

			if(world_add_directional_light(lightIter->name.GetString(), util::pun<Vec3>(direction),
										   util::pun<Vec3>(radiance)) == nullptr)
				throw std::runtime_error("Failed to add directional light '"
										 + std::string(lightIter->name.GetString()) + "'");
		} else if(type.compare("envmap") == 0) {
			// Environment-mapped light
			const char* texPath = read<const char*>(m_state, get(m_state, light, "map"));
			const float scale = read_opt<float>(m_state, light, "scale", 1.f);
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
			const ei::Vec3 position = read<ei::Vec3>(m_state, get(m_state, light, "position"));
			const char* texPath = read<const char*>(m_state, get(m_state, light, "map"));
			const float scale = read_opt<float>(m_state, light, "scale", 1.f);
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

		m_state.objectNames.pop_back();
	}
}

void JsonLoader::load_materials() {
	using namespace rapidjson;
	const Value& materials = m_materials->value;
	assertObject(m_state, materials);
	m_state.current = ParserState::Level::MATERIALS;

	for(auto matIter = materials.MemberBegin(); matIter != materials.MemberEnd(); ++matIter) {
	}
}

void JsonLoader::load_scenarios() {
	using namespace rapidjson;
	const Value& scenarios = m_scenarios->value;
	assertObject(m_state, scenarios);
	m_state.current = ParserState::Level::SCENARIOS;

	for(auto scenarioIter = scenarios.MemberBegin(); scenarioIter != scenarios.MemberEnd(); ++scenarioIter) {
		const Value& scenario = scenarioIter->value;
		assertObject(m_state, scenario);
		m_state.objectNames.push_back(scenarioIter->name.GetString());

		const char* camera = read<const char*>(m_state, get(m_state, scenario, "camera"));
		ei::IVec2 resolution = read<ei::IVec2>(m_state, get(m_state, scenario, "resolution"));
		std::vector<const char*> lights;
		auto lightIter = get(m_state, scenario, "lights", false);
		std::size_t lod = read_opt<std::size_t>(m_state, scenario, "lod", 0u);

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
			assertArray(m_state, lightIter);
			m_state.objectNames.push_back("lights");
			for(SizeType i = 0u; i < lightIter->value.Size(); ++i) {
				const char* lightName = read<const char*>(m_state, lightIter->value[i]);
				if(!scenario_add_light(scenarioHdl, lightName))
					throw std::runtime_error("Failed to add light '" + std::string(lightName) + "' to scenario '"
											 + std::string(scenarioIter->name.GetString()) + "'");
			}
		}

		// Add objects
		auto objectsIter = get(m_state, scenario, "objectProperties", false);
		if(objectsIter != scenario.MemberEnd()) {
			m_state.objectNames.push_back(objectsIter->name.GetString());
			assertObject(m_state, objectsIter->value);
			for(auto objIter = objectsIter->value.MemberBegin(); objIter != objectsIter->value.MemberEnd(); ++objIter) {
				std::string_view objectName = objIter->name.GetString();
				m_state.objectNames.push_back(&objectName[0u]);
				const Value& object = objIter->value;
				assertObject(m_state, object);
				// Check for object name meta-tag
				if(std::strncmp(&objectName[0u], "[obj:", 5u) != 0)
					continue;
				std::string subName{ objectName.substr(5u, objectName.length() - 6u) };
				ObjectHdl objHdl = world_get_object(&objectName[0u]);
				if(objHdl == nullptr)
					throw std::runtime_error("Failed to find object '" + subName + "' from scenario '"
											 + std::string(scenarioIter->name.GetString()) + "'");
				// Check for LoD and masked
				auto lodIter = get(m_state, object, "lod", false);
				if(lodIter != object.MemberEnd())
					if(!scenario_set_object_lod(scenarioHdl, objHdl, read<std::size_t>(m_state, lodIter)))
						throw std::runtime_error("Failed to set LoD level of object '" + subName
												 + "' for scenario '" + std::string(scenarioIter->name.GetString())
												 + "'");
				if(object.HasMember("masked"))
					if(!scenario_mask_object(scenarioHdl, objHdl))
						throw std::runtime_error("Failed to set mask for object '" + subName
												 + "' for scenario '" + std::string(scenarioIter->name.GetString())
												 + "'");

				m_state.objectNames.pop_back();
			}
			m_state.objectNames.pop_back();
		}

		auto materialsIter = get(m_state, scenario, "materialAssignments");
		m_state.objectNames.push_back(materialsIter->name.GetString());
		assertObject(m_state, materialsIter->value);
		for(auto matIter = materialsIter->value.MemberBegin(); matIter != materialsIter->value.MemberEnd(); ++matIter) {
			std::string_view matName = matIter->name.GetString();
			m_state.objectNames.push_back(&matName[0u]);
			// Check for object name meta-tag
			if(std::strncmp(&matName[0u], "[mat:", 5u) != 0)
				continue;
			std::string subName{ matName.substr(5u, matName.length() - 6u) };
			const std::string_view inScenName = read<const char*>(m_state, matIter->value);
			// TODO: create material association
			// TODO: check if all materials were associated
		}
	}
}

void JsonLoader::load_file() {
	using namespace rapidjson;

	this->clear_state();
	logInfo("[", FUNCTION_NAME, "] Parsing scene file '", m_filePath.string(), "'");

	// JSON text
	m_jsonString = read_file(m_filePath);

	Document document;
	// Parse and check for errors
	ParseResult res = document.Parse(m_jsonString.c_str());
	if(res.IsError()) {
		// Walk the string and determine line number
		std::stringstream ss(m_jsonString);
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
	assertObject(m_state, document);
	// Version
	auto versionIter = get(m_state, document, "version", false);
	if(versionIter == document.MemberEnd()) {
		logWarning("[", FUNCTION_NAME, "] Scene file: no version specified (current one assumed)");
	} else {
		m_version = read<const char*>(m_state, versionIter);
		if(m_version.compare(FILE_VERSION) != 0)
			logWarning("[", FUNCTION_NAME, "] Scene file: version mismatch (",
					   m_version, "(file) vs ", FILE_VERSION, "(current))");
	}
	// Binary file path
	m_binaryFile = read<const char*>(m_state, get(m_state, document, "binary"));
	if(m_binaryFile.empty()) {
		logError("[", FUNCTION_NAME, "] Scene file: has an empty binary file path");
		return;
	}
	// Make the file path absolute
	if(m_binaryFile.is_relative())
		m_binaryFile = fs::canonical(m_filePath.parent_path() / m_binaryFile);
	if(!fs::exists(m_binaryFile)) {
		logError("[", FUNCTION_NAME, "] Scene file: specifies a binary file that doesn't exist ('",
				 m_binaryFile.string(), "'");
		return;
	}
	// Default scenario
	m_defaultScenario = read_opt<const char*>(m_state, document, "defaultScenario", "");

	m_scenarios = get(m_state, document, "scenarios");
	m_cameras = get(m_state, document, "cameras");
	m_lights = get(m_state, document, "lights");
	m_materials = get(m_state, document, "materials");

	// First parse binary file
	binary::BinaryLoader binLoader{ m_binaryFile };
	// Choose first one in JSON - no guarantees
	if(m_defaultScenario.empty())
		m_defaultScenario = m_scenarios->value.MemberBegin()->name.GetString();
	// Partially parse the default scenario
	m_state.current = ParserState::Level::SCENARIOS;
	const Value& defScen = get(m_state, m_scenarios->value, &m_defaultScenario[0u])->value;
	const u64 defaultGlobalLod = read_opt<u64>(m_state, defScen, "lod", 0u);
	std::unordered_map<std::string_view, u64> defaultLocalLods;
	auto objPropsIter = get(m_state, defScen, "objectProperties", false);
	if(objPropsIter != defScen.MemberEnd()) {
		m_state.objectNames.push_back(&m_defaultScenario[0u]);
		m_state.objectNames.push_back("objectProperties");
		for(auto propIter = objPropsIter->value.MemberBegin(); propIter != objPropsIter->value.MemberEnd(); ++propIter) {
			std::string_view objectName = propIter->name.GetString();
			m_state.objectNames.push_back(&objectName[0u]);
			const Value& object = propIter->value;
			assertObject(m_state, object);
			// Check for object name meta-tag
			if(std::strncmp(&objectName[0u], "[obj:", 5u) != 0)
				continue;
			std::string_view subName = objectName.substr(5u, objectName.length() - 6u);
			auto lodIter = get(m_state, object, "lod", false);
			if(lodIter != object.MemberEnd())
				defaultLocalLods.insert({ subName, read<u64>(m_state, lodIter) });
		}
	}
	// TODO
	binLoader.load_file(defaultGlobalLod, defaultLocalLods);

	// Cameras
	m_state.current = ParserState::Level::ROOT;
	load_cameras();
	// Lights
	m_state.current = ParserState::Level::ROOT;
	load_lights();
	// Materials
	m_state.current = ParserState::Level::ROOT;
	load_materials();
	// Scenarios
	m_state.current = ParserState::Level::ROOT;
	load_scenarios();


	// TODO: parse binary file
}


} // namespace loader::json