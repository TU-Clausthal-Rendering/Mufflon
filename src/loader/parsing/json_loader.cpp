#include "json_loader.hpp"
#include "binary.hpp"
#include "profiler/cpu_profiler.hpp"
#include "util/log.hpp"
#include "util/punning.hpp"
#include "util/int_types.hpp"
#include "util/degrad.hpp"
#include "core/export/interface.h"
#include <rapidjson/document.h>
#include <rapidjson/error/en.h>
#include <fstream>
#include <sstream>

namespace loader::json {

using namespace mufflon;

namespace {

// Reads a file completely and returns the string containing all bytes
std::string read_file(fs::path path) {
	auto scope = Profiler::instance().start<CpuProfileState>("JSON read_file", ProfileLevel::HIGH);
	const std::uintmax_t fileSize = fs::file_size(path);
	std::string fileString;
	fileString.resize(fileSize);

	std::ifstream file(path, std::ios::binary);
	file.read(&fileString[0u], fileSize);
	if(file.gcount() != fileSize)
		logWarning("[read_file] File '", path.string(), "'not read completely");
	// Finalize the string
	fileString[file.gcount()] = '\0';
	return fileString;
}

} // namespace

JsonException::JsonException(const std::string& str, rapidjson::ParseResult res) :
	m_error("Parser error: " + std::string(rapidjson::GetParseError_En(res.Code()))
			 + " at offset " + std::to_string(res.Offset()))  {

		// Walk the string and determine line number
	std::stringstream ss(str);
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

	m_error += " (line " + std::to_string(currLine) + ')';
}

void JsonLoader::clear_state() {
	m_jsonString.clear();
	m_state.reset();
	m_binaryFile.clear();
	m_materialMap.clear();
}

TextureHdl JsonLoader::load_texture(const char* name) {
	auto scope = Profiler::instance().start<CpuProfileState>("JsonLoader::load_texture", ProfileLevel::HIGH);
	// Make the path relative to the file
	fs::path path(name);
	if(!path.is_absolute())
		path = fs::canonical(m_filePath.parent_path() / name);
	if(!fs::exists(path))
		throw std::runtime_error("Cannot find texture file '" + path.string() + '\'');
	TextureHdl tex = world_add_texture(path.string().c_str(), TextureSampling::SAMPLING_LINEAR);
	if(tex == nullptr)
		throw std::runtime_error("Failed to load texture '" + std::string(name) + "'");
	return tex;
}

MaterialParams* JsonLoader::load_material(rapidjson::Value::ConstMemberIterator matIter) {
	auto scope = Profiler::instance().start<CpuProfileState>("JsonLoader::load_material", ProfileLevel::HIGH);
	using namespace rapidjson;
	MaterialParams* mat = new MaterialParams{};

	try {
		const Value& material = matIter->value;
		assertObject(m_state, material);
		const char* materialName = matIter->name.GetString();
		m_state.objectNames.push_back(materialName);

		// Read the outer medium
		if(auto outerIter = get(m_state, material, "outerMedium", false); outerIter != material.MemberEnd()) {
			// Parse the outer medium of the material
			m_state.objectNames.push_back(outerIter->name.GetString());
			const Value& outerMedium = outerIter->value;
			mat->outerMedium.absorption = util::pun<Vec3>(read<ei::Vec3>(m_state, get(m_state, outerMedium, "absorption")));
			auto refractIter = get(m_state, outerMedium, "refractionIndex");
			if(refractIter->value.IsArray()) {
				mat->outerMedium.refractionIndex = util::pun<Vec2>(read<ei::Vec2>(m_state, refractIter));
			} else {
				mat->outerMedium.refractionIndex = Vec2{ read<float>(m_state, refractIter), 0.0f };
			}
			m_state.objectNames.pop_back();
		} else {
			mat->outerMedium.absorption = Vec3{ 0.0f };
			mat->outerMedium.refractionIndex = Vec2{ 1.0f, 0.0f };
		}

		std::string_view type = read<const char*>(m_state, get(m_state, material, "type"));
		if(type.compare("lambert") == 0) {
			// Lambert material
			mat->innerType = MaterialParamType::MATERIAL_LAMBERT;
			auto albedoIter = get(m_state, material, "albedo");
			MaterialHdl hdl = nullptr;
			if(albedoIter->value.IsArray()) {
				ei::Vec3 albedo = read<ei::Vec3>(m_state, albedoIter);
				mat->inner.lambert.albedo = world_add_texture_value(reinterpret_cast<float*>(&albedo), 3, TextureSampling::SAMPLING_NEAREST);
			} else if(albedoIter->value.IsString()) {
				mat->inner.lambert.albedo = load_texture(read<const char*>(m_state, albedoIter));
			} else
				throw std::runtime_error("Invalid type for albedo.");

		} else if(type.compare("torrance") == 0) {
			// Torrance material
			mat->innerType = MaterialParamType::MATERIAL_TORRANCE;
			std::string_view ndf = read<const char*>(m_state, get(m_state, material, "ndf"));
			if(ndf.compare("BS") == 0)
				mat->inner.torrance.ndf = NormalDistFunction::NDF_BECKMANN;
			else if(ndf.compare("GGC") == 0)
				mat->inner.torrance.ndf = NormalDistFunction::NDF_GGX;
			else if(ndf.compare("GGC") == 0)
				mat->inner.torrance.ndf = NormalDistFunction::NDF_GGX;
			else
				throw std::runtime_error("Unknown normal distribution function '" + std::string(ndf) + "'");
			auto roughnessIter = get(m_state, material, "roughness");
			if(roughnessIter->value.IsArray()) {
				ei::Vec3 xyr = read<ei::Vec3>(m_state, roughnessIter);
				mat->inner.torrance.roughness = world_add_texture_value(reinterpret_cast<float*>(&xyr), 3, TextureSampling::SAMPLING_NEAREST);
			} else if(roughnessIter->value.IsNumber()) {
				float alpha = read<float>(m_state, roughnessIter);
				mat->inner.torrance.roughness = world_add_texture_value(&alpha, 1, TextureSampling::SAMPLING_NEAREST);
			} else if(roughnessIter->value.IsString()) {
				mat->inner.torrance.roughness = load_texture(read<const char*>(m_state, roughnessIter));
			} else
				throw std::runtime_error("Invalid type for roughness.");
			auto albedoIter = get(m_state, material, "albedo");
			if(albedoIter->value.IsArray()) {
				ei::Vec3 albedo = read<ei::Vec3>(m_state, albedoIter);
				mat->inner.torrance.albedo = world_add_texture_value(reinterpret_cast<float*>(&albedo), 3, TextureSampling::SAMPLING_NEAREST);
			} else if(albedoIter->value.IsString()) {
				mat->inner.torrance.albedo = load_texture(read<const char*>(m_state, albedoIter));
			} else
				throw std::runtime_error("Invalid type for albedo.");

		} else if(type.compare("walter") == 0) {
			// Walter material
			mat->innerType = MaterialParamType::MATERIAL_WALTER;
			std::string_view ndf = read<const char*>(m_state, get(m_state, material, "ndf"));
			if(ndf.compare("BS") == 0)
				mat->inner.walter.ndf = NormalDistFunction::NDF_BECKMANN;
			else if(ndf.compare("GGC") == 0)
				mat->inner.walter.ndf = NormalDistFunction::NDF_GGX;
			else if(ndf.compare("GGC") == 0)
				mat->inner.walter.ndf = NormalDistFunction::NDF_GGX;
			else
				throw std::runtime_error("Unknown normal distribution function '" + std::string(ndf) + "'");
			auto roughnessIter = get(m_state, material, "roughness");
			mat->inner.walter.absorption = util::pun<Vec3>(read<ei::Vec3>(m_state, get(m_state, material, "absorption")));
			if(roughnessIter->value.IsNumber()) {
				float alpha = read<float>(m_state, roughnessIter);
				mat->inner.walter.roughness = world_add_texture_value(&alpha, 1, TextureSampling::SAMPLING_NEAREST);
			} else if(roughnessIter->value.IsString()) {
				mat->inner.walter.roughness = load_texture(read<const char*>(m_state, roughnessIter));
			} else if(roughnessIter->value.IsArray()) {
				ei::Vec3 xyr = read<ei::Vec3>(m_state, roughnessIter);
				mat->inner.walter.roughness = world_add_texture_value(reinterpret_cast<float*>(&xyr), 3, TextureSampling::SAMPLING_NEAREST);
			} else
				throw std::runtime_error("Invalid type for roughness");

		} else if(type.compare("emissive") == 0) {
			// Emissive material
			mat->innerType = MaterialParamType::MATERIAL_EMISSIVE;
			mat->inner.emissive.scale = read_opt<float>(m_state, material, "scale", 1.f);
			auto radianceIter = get(m_state, material, "radiance");
			if(radianceIter->value.IsArray()) {
				ei::Vec3 rgb = read<ei::Vec3>(m_state, radianceIter);
				mat->inner.emissive.radiance = world_add_texture_value(reinterpret_cast<float*>(&rgb), 3, TextureSampling::SAMPLING_NEAREST);
			} else if(radianceIter->value.IsString()) {
				mat->inner.emissive.radiance = load_texture(read<const char*>(m_state, radianceIter));
			} else
				throw std::runtime_error("Invalid type for radiance");

		} else if(type.compare("orennayar") == 0) {
			// Oren-Nayar material
			mat->innerType = MaterialParamType::MATERIAL_ORENNAYAR;
			mat->inner.orennayar.roughness = read_opt<float>(m_state, material, "roughness", 1.f);
			auto albedoIter = get(m_state, material, "albedo");
			if(albedoIter->value.IsArray()) {
				ei::Vec3 rgb = read<ei::Vec3>(m_state, albedoIter);
				mat->inner.orennayar.albedo = world_add_texture_value(reinterpret_cast<float*>(&rgb), 3, TextureSampling::SAMPLING_NEAREST);
			} else if(albedoIter->value.IsString()) {
				mat->inner.orennayar.albedo = load_texture(read<const char*>(m_state, albedoIter));
			} else
				throw std::runtime_error("Invalid type for albedo");

		} else if(type.compare("blend") == 0) {
			// Blend material
			mat->innerType = MaterialParamType::MATERIAL_BLEND;
			mat->inner.blend.a.factor = read<float>(m_state, get(m_state, material, "factorA"));
			mat->inner.blend.b.factor = read<float>(m_state, get(m_state, material, "factorB"));
			mat->inner.blend.a.mat = load_material(get(m_state, material, "layerA"));
			mat->inner.blend.b.mat = load_material(get(m_state, material, "layerB"));
		} else if(type.compare("fresnel") == 0) {
			// Fresnel material
			mat->innerType = MaterialParamType::MATERIAL_FRESNEL;
			auto refrIter = get(m_state, material, "refractionIndex");
			if(refrIter->value.IsNumber()) {
				mat->inner.fresnel.refractionIndex = Vec2{ read<float>(m_state, refrIter), 0.0f };
			} else if(refrIter->value.IsArray()) {
				mat->inner.fresnel.refractionIndex = util::pun<Vec2>(read<ei::Vec2>(m_state, refrIter));
			} else {
				throw std::runtime_error("Invalid type for refraction index");
			}
			mat->inner.fresnel.a = load_material(get(m_state, material, "layerA"));
			mat->inner.fresnel.b = load_material(get(m_state, material, "layerB"));
		} else if(type.compare("glass") == 0) {
			// TODO: glass material
		} else if(type.compare("opaque") == 0) {
			// TODO: opaque material
		} else {
			throw std::runtime_error("Unknown material type '" + std::string(type) + "'");
		}
	} catch(const std::exception&) {
		free_material(mat);
	}

	m_state.objectNames.pop_back();
	return mat;
}

void JsonLoader::free_material(MaterialParams* mat) {
	switch(mat->innerType) {
		case MATERIAL_LAMBERT:
		case MATERIAL_TORRANCE:
		case MATERIAL_WALTER:
		case MATERIAL_EMISSIVE:
		case MATERIAL_ORENNAYAR:
			if(mat != nullptr)
				delete mat;
			return;
		case MATERIAL_BLEND:
			if(mat->inner.blend.a.mat != nullptr)
				free_material(mat->inner.blend.a.mat);
			if(mat->inner.blend.b.mat != nullptr)
				free_material(mat->inner.blend.b.mat);
			return;
		case MATERIAL_FRESNEL:
			if(mat->inner.fresnel.a != nullptr)
			free_material(mat->inner.fresnel.a);
			if(mat->inner.blend.a.mat != nullptr)
				if(mat->inner.fresnel.b != nullptr)
			free_material(mat->inner.fresnel.b);
			return;
		default: return;
	}
}

void JsonLoader::load_cameras(const ei::Box& aabb) {
	auto scope = Profiler::instance().start<CpuProfileState>("JsonLoader::load_cameras", ProfileLevel::HIGH);
	using namespace rapidjson;
	const Value& cameras = m_cameras->value;
	assertObject(m_state, cameras);
	m_state.current = ParserState::Level::CAMERAS;

	for(auto cameraIter = cameras.MemberBegin(); cameraIter != cameras.MemberEnd(); ++cameraIter) {
		const Value& camera = cameraIter->value;
		assertObject(m_state, camera);
		m_state.objectNames.push_back(cameraIter->name.GetString());

		// Read common values
		// Default camera planes depend on scene bounding box size
		const float sceneDiag = ei::abs(ei::len(aabb.max - aabb.min));
		const float near = read_opt<float>(m_state, camera, "near", DEFAULT_NEAR_PLANE * sceneDiag);
		const float far = read_opt<float>(m_state, camera, "far", DEFAULT_FAR_PLANE * sceneDiag);
		std::string_view type = read<const char*>(m_state, get(m_state, camera, "type"));
		std::vector<ei::Vec3> camPath;
		std::vector<ei::Vec3> camViewDir;
		std::vector<ei::Vec3> camUp;
		read(m_state, get(m_state, camera, "path"), camPath);
		read(m_state, get(m_state, camera, "viewDir"), camViewDir, camPath.size());
		if(auto upIter = get(m_state, camera, "up", false); upIter != camera.MemberEnd()) {
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
				logWarning("[JsonLoader::load_cameras] Scene file: camera paths are not supported yet");
			if(world_add_pinhole_camera(cameraIter->name.GetString(), util::pun<Vec3>(camPath[0u]),
										util::pun<Vec3>(camViewDir[0u]), util::pun<Vec3>(camUp[0u]),
										near, far, static_cast<Radians>(Degrees(fovDegree))) == nullptr)
				throw std::runtime_error("Failed to add pinhole camera");
		} else if(type.compare("focus") == 0) {
			const float focalLength = read<float>(m_state, get(m_state, camera, "focalLength")) / 1000.f;
			const float focusDistance = read<float>(m_state, get(m_state, camera, "focusDistance"));
			const float sensorHeight = read<float>(m_state, get(m_state, camera, "chipHeight")) / 1000.f;
			const float lensRadius = read<float>(m_state, get(m_state, camera, "aperture")) / (2.f * focalLength);
			if(world_add_focus_camera(cameraIter->name.GetString(), util::pun<Vec3>(camPath[0u]),
									   util::pun<Vec3>(camViewDir[0u]), util::pun<Vec3>(camUp[0u]),
									   near, far, focalLength, focusDistance, lensRadius, sensorHeight) == nullptr)
				throw std::runtime_error("Failed to add focus camera");
		} else if(type.compare("ortho") == 0) {
			// TODO: Orthogonal camera
			logWarning("[JsonLoader::load_cameras] Scene file: Focus cameras are not supported yet");
		} else {
			logWarning("[JsonLoader::load_cameras] Scene file: camera object '",
					   cameraIter->name.GetString(), "' has unknown type '", type, "'");
		}

		m_state.objectNames.pop_back();
	}
}

void JsonLoader::load_lights() {
	auto scope = Profiler::instance().start<CpuProfileState>("JsonLoader::load_lights", ProfileLevel::HIGH);
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
			if(auto intensityIter = get(m_state, light, "intensity", false); intensityIter != light.MemberEnd())
				intensity = read<ei::Vec3>(m_state, intensityIter);
			else
				intensity = read<ei::Vec3>(m_state, get(m_state, light, "flux")) * 4.f * ei::PI;
			intensity *= read_opt<float>(m_state, light, "scale", 1.f);

			if(world_add_point_light(lightIter->name.GetString(), util::pun<Vec3>(position),
									 util::pun<Vec3>(intensity)) == nullptr)
				throw std::runtime_error("Failed to add point light");
		} else if(type.compare("spot") == 0) {
			// Spot light
			const ei::Vec3 position = read<ei::Vec3>(m_state, get(m_state, light, "position"));
			const ei::Vec3 direction = read<ei::Vec3>(m_state, get(m_state, light, "direction"));
			const ei::Vec3 intensity = read<ei::Vec3>(m_state, get(m_state, light, "intensity"))
				* read_opt<float>(m_state, light, "scale", 1.f);
			Radians angle;
			Radians falloffStart;
			if(auto angleIter = get(m_state, light, "cosWidth", false); angleIter != light.MemberEnd())
				angle = std::acos(static_cast<Radians>(Degrees(read<float>(m_state, angleIter))));
			else
				angle = static_cast<Radians>(Degrees(read<float>(m_state, get(m_state, light, "width"))));
			if(auto falloffIter = get(m_state, light, "cosFalloffStart", false);  falloffIter != light.MemberEnd())
				falloffStart = std::acos(read<float>(m_state, falloffIter));
			else
				falloffStart = static_cast<Radians>(Degrees(read_opt<float>(m_state, light, "falloffWidth",
																			static_cast<Radians>(Degrees(angle)))));

			if(world_add_spot_light(lightIter->name.GetString(), util::pun<Vec3>(position),
									util::pun<Vec3>(direction), util::pun<Vec3>(intensity),
									angle, falloffStart) == nullptr)
				throw std::runtime_error("Failed to add spot light");
		} else if(type.compare("directional") == 0) {
			// Directional light
			const ei::Vec3 direction = read<ei::Vec3>(m_state, get(m_state, light, "direction"));
			const ei::Vec3 radiance = read<ei::Vec3>(m_state, get(m_state, light, "radiance"))
				* read_opt<float>(m_state, light, "scale", 1.f);

			if(world_add_directional_light(lightIter->name.GetString(), util::pun<Vec3>(direction),
										   util::pun<Vec3>(radiance)) == nullptr)
				throw std::runtime_error("Failed to add directional light");
		} else if(type.compare("envmap") == 0) {
			// Environment-mapped light
			TextureHdl texture = load_texture(read<const char*>(m_state, get(m_state, light, "map")));
			const float scale = read_opt<float>(m_state, light, "scale", 1.f);
			// TODO: incorporate scale

			if(world_add_envmap_light(lightIter->name.GetString(), texture) == nullptr)
				throw std::runtime_error("Failed to add directional light");
		} else if(type.compare("goniometric") == 0) {
			// TODO: Goniometric light
			const ei::Vec3 position = read<ei::Vec3>(m_state, get(m_state, light, "position"));
			TextureHdl texture = load_texture(read<const char*>(m_state, get(m_state, light, "map")));
			const float scale = read_opt<float>(m_state, light, "scale", 1.f);
			// TODO: incorporate scale

			logWarning("[JsonLoader::load_lights] Scene file: Goniometric lights are not supported yet");
		} else {
			logWarning("[JsonLoader::load_cameras] Scene file: light object '",
					   lightIter->name.GetString(), "' has unknown type '", type, "'");
		}

		m_state.objectNames.pop_back();
	}
}

void JsonLoader::load_materials() {
	auto scope = Profiler::instance().start<CpuProfileState>("JsonLoader::load_materials", ProfileLevel::HIGH);
	using namespace rapidjson;
	const Value& materials = m_materials->value;
	assertObject(m_state, materials);
	m_state.current = ParserState::Level::MATERIALS;

	for(auto matIter = materials.MemberBegin(); matIter != materials.MemberEnd(); ++matIter) {
		MaterialParams* mat = load_material(matIter);
		if(mat != nullptr) {
			auto hdl = world_add_material(matIter->name.GetString(), mat);
			free_material(mat);
			if(hdl == nullptr)
				throw std::runtime_error("Failed to add material to world");
			m_materialMap.emplace(matIter->name.GetString(), hdl);
		} else {
			throw std::runtime_error("Failed to load material '" + std::string(matIter->name.GetString()) + "'");
		}
	}
}

void JsonLoader::load_scenarios(const std::vector<std::string>& binMatNames) {
	auto scope = Profiler::instance().start<CpuProfileState>("JsonLoader::load_scenarios", ProfileLevel::HIGH);
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
			throw std::runtime_error("Failed to create scenario");

		if(!scenario_set_camera(scenarioHdl, camHdl))
			throw std::runtime_error("Failed to set camera '" + std::string(camera) + "'");
		if(!scenario_set_resolution(scenarioHdl, resolution.x, resolution.y))
			throw std::runtime_error("Failed to set resolution '" + std::to_string(resolution.x) + "x"
									 + std::to_string(resolution.y) + "'");
		if(!scenario_set_global_lod_level(scenarioHdl, lod))
			throw std::runtime_error("Failed to set LoD " + std::to_string(lod));

		// Add lights
		if(lightIter != scenario.MemberEnd()) {
			assertArray(m_state, lightIter);
			m_state.objectNames.push_back("lights");
			for(SizeType i = 0u; i < lightIter->value.Size(); ++i) {
				const char* lightName = read<const char*>(m_state, lightIter->value[i]);
				if(!scenario_add_light(scenarioHdl, lightName))
					throw std::runtime_error("Failed to add light '" + std::string(lightName) + "'");
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
					throw std::runtime_error("Failed to find object '" + subName + "'");
				// Check for LoD and masked
				if(auto lodIter = get(m_state, object, "lod", false); lodIter != object.MemberEnd())
					if(!scenario_set_object_lod(scenarioHdl, objHdl, read<std::size_t>(m_state, lodIter)))
						throw std::runtime_error("Failed to set LoD level of object '" + subName + "'");
				if(object.HasMember("masked"))
					if(!scenario_mask_object(scenarioHdl, objHdl))
						throw std::runtime_error("Failed to set mask for object '" + subName + "'");

				m_state.objectNames.pop_back();
			}
			m_state.objectNames.pop_back();
		}

		// Associate binary with JSON material names
		auto materialsIter = get(m_state, scenario, "materialAssignments");
		m_state.objectNames.push_back(materialsIter->name.GetString());
		assertObject(m_state, materialsIter->value);
		for(const std::string& binName : binMatNames) {
			// The binary names from the loader already wrap the name in the desired format
			std::string_view matName = read<const char*>(m_state, get(m_state, materialsIter->value,
																 binName.c_str()));
			// Offset to remove the [mat:...] wrapping
			MatIdx slot = scenario_declare_material_slot(scenarioHdl, &binName.c_str()[5u], binName.length() - 6u);
			if(slot == INVALID_MATERIAL)
				throw std::runtime_error("Failed to declare material slot");
			auto matHdl = m_materialMap.find(matName);
			if(matHdl == m_materialMap.cend())
				throw std::runtime_error("Unknown material name '" + std::string(matName) + "'in association");
			if(!scenario_assign_material(scenarioHdl, slot, matHdl->second))
				throw std::runtime_error("Failed to associate material '" + matHdl->first + "'");
		}
	}
}

void JsonLoader::load_file() {
	using namespace rapidjson;
	auto scope = Profiler::instance().start<CpuProfileState>("JsonLoader::load_file");

	this->clear_state();
	logInfo("[JsonLoader::load_file] Parsing scene file '", m_filePath.string(), "'");

	// JSON text
	m_jsonString = read_file(m_filePath);

	Document document;
	// Parse and check for errors
	ParseResult res = document.Parse(m_jsonString.c_str());
	if(res.IsError())
		throw JsonException(m_jsonString, res);

	// Parse our file specification
	assertObject(m_state, document);
	// Version
	auto versionIter = get(m_state, document, "version", false);
	if(versionIter == document.MemberEnd()) {
		logWarning("[JsonLoader::load_file] Scene file: no version specified (current one assumed)");
	} else {
		m_version = read<const char*>(m_state, versionIter);
		if(m_version.compare(FILE_VERSION) != 0)
			logWarning("[JsonLoader::load_file] Scene file: version mismatch (",
					   m_version, "(file) vs ", FILE_VERSION, "(current))");
	}
	// Binary file path
	m_binaryFile = read<const char*>(m_state, get(m_state, document, "binary"));
	if(m_binaryFile.empty()) {
		logError("[JsonLoader::load_file] Scene file: has an empty binary file path");
		return;
	}
	// Make the file path absolute
	if(m_binaryFile.is_relative())
		m_binaryFile = fs::canonical(m_filePath.parent_path() / m_binaryFile);
	if(!fs::exists(m_binaryFile)) {
		logError("[JsonLoader::load_file] Scene file: specifies a binary file that doesn't exist ('",
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
	// Load the binary file before we load the rest of the JSON
	binLoader.load_file(defaultGlobalLod, defaultLocalLods);

	try {
		// Cameras
		m_state.current = ParserState::Level::ROOT;
		load_cameras(binLoader.get_bounding_box());
		// Lights
		m_state.current = ParserState::Level::ROOT;
		load_lights();
		// Materials
		m_state.current = ParserState::Level::ROOT;
		load_materials();
		// Scenarios
		m_state.current = ParserState::Level::ROOT;
		load_scenarios(binLoader.get_material_names());
		// Load the default scenario
		m_state.current = ParserState::Level::ROOT;
		ScenarioHdl defScenHdl = world_find_scenario(&m_defaultScenario[0u]);
		if(defScenHdl == nullptr)
			throw std::runtime_error("Cannot find the default scenario '" + std::string(m_defaultScenario) + '\'');

		auto scope = Profiler::instance().start<CpuProfileState>("JsonLoader::load_file - load default scenario", ProfileLevel::LOW);
		if(!world_load_scenario(defScenHdl))
			throw std::runtime_error("Cannot load the default scenario '" + std::string(m_defaultScenario) + '\'');
	} catch(const std::runtime_error& e) {
		throw std::runtime_error(m_state.get_parser_level() + ": " + e.what());
	}
}


} // namespace loader::json