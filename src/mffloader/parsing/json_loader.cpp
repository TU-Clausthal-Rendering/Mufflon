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

namespace mff_loader::json {

using namespace mufflon;

namespace {

// Reads a file completely and returns the string containing all bytes
std::string read_file(fs::path path) {
	auto scope = Profiler::instance().start<CpuProfileState>("JSON read_file", ProfileLevel::HIGH);
	logPedantic("[read_file] Loading JSON file '", path.string(), "' into RAM");
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

TextureHdl JsonLoader::load_texture(const char* name, TextureSampling sampling) {
	auto scope = Profiler::instance().start<CpuProfileState>("JsonLoader::load_texture", ProfileLevel::HIGH);
	logPedantic("[JsonLoader::load_texture] Loading texture '", name, "'");
	// Make the path relative to the file
	fs::path path(name);
	if (!path.is_absolute())
		path = m_filePath.parent_path() / name;
	if (!fs::exists(path))
		throw std::runtime_error("Cannot find texture file '" + path.string() + "'");
	path = fs::canonical(path);
	TextureHdl tex = world_add_texture(path.string().c_str(), sampling);
	if(tex == nullptr)
		throw std::runtime_error("Failed to load texture '" + std::string(name) + "'");
	return tex;
}

MaterialParams* JsonLoader::load_material(rapidjson::Value::ConstMemberIterator matIter) {
	auto scope = Profiler::instance().start<CpuProfileState>("JsonLoader::load_material", ProfileLevel::HIGH);
	logPedantic("[JsonLoader::load_material] Loading material '", matIter->name.GetString(), "'");
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
			auto refractIter = get(m_state, outerMedium, "ior");
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

		StringView type = read<const char*>(m_state, get(m_state, material, "type"));
		if(type.compare("lambert") == 0) {
			// Lambert material
			mat->innerType = MaterialParamType::MATERIAL_LAMBERT;
			auto albedoIter = get(m_state, material, "albedo");
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
			StringView ndf = read<const char*>(m_state, get(m_state, material, "ndf"));
			if(ndf.compare("BS") == 0)
				mat->inner.torrance.ndf = NormalDistFunction::NDF_BECKMANN;
			else if(ndf.compare("GGX") == 0)
				mat->inner.torrance.ndf = NormalDistFunction::NDF_GGX;
			else if(ndf.compare("Cos") == 0)
				mat->inner.torrance.ndf = NormalDistFunction::NDF_COSINE;
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
			StringView ndf = read<const char*>(m_state, get(m_state, material, "ndf"));
			if(ndf.compare("BS") == 0)
				mat->inner.walter.ndf = NormalDistFunction::NDF_BECKMANN;
			else if(ndf.compare("GGX") == 0)
				mat->inner.walter.ndf = NormalDistFunction::NDF_GGX;
			else if(ndf.compare("Cos") == 0)
				mat->inner.walter.ndf = NormalDistFunction::NDF_COSINE;
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

			mat->inner.walter.refractionIndex = read<float>(m_state, get(m_state, material, "ior"));

		} else if(type.compare("emissive") == 0) {
			// Emissive material
			mat->innerType = MaterialParamType::MATERIAL_EMISSIVE;
			mat->inner.emissive.scale = util::pun<Vec3>(read_opt<ei::Vec3>(m_state, material, "scale", ei::Vec3{1.0f, 1.0f, 1.0f}));
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
			auto refrIter = get(m_state, material, "ior");
			if(refrIter->value.IsNumber()) {
				mat->inner.fresnel.refractionIndex = Vec2{ read<float>(m_state, refrIter), 0.0f };
			} else if(refrIter->value.IsArray()) {
				mat->inner.fresnel.refractionIndex = util::pun<Vec2>(read<ei::Vec2>(m_state, refrIter));
			} else {
				throw std::runtime_error("Invalid type for refraction index");
			}
			mat->inner.fresnel.a = load_material(get(m_state, material, "layerReflection"));
			mat->inner.fresnel.b = load_material(get(m_state, material, "layerRefraction"));
		} else if(type.compare("glass") == 0) {
			// TODO: glass material
		} else if(type.compare("opaque") == 0) {
			// TODO: opaque material
		} else {
			throw std::runtime_error("Unknown material type '" + std::string(type) + "'");
		}
	} catch(const std::exception&) {
		free_material(mat);
		throw;
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

bool JsonLoader::load_cameras(const ei::Box& aabb) {
	auto scope = Profiler::instance().start<CpuProfileState>("JsonLoader::load_cameras", ProfileLevel::HIGH);
	using namespace rapidjson;
	const Value& cameras = m_cameras->value;
	assertObject(m_state, cameras);
	m_state.current = ParserState::Level::CAMERAS;

	for(auto cameraIter = cameras.MemberBegin(); cameraIter != cameras.MemberEnd(); ++cameraIter) {
		logPedantic("[JsonLoader::load_cameras] Loading camera '", cameraIter->name.GetString(), "'");
		if(m_abort)
			return false;
		const Value& camera = cameraIter->value;
		assertObject(m_state, camera);
		m_state.objectNames.push_back(cameraIter->name.GetString());

		// Read common values
		// Default camera planes depend on scene bounding box size
		const float sceneDiag = ei::abs(ei::len(aabb.max - aabb.min));
		const float near = read_opt<float>(m_state, camera, "near", DEFAULT_NEAR_PLANE * sceneDiag);
		const float far = read_opt<float>(m_state, camera, "far", DEFAULT_FAR_PLANE * sceneDiag);
		StringView type = read<const char*>(m_state, get(m_state, camera, "type"));
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
			const float focalLength = read_opt<float>(m_state, camera, "focalLength", 35.f) / 1000.f;
			const float focusDistance = read<float>(m_state, get(m_state, camera, "focusDistance"));
			const float sensorHeight = read_opt<float>(m_state, camera, "chipHeight", 24.f) / 1000.f;
			const float lensRadius = (focalLength / read_opt<float>(m_state, camera, "aperture", focalLength)) / 2.f;
			if(world_add_focus_camera(cameraIter->name.GetString(), util::pun<Vec3>(camPath[0u]),
									   util::pun<Vec3>(camViewDir[0u]), util::pun<Vec3>(camUp[0u]),
									   near, far, focalLength, focusDistance, lensRadius, sensorHeight) == nullptr)
				throw std::runtime_error("Failed to add focus camera");
		} else if(type.compare("ortho") == 0) {
			// TODO: Orthogonal camera
			logWarning("[JsonLoader::load_cameras] Scene file: Orthogonal cameras are not supported yet");
		} else {
			logWarning("[JsonLoader::load_cameras] Scene file: camera object '",
					   cameraIter->name.GetString(), "' has unknown type '", type, "'");
		}

		m_state.objectNames.pop_back();
	}
	return true;
}

bool JsonLoader::load_lights() {
	auto scope = Profiler::instance().start<CpuProfileState>("JsonLoader::load_lights", ProfileLevel::HIGH);
	using namespace rapidjson;
	const Value& lights = m_lights->value;
	assertObject(m_state, lights);
	m_state.current = ParserState::Level::LIGHTS;

	for(auto lightIter = lights.MemberBegin(); lightIter != lights.MemberEnd(); ++lightIter) {
		logPedantic("[JsonLoader::load_lights] Loading light '", lightIter->name.GetString(), "'");
		if(m_abort)
			return false;
		const Value& light = lightIter->value;
		assertObject(m_state, light);
		m_state.objectNames.push_back(lightIter->name.GetString());

		// Read common values (aka the type only)
		StringView type = read<const char*>(m_state, get(m_state, light, "type"));
		if(type.compare("point") == 0) {
			// Point light
			const ei::Vec3 position = read<ei::Vec3>(m_state, get(m_state, light, "position"));
			ei::Vec3 intensity;
			if(auto intensityIter = get(m_state, light, "intensity", false); intensityIter != light.MemberEnd())
				intensity = read<ei::Vec3>(m_state, intensityIter);
			else
				intensity = read<ei::Vec3>(m_state, get(m_state, light, "flux")) / (4.0f * ei::PI);
			intensity *= read_opt<float>(m_state, light, "scale", 1.f);

			if(auto hdl = world_add_light(lightIter->name.GetString(), LIGHT_POINT); hdl.type == LIGHT_POINT) {
				world_set_point_light_position(hdl, util::pun<Vec3>(position));
				world_set_point_light_intensity(hdl, util::pun<Vec3>(intensity));
				m_lightMap.emplace(lightIter->name.GetString(), hdl);
			} else throw std::runtime_error("Failed to add point light");
		} else if(type.compare("spot") == 0) {
			// Spot light
			const ei::Vec3 position = read<ei::Vec3>(m_state, get(m_state, light, "position"));
			const ei::Vec3 direction = read<ei::Vec3>(m_state, get(m_state, light, "direction"));
			const ei::Vec3 intensity = read<ei::Vec3>(m_state, get(m_state, light, "intensity"))
				* read_opt<float>(m_state, light, "scale", 1.f);
			Radians angle;
			Radians falloffStart;
			if(auto angleIter = get(m_state, light, "cosWidth", false); angleIter != light.MemberEnd())
				angle = std::acos(read<float>(m_state, angleIter));
			else
				angle = Radians(read<float>(m_state, get(m_state, light, "width")));
			if(auto falloffIter = get(m_state, light, "cosFalloffStart", false);  falloffIter != light.MemberEnd())
				falloffStart = std::acos(read<float>(m_state, falloffIter));
			else
				falloffStart = Radians(read_opt<float>(m_state, light, "falloffStart", angle));

			if(auto hdl = world_add_light(lightIter->name.GetString(), LIGHT_SPOT); hdl.type == LIGHT_SPOT) {
				world_set_spot_light_position(hdl, util::pun<Vec3>(position));
				world_set_spot_light_intensity(hdl, util::pun<Vec3>(intensity));
				world_set_spot_light_direction(hdl, util::pun<Vec3>(direction));
				world_set_spot_light_angle(hdl, angle);
				world_set_spot_light_falloff(hdl, falloffStart);
				m_lightMap.emplace(lightIter->name.GetString(), hdl);
			} else throw std::runtime_error("Failed to add spot light");
		} else if(type.compare("directional") == 0) {
			// Directional light
			const ei::Vec3 direction = read<ei::Vec3>(m_state, get(m_state, light, "direction"));
			const ei::Vec3 irradiance = read<ei::Vec3>(m_state, get(m_state, light, "radiance"))
				* read_opt<float>(m_state, light, "scale", 1.f);

			if(auto hdl = world_add_light(lightIter->name.GetString(), LIGHT_DIRECTIONAL); hdl.type == LIGHT_DIRECTIONAL) {
				world_set_dir_light_direction(hdl, util::pun<Vec3>(direction));
				world_set_dir_light_irradiance(hdl, util::pun<Vec3>(irradiance));
				m_lightMap.emplace(lightIter->name.GetString(), hdl);
			} else throw std::runtime_error("Failed to add directional light");
		} else if(type.compare("envmap") == 0) {
			// Environment-mapped light
			TextureHdl texture = load_texture(read<const char*>(m_state, get(m_state, light, "map")), TextureSampling::SAMPLING_NEAREST);
			auto scaleIter = get(m_state, light, "scale", false);
			ei::Vec3 color { 1.0f };
			if(scaleIter != light.MemberEnd()) {
				if(scaleIter->value.IsArray())
					color = read<ei::Vec3>(m_state, scaleIter);
				else
					color = ei::Vec3{ read<float>(m_state, scaleIter) };
			}

			if(auto hdl = world_add_light(lightIter->name.GetString(), LIGHT_ENVMAP); hdl.type == LIGHT_ENVMAP) {
				world_set_env_light_map(hdl, texture);
				world_set_env_light_scale(hdl, util::pun<Vec3>(color));
				m_lightMap.emplace(lightIter->name.GetString(), hdl);
			} else throw std::runtime_error("Failed to add environment light");
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
	return true;
}

bool JsonLoader::load_materials() {
	auto scope = Profiler::instance().start<CpuProfileState>("JsonLoader::load_materials", ProfileLevel::HIGH);
	using namespace rapidjson;
	const Value& materials = m_materials->value;
	assertObject(m_state, materials);
	m_state.current = ParserState::Level::MATERIALS;

	for(auto matIter = materials.MemberBegin(); matIter != materials.MemberEnd(); ++matIter) {
		if(m_abort)
			return false;
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
	return true;
}

bool JsonLoader::load_scenarios(const std::vector<std::string>& binMatNames) {
	auto scope = Profiler::instance().start<CpuProfileState>("JsonLoader::load_scenarios", ProfileLevel::HIGH);
	using namespace rapidjson;
	const Value& scenarios = m_scenarios->value;
	assertObject(m_state, scenarios);
	m_state.current = ParserState::Level::SCENARIOS;

	for(auto scenarioIter = scenarios.MemberBegin(); scenarioIter != scenarios.MemberEnd(); ++scenarioIter) {
		logPedantic("[JsonLoader::load_scenarios] Loading scenario '", scenarioIter->name.GetString(), "'");
		if(m_abort)
			return false;
		m_state.objectNames.push_back(scenarioIter->name.GetString());
		const Value scenario = load_scenario(scenarioIter, m_scenarios->value.MemberCount());
		assertObject(m_state, scenario);

		const char* camera = read<const char*>(m_state, get(m_state, scenario, "camera"));
		ei::IVec2 resolution = read<ei::IVec2>(m_state, get(m_state, scenario, "resolution"));
		std::vector<const char*> lights;
		auto lightIter = get(m_state, scenario, "lights", false);
		u32 lod = read_opt<u32>(m_state, scenario, "lod", 0u);

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

		if(m_abort)
			return false;

		// Add lights
		if(lightIter != scenario.MemberEnd()) {
			assertArray(m_state, lightIter);
			m_state.objectNames.push_back("lights");
			for(SizeType i = 0u; i < lightIter->value.Size(); ++i) {
				StringView lightName = read<const char*>(m_state, lightIter->value[i]);
				auto nameIter = m_lightMap.find(lightName);
				if(nameIter == m_lightMap.cend()) {
					logWarning("[JsonLoader::load_scenarios] Unknown light source '", lightName, "' will be ignored");
				} else {
					if(!scenario_add_light(scenarioHdl, nameIter->second))
						throw std::runtime_error("Failed to add light '" + std::string(lightName) + "'");
				}
			}
			m_state.objectNames.pop_back();
		}

		// Add objects
		if(auto objectsIter = get(m_state, scenario, "objectProperties", false);
		   objectsIter != scenario.MemberEnd()) {
			if(m_abort)
				return false;
			m_state.objectNames.push_back(objectsIter->name.GetString());
			assertObject(m_state, objectsIter->value);
			for(auto objIter = objectsIter->value.MemberBegin(); objIter != objectsIter->value.MemberEnd(); ++objIter) {
				StringView objectName = objIter->name.GetString();
				m_state.objectNames.push_back(&objectName[0u]);
				const Value& object = objIter->value;
				assertObject(m_state, object);
				ObjectHdl objHdl = world_get_object(&objectName[0u]);
				if(objHdl == nullptr)
					throw std::runtime_error("Failed to find object '" + std::string(objectName) + "'");
				// Check for LoD and masked
				if(auto lodIter = get(m_state, object, "lod", false); lodIter != object.MemberEnd())
					if(!scenario_set_object_lod(scenarioHdl, objHdl, read<u32>(m_state, lodIter)))
						throw std::runtime_error("Failed to set LoD level of object '" + std::string(objectName) + "'");
				if(auto maskIter = get(m_state, object, "mask", false); maskIter != object.MemberEnd()
					&& read<bool>(m_state, maskIter))
					if(!scenario_mask_object(scenarioHdl, objHdl))
						throw std::runtime_error("Failed to set mask for object '" + std::string(objectName) + "'");

				m_state.objectNames.pop_back();
			}
			m_state.objectNames.pop_back();
		}

		// Read Instance LOD and masking information
		if(auto instancesIter = get(m_state, scenario, "instanceProperties", false);
		   instancesIter != scenario.MemberEnd()) {
			m_state.objectNames.push_back(instancesIter->name.GetString());
			assertObject(m_state, instancesIter->value);
			for(auto instIter = instancesIter->value.MemberBegin(); instIter != instancesIter->value.MemberEnd(); ++instIter) {
				StringView instName = instIter->name.GetString();
				m_state.objectNames.push_back(&instName[0u]);
				const Value& instance = instIter->value;
				assertObject(m_state, instance);
				InstanceHdl instHdl = world_get_instance(&instName[0u]);
				if(instHdl == nullptr)
					throw std::runtime_error("Failed to find instance '" + std::string(instName) + "'");
				// Read LoD
				if(auto lodIter = get(m_state, instance, "lod", false); lodIter != instance.MemberEnd())
					if(!scenario_set_instance_lod(scenarioHdl, instHdl, read<u32>(m_state, lodIter)))
						throw std::runtime_error("Failed to set LoD level of instance '" + std::string(instName) + "'");
				// Read masking information
				if(auto maskIter = get(m_state, instance, "mask", false); maskIter != instance.MemberEnd()
					&& read<bool>(m_state, maskIter))
					if(!scenario_mask_instance(scenarioHdl, instHdl))
						throw std::runtime_error("Failed to set mask of instance '" + std::string(instName) + "'");
				m_state.objectNames.pop_back();
			}
			m_state.objectNames.pop_back();
		}

		// Associate binary with JSON material names
		auto materialsIter = get(m_state, scenario, "materialAssignments");
		m_state.objectNames.push_back(materialsIter->name.GetString());
		assertObject(m_state, materialsIter->value);
		for(const std::string& binName : binMatNames) {
			if(m_abort)
				return false;
			// The binary names from the loader already wrap the name in the desired format
			StringView matName = read<const char*>(m_state, get(m_state, materialsIter->value,
																 binName.c_str()));
			logPedantic("[JsonLoader::load_scenarios] Associating material '", matName,
						"' with binary name '", binName, "'");
			MatIdx slot = scenario_declare_material_slot(scenarioHdl, binName.c_str(), binName.length());
			if(slot == INVALID_MATERIAL)
				throw std::runtime_error("Failed to declare material slot");
			auto matHdl = m_materialMap.find(matName);
			if(matHdl == m_materialMap.cend())
				throw std::runtime_error("Unknown material name '" + std::string(matName) + "' in association");
			if(!scenario_assign_material(scenarioHdl, slot, matHdl->second))
				throw std::runtime_error("Failed to associate material '" + matHdl->first + "'");
		}

		m_state.objectNames.pop_back();

		const char* sanityMsg = "";
		if(!scenario_is_sane(scenarioHdl, &sanityMsg))
			throw std::runtime_error("Scenario '" + std::string(scenarioIter->name.GetString())
									 + "' did not pass sanity check: " + std::string(sanityMsg));
		m_state.objectNames.pop_back();
	}
	return true;
}

rapidjson::Value JsonLoader::load_scenario(const rapidjson::GenericMemberIterator<true, rapidjson::UTF8<>, rapidjson::MemoryPoolAllocator<>>& scenarioIter,
	int maxRecursionDepth) {
	if(maxRecursionDepth < 1)
		throw std::runtime_error("Loop in scenario inheritance");
	using namespace rapidjson;
	const Value& scenarios = m_scenarios->value;

	const Value& scenario = scenarioIter->value;
	assertObject(m_state, scenario);

	if(auto parentIter = get(m_state, scenario, "parentScenario", false); parentIter != scenario.MemberEnd()) {
		const char* parentScenario = read<const char*>(m_state, parentIter);
		if(auto parent = scenarios.FindMember(parentScenario); parent != scenarios.MemberEnd()) {
			Value returnValue = load_scenario(parent, maxRecursionDepth-1);
			// returnValue is a deep copy => we can simply replace the
			// diferences to the current scenario.
			selective_replace_keys(scenario, returnValue);
			return returnValue;
		} else
			throw std::runtime_error("Failed to find parent scenario: " + std::string(parentScenario));
	}
	// Deep copy :-( to satisfy the interfaces of value return (no shallow copy possible)
	return Value(scenario, m_document.GetAllocator());
}

void JsonLoader::selective_replace_keys(const rapidjson::Value& objectToCopy, rapidjson::Value& target) {
	using namespace rapidjson;
	for(Value::ConstMemberIterator iter = objectToCopy.MemberBegin(); iter != objectToCopy.MemberEnd(); ++iter) {
		auto member = target.FindMember(iter->name);
		// Overwrite or add the value
		if(member != target.MemberEnd()) {
			if(iter->value.IsObject()) {
				// Recursion for objects
				selective_replace_keys(iter->value, member->value);
			} else
				member->value = Value(iter->value, m_document.GetAllocator());
		} else {
			// The target does not contain the key. Get a deep copy of whatever (objects/values).
			target.AddMember(Value(iter->name, m_document.GetAllocator()), Value(iter->value, m_document.GetAllocator()), m_document.GetAllocator());
		}
	}
}


bool JsonLoader::load_file() {
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
		if(m_version.compare(FILE_VERSION) != 0 && m_version.compare("1.0") != 0)
			logWarning("[JsonLoader::load_file] Scene file: version mismatch (",
					   m_version, "(file) vs ", FILE_VERSION, "(current))");
		logInfo("[JsonLoader::load_file] Detected file version '", m_version, "'");
	}
	// Binary file path
	m_binaryFile = read<const char*>(m_state, get(m_state, document, "binary"));
	if(m_binaryFile.empty())
		throw std::runtime_error("Scene file has an empty binary file path");
	logInfo("[JsonLoader::load_file] Detected binary file path '", m_binaryFile.string(), "'");
	// Make the file path absolute
	if(m_binaryFile.is_relative())
		m_binaryFile = fs::canonical(m_filePath.parent_path() / m_binaryFile);
	if(!fs::exists(m_binaryFile)) {
		logError("[JsonLoader::load_file] Scene file: specifies a binary file that doesn't exist ('",
				 m_binaryFile.string(), "'");
		throw std::runtime_error("Binary file '" + m_binaryFile.string() + "' does not exist");
	}

	if(m_abort)
		return false;

	m_lights = get(m_state, document, "lights");
	m_scenarios = get(m_state, document, "scenarios");
	m_cameras = get(m_state, document, "cameras");
	m_materials = get(m_state, document, "materials");

	// Partially parse the default scenario
	m_state.current = ParserState::Level::SCENARIOS;
	m_defaultScenario = read_opt<const char*>(m_state, document, "defaultScenario", "");
	// Choose first one in JSON - no guarantees
	if(m_defaultScenario.empty())
		m_defaultScenario = m_scenarios->value.MemberBegin()->name.GetString();
	logInfo("[JsonLoader::load_file] Detected default scenario '", m_defaultScenario, "'");
	const Value& defScen = get(m_state, m_scenarios->value, &m_defaultScenario[0u])->value;
	const u32 defaultGlobalLod = read_opt<u32>(m_state, defScen, "lod", 0u);
	logInfo("[JsonLoader::load_file] Detected global LoD '", m_defaultScenario, "'");

	// First parse binary file
	std::unordered_map<StringView, u32> defaultObjectLods;
	std::unordered_map<StringView, u32> defaultInstanceLods;
	auto objPropsIter = get(m_state, defScen, "objectProperties", false);
	if(objPropsIter != defScen.MemberEnd()) {
		m_state.objectNames.push_back(&m_defaultScenario[0u]);
		m_state.objectNames.push_back("objectProperties");
		for(auto propIter = objPropsIter->value.MemberBegin(); propIter != objPropsIter->value.MemberEnd(); ++propIter) {
			// Read the object name
			StringView objectName = propIter->name.GetString();
			m_state.objectNames.push_back(&objectName[0u]);
			const Value& object = propIter->value;
			assertObject(m_state, object);
			auto lodIter = get(m_state, object, "lod", false);
			if(lodIter != object.MemberEnd()) {
				const u32 localLod = read<u32>(m_state, lodIter);
				logPedantic("[JsonLoader::load_file] Custom LoD '", localLod, "' for object '", objectName, "'");
				defaultObjectLods.insert({ objectName, localLod });
			}
		}
		m_state.objectNames.pop_back();
		m_state.objectNames.pop_back();
	}
	auto instPropsIter = get(m_state, defScen, "instanceProperties", false);
	if(instPropsIter != defScen.MemberEnd()) {
		m_state.objectNames.push_back(&m_defaultScenario[0u]);
		m_state.objectNames.push_back("instanceProperties");
		for(auto propIter = instPropsIter->value.MemberBegin(); propIter != instPropsIter->value.MemberEnd(); ++propIter) {
			// Read the instance name
			StringView instanceName = propIter->name.GetString();
			m_state.objectNames.push_back(&instanceName[0u]);
			const Value& instance = propIter->value;
			assertObject(m_state, instance);
			auto lodIter = get(m_state, instance, "lod", false);
			if(lodIter != instance.MemberEnd()) {
				const u32 localLod = read<u32>(m_state, lodIter);
				logPedantic("[JsonLoader::load_file] Custom LoD '", localLod, "' for object '", instanceName, "'");
				defaultInstanceLods.insert({ instanceName, localLod });
			}
		}
		m_state.objectNames.pop_back();
		m_state.objectNames.pop_back();
	}
	bool deinstance = false;
	if(auto deinstanceIter = get(m_state, document, "deinstance", false); deinstanceIter != document.MemberEnd()) {
		deinstance = deinstanceIter->value.GetBool();
	}
	// Load the binary file before we load the rest of the JSON
	if(!m_binLoader.load_file(m_binaryFile, defaultGlobalLod, defaultObjectLods, defaultInstanceLods, deinstance))
		return false;

	try {
		// Cameras
		m_state.current = ParserState::Level::ROOT;
		if(!load_cameras(m_binLoader.get_bounding_box()))
			return false;
		// Lights
		m_state.current = ParserState::Level::ROOT;
		if(!load_lights())
			return false;
		// Materials
		m_state.current = ParserState::Level::ROOT;
		if(!load_materials())
			return false;
		// Before we load scenarios, perform a sanity check for the currently loaded world
		const char* sanityMsg = "";
		if(!world_is_sane(&sanityMsg))
			throw std::runtime_error("World did not pass sanity check: " + std::string(sanityMsg));
		// Scenarios
		m_state.current = ParserState::Level::ROOT;
		if(!load_scenarios(m_binLoader.get_material_names()))
			return false;
		// Load the default scenario
		m_state.reset();
		ScenarioHdl defScenHdl = world_find_scenario(&m_defaultScenario[0u]);
		if(defScenHdl == nullptr)
			throw std::runtime_error("Cannot find the default scenario '" + std::string(m_defaultScenario) + "'");

		auto scope = Profiler::instance().start<CpuProfileState>("JsonLoader::load_file - load default scenario", ProfileLevel::LOW);
		if(!world_load_scenario(defScenHdl))
			throw std::runtime_error("Cannot load the default scenario '" + std::string(m_defaultScenario) + "'");
	} catch(const std::runtime_error& e) {
		throw std::runtime_error(m_state.get_parser_level() + ": " + e.what());
	}
	return true;
}


} // namespace mff_loader::json