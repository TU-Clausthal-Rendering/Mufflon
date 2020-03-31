#include "json_loader.hpp"
#include "binary.hpp"
#include "profiler/cpu_profiler.hpp"
#include "util/log.hpp"
#include "util/punning.hpp"
#include "util/int_types.hpp"
#include "util/degrad.hpp"
#include "util/cie_xyz.hpp"
#include "core/scene/handles.hpp"
#include <rapidjson/document.h>
#include <rapidjson/error/en.h>
#include <fstream>
#include <sstream>

namespace mff_loader::json {

using namespace mufflon;

namespace {

// Reads a file completely and returns the string containing all bytes
std::string read_file(fs::path path) {
	auto scope = Profiler::loader().start<CpuProfileState>("JSON read_file", ProfileLevel::HIGH);
	logPedantic("[read_file] Loading JSON file '", path.u8string(), "' into RAM");
	const std::uintmax_t fileSize = fs::file_size(path);
	std::string fileString;
	fileString.resize(fileSize);

	std::ifstream file(path, std::ios::binary);
	file.read(&fileString[0u], fileSize);
	if(file.gcount() != static_cast<std::streamsize>(fileSize))
		logWarning("[read_file] File '", path.u8string(), "'not read completely");
	// Finalize the string
	fileString[file.gcount()] = '\0';
	return fileString;
}

// Reads an optional array
template < class T >
std::enable_if_t<is_array<T>(), std::vector<T>> read_opt_array_array(ParserState& state,
																	 const rapidjson::Value& parent,
																	 const char* name) {
	std::vector<T> vec{};
	if(auto valIter = get(state, parent, name, false); valIter != parent.MemberEnd()) {
		// Check if it's cascaded arrays
		if(valIter->value.IsArray() && valIter->value.Size() > 0 && valIter->value[0].IsArray())
			read(state, valIter, vec);
		else
			vec.push_back(read<T>(state, valIter));
	}
	return vec;
}
template < class T >
std::enable_if_t<!is_array<T>(), std::vector<T>> read_opt_array(ParserState& state,
																const rapidjson::Value& parent,
																const char* name) {
	std::vector<T> vec{};
	if(auto valIter = get(state, parent, name, false); valIter != parent.MemberEnd())
		read(state, valIter, vec);
	return vec;
}
template < class T >
std::enable_if_t<is_array<T>(), std::vector<T>> read_opt_array_array(ParserState& state,
																	 const rapidjson::Value& parent,
																	 const char* name, const T& optVal) {
	std::vector<T> vec{};
	if(auto valIter = get(state, parent, name, false); valIter != parent.MemberEnd()) {
		// Check if it's cascaded arrays
		if(valIter->value.IsArray() && valIter->value.Size() > 0 && valIter->value[0].IsArray())
			read(state, valIter, vec);
		else
			vec.push_back(read<T>(state, valIter));
	} else {
		vec.push_back(optVal);
	}
	return vec;
}
template < class T >
std::enable_if_t<!is_array<T>(), std::vector<T>> read_opt_array(ParserState& state,
																const rapidjson::Value& parent,
																const char* name, const T& optVal) {
	std::vector<T> vec{};
	if(auto valIter = get(state, parent, name, false); valIter != parent.MemberEnd())
		read(state, valIter, vec);
	else
		vec.push_back(optVal);
	return vec;
}

void parse_object_instance_properties(ParserState& state, const rapidjson::Value& scenario,
									  util::FixedHashMap<StringView, u32>& objProps,
									  util::FixedHashMap<StringView, binary::InstanceMapping>& instProps,
									  const u32 defaultGlobalLod) {
	using namespace rapidjson;
	if(const auto objPropsIter = get(state, scenario, "objectProperties", false);
	   objPropsIter != scenario.MemberEnd()) {
		for(auto propIter = objPropsIter->value.MemberBegin(); propIter != objPropsIter->value.MemberEnd(); ++propIter) {
			// Read the object name
			StringView objectName = propIter->name.GetString();
			if(objProps.find(objectName) != objProps.cend())
				continue;
			state.objectNames.push_back(&objectName[0u]);
			const Value& object = propIter->value;
			assertObject(state, object);
			if(auto lodIter = get(state, object, "lod", false); lodIter != object.MemberEnd()) {
				const u32 localLod = read<u32>(state, lodIter);
				logPedantic("[JsonLoader::load_file] Custom LoD '", localLod, "' for object '", objectName, "'");
				objProps.insert(objectName, localLod);
			}
			state.objectNames.pop_back();
		}
	}
	if(const auto instPropsIter = get(state, scenario, "instanceProperties", false);
	   instPropsIter != scenario.MemberEnd()) {
		for(auto propIter = instPropsIter->value.MemberBegin(); propIter != instPropsIter->value.MemberEnd(); ++propIter) {
			// Read the instance name
			StringView instanceName = propIter->name.GetString();
			if(instProps.find(instanceName) != instProps.cend())
				continue;
			state.objectNames.push_back(&instanceName[0u]);
			const Value& instance = propIter->value;
			assertObject(state, instance);
			const u32 localLod = read_opt<u32>(state, instance, "lod", defaultGlobalLod);
			instProps.insert(instanceName, { localLod, nullptr });

			if(localLod != defaultGlobalLod)
				logPedantic("[JsonLoader::load_file] Custom LoD '", localLod,
							"' for object '", instanceName, "'");
			state.objectNames.pop_back();
		}
	}
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
	m_materialMap.clear();
	m_loadingStage.clear();
}

TextureHdl JsonLoader::load_texture(const char* name, TextureSampling sampling, MipmapType mipmapType,
									std::optional<TextureFormat> targetFormat,
									TextureCallback callback, void* userParams) {
	auto scope = Profiler::loader().start<CpuProfileState>("JsonLoader::load_texture", ProfileLevel::HIGH);
	logPedantic("[JsonLoader::load_texture] Loading texture '", name, "'");
	// Make the path relative to the file
	auto path = fs::u8path(name);
	if (!path.is_absolute())
		path = m_filePath.parent_path() / name;
	if (!fs::exists(path))
		throw std::runtime_error("Cannot find texture file '" + path.u8string() + "'");
	path = fs::canonical(path);
	TextureHdl tex;
	if(targetFormat.has_value())
		tex = world_add_texture_converted(m_mffInstHdl, path.u8string().c_str(), sampling, targetFormat.value(), mipmapType, callback, userParams);
	else
		tex = world_add_texture(m_mffInstHdl, path.u8string().c_str(), sampling, mipmapType, callback, userParams);
	if(tex == nullptr)
		throw std::runtime_error("Failed to load texture '" + std::string(name) + "'");
	return tex;
}

std::pair<TextureHdl, TextureHdl> JsonLoader::load_displacement_map(const char* name) {
	auto scope = Profiler::loader().start<CpuProfileState>("JsonLoader::load_displacement_map", ProfileLevel::HIGH);
	// Make the path relative to the file
	auto path = fs::u8path(name);
	if(!path.is_absolute())
		path = m_filePath.parent_path() / name;
	if(!fs::exists(path))
		throw std::runtime_error("Cannot find texture file '" + path.u8string() + "'");
	path = fs::canonical(path);
	TextureHdl tex = nullptr;
	TextureHdl texMips = nullptr;
	if(!world_add_displacement_map(m_mffInstHdl, path.u8string().c_str(), &tex, &texMips) || tex == nullptr || texMips == nullptr)
		throw std::runtime_error("Failed to add displacement map '" + std::string(name) + "'");
	return { tex, texMips };
}

MaterialParams* JsonLoader::load_material(rapidjson::Value::ConstMemberIterator matIter) {
	auto scope = Profiler::loader().start<CpuProfileState>("JsonLoader::load_material", ProfileLevel::HIGH);
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
			mat->outerMedium.absorption = Vec3{ 0.f, 0.f, 0.f };
			mat->outerMedium.refractionIndex = Vec2{ 1.f, 0.f };
		}

		// Read an optional alpha texture
		const char* alphaPath = read_opt<const char*>(m_state, material, "alpha", nullptr);

		StringView type = read<const char*>(m_state, get(m_state, material, "type"));
		if(type.compare("lambert") == 0) {
			// Lambert material
			mat->innerType = MaterialParamType::MATERIAL_LAMBERT;
			auto albedoIter = get(m_state, material, "albedo");
			if(albedoIter->value.IsArray()) {
				ei::Vec3 albedo = read<ei::Vec3>(m_state, albedoIter);
				mat->inner.lambert.albedo = world_add_texture_value(m_mffInstHdl, reinterpret_cast<float*>(&albedo), 3, TextureSampling::SAMPLING_NEAREST);
			} else if(albedoIter->value.IsString()) {
				mat->inner.lambert.albedo = load_texture(read<const char*>(m_state, albedoIter), TextureSampling::SAMPLING_LINEAR);
			} else
				throw std::runtime_error("Invalid type for albedo.");

		} else if(type.compare("torrance") == 0) {
			// Torrance material
			mat->innerType = MaterialParamType::MATERIAL_TORRANCE;
			StringView shadowingModel = read_opt<const char*>(m_state, material, "shadowingModel", "vcavity");
			if(shadowingModel.compare("smith") == 0)
				mat->inner.torrance.shadowingModel = ShadowingModel::SHADOWING_SMITH;
			else if(shadowingModel.compare("vcavity") == 0)
				mat->inner.torrance.shadowingModel = ShadowingModel::SHADOWING_VCAVITY;
			else
				throw std::runtime_error("Unknown shadowing model '" + std::string(shadowingModel) + "'");
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
				ei::Vec2 xy = read<ei::Vec2>(m_state, roughnessIter);
				mat->inner.torrance.roughness = world_add_texture_value(m_mffInstHdl, reinterpret_cast<float*>(&xy), 2, TextureSampling::SAMPLING_NEAREST);
			} else if(roughnessIter->value.IsNumber()) {
				float alpha = read<float>(m_state, roughnessIter);
				mat->inner.torrance.roughness = world_add_texture_value(m_mffInstHdl, &alpha, 1, TextureSampling::SAMPLING_NEAREST);
			} else if(roughnessIter->value.IsString()) {
				mat->inner.torrance.roughness = load_texture(read<const char*>(m_state, roughnessIter));
			} else
				throw std::runtime_error("Invalid type for roughness.");
			auto albedoIter = get(m_state, material, "albedo");
			if(albedoIter->value.IsArray()) {
				ei::Vec3 albedo = read<ei::Vec3>(m_state, albedoIter);
				mat->inner.torrance.albedo = world_add_texture_value(m_mffInstHdl, reinterpret_cast<float*>(&albedo), 3, TextureSampling::SAMPLING_NEAREST);
			} else if(albedoIter->value.IsString()) {
				mat->inner.torrance.albedo = load_texture(read<const char*>(m_state, albedoIter));
			} else
				throw std::runtime_error("Invalid type for albedo.");

		} else if(type.compare("walter") == 0 || type.compare("microfacet") == 0) {
			// Walter and Microfacet materials have the same parametrization (load as
			// Walter pack, but modify the enum accordingly).
			mat->innerType = type.compare("walter") == 0
				? MaterialParamType::MATERIAL_WALTER
				: MaterialParamType::MATERIAL_MICROFACET;
			StringView shadowingModel = read_opt<const char*>(m_state, material, "shadowingModel", "vcavity");
			if(shadowingModel.compare("smith") == 0)
				mat->inner.walter.shadowingModel = ShadowingModel::SHADOWING_SMITH;
			else if(shadowingModel.compare("vcavity") == 0)
				mat->inner.walter.shadowingModel = ShadowingModel::SHADOWING_VCAVITY;
			else 
				throw std::runtime_error("Unknown shadowing model '" + std::string(shadowingModel) + "'");
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
				mat->inner.walter.roughness = world_add_texture_value(m_mffInstHdl, &alpha, 1, TextureSampling::SAMPLING_NEAREST);
			} else if(roughnessIter->value.IsString()) {
				mat->inner.walter.roughness = load_texture(read<const char*>(m_state, roughnessIter));
			} else if(roughnessIter->value.IsArray()) {
				ei::Vec2 xy = read<ei::Vec2>(m_state, roughnessIter);
				mat->inner.walter.roughness = world_add_texture_value(m_mffInstHdl, reinterpret_cast<float*>(&xy), 2, TextureSampling::SAMPLING_NEAREST);
			} else
				throw std::runtime_error("Invalid type for roughness");

			mat->inner.walter.refractionIndex = read<float>(m_state, get(m_state, material, "ior"));

		} else if(type.compare("emissive") == 0) {
			// Emissive material
			if(alphaPath != nullptr) {
				logWarning("[JsonLoader::load_material] Found alpha texture for emissive material; will be ignored");
				alphaPath = nullptr;
			}
			mat->innerType = MaterialParamType::MATERIAL_EMISSIVE;
			mat->inner.emissive.scale = util::pun<Vec3>(read_opt<ei::Vec3>(m_state, material, "scale", ei::Vec3{ 1.0f, 1.0f, 1.0f }));
			if(auto radianceIter = get(m_state, material, "radiance", false); radianceIter != material.MemberEnd()) {
				if(radianceIter->value.IsArray()) {
					ei::Vec3 rgb = read<ei::Vec3>(m_state, radianceIter);
					mat->inner.emissive.radiance = world_add_texture_value(m_mffInstHdl, reinterpret_cast<const float*>(&rgb), 3, TextureSampling::SAMPLING_NEAREST);
				} else if(radianceIter->value.IsString()) {
					mat->inner.emissive.radiance = load_texture(read<const char*>(m_state, radianceIter));
				} else
					throw std::runtime_error("Invalid type for radiance");
			} else {
				auto temperatureIter = get(m_state, material, "temperature");
				if(temperatureIter->value.IsNumber()) {
					const float temperature = read<float>(m_state, temperatureIter);
					const auto radiance = spectrum::compute_black_body_color(spectrum::Kelvin{ temperature });
					mat->inner.emissive.radiance = world_add_texture_value(m_mffInstHdl, reinterpret_cast<const float*>(&radiance), 3, TextureSampling::SAMPLING_NEAREST);
				} else if(temperatureIter->value.IsString()) {
					mat->inner.emissive.radiance = load_texture(read<const char*>(m_state, radianceIter),
																TextureSampling::SAMPLING_NEAREST, MipmapType::MIPMAP_NONE,
																TextureFormat::FORMAT_RGBA32F,
																[](uint32_t /*x*/, uint32_t /*y*/, uint32_t /*layer*/,
																   TextureFormat /*format*/, Vec4 value,
																   void* /*userParams*/) {
						const auto radiance = spectrum::compute_black_body_color(spectrum::Kelvin{ value.x });
						return Vec4{ radiance.x, radiance.y, radiance.z, 0.f };
					}, nullptr);
				} else
					throw std::runtime_error("Invalid type for temperature");
			}

		} else if(type.compare("orennayar") == 0) {
			// Oren-Nayar material
			mat->innerType = MaterialParamType::MATERIAL_ORENNAYAR;
			mat->inner.orennayar.roughness = read_opt<float>(m_state, material, "roughness", 1.f);
			auto albedoIter = get(m_state, material, "albedo");
			if(albedoIter->value.IsArray()) {
				ei::Vec3 rgb = read<ei::Vec3>(m_state, albedoIter);
				mat->inner.orennayar.albedo = world_add_texture_value(m_mffInstHdl, reinterpret_cast<const float*>(&rgb), 3, TextureSampling::SAMPLING_NEAREST);
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

		// We load the alpha texture last to give emissive materials to deny it, if present
		if(alphaPath != nullptr) {
			mat->alpha = load_texture(alphaPath, TextureSampling::SAMPLING_LINEAR, MipmapType::MIPMAP_NONE,
									  TextureFormat::FORMAT_R8U);
		} else {
			mat->alpha = nullptr;
		}

		// Load optional displacement map
		if(const auto dispIter = get(m_state, material, "displacement", false); dispIter != material.MemberEnd()) {
			// Parse the outer medium of the material
			m_state.objectNames.push_back(dispIter->name.GetString());
			mat->displacement.bias = read_opt<float>(m_state, dispIter->value, "bias", 0.f);
			mat->displacement.scale = read_opt<float>(m_state, dispIter->value, "scale", 1.f);
			auto[tex, mipsTex] = load_displacement_map(read<const char*>(m_state, get(m_state, dispIter->value, "map")));
			mat->displacement.map = tex;
			mat->displacement.maxMips = mipsTex;
			m_state.objectNames.pop_back();
		} else {
			mat->displacement.map = nullptr;
		}
	} catch(const std::exception&) {
		free_material(mat);
		throw;
	}

	m_state.objectNames.pop_back();
	return mat;
}

void JsonLoader::free_material(MaterialParams* mat) {
	if(mat == nullptr)
		return;
	switch(mat->innerType) {
		case MATERIAL_LAMBERT:
		case MATERIAL_TORRANCE:
		case MATERIAL_WALTER:
		case MATERIAL_EMISSIVE:
		case MATERIAL_ORENNAYAR:
			break;
		case MATERIAL_BLEND:
			free_material(mat->inner.blend.a.mat);
			free_material(mat->inner.blend.b.mat);
			break;
		case MATERIAL_FRESNEL:
			free_material(mat->inner.fresnel.a);
			free_material(mat->inner.fresnel.b);
			break;
		default: break;
	}
	delete mat;
}

bool JsonLoader::load_cameras(const ei::Box& aabb) {
	auto scope = Profiler::loader().start<CpuProfileState>("JsonLoader::load_cameras", ProfileLevel::HIGH);
	sprintf(m_loadingStage.data(), "Parsing cameras%c", '\0');
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
		const float near = read_opt<float>(m_state, camera, "near",
										   m_absoluteCamNearFar ? DEFAULT_NEAR_PLANE : (DEFAULT_NEAR_PLANE_FACTOR * sceneDiag));
		const float far = read_opt<float>(m_state, camera, "far",
										  m_absoluteCamNearFar ? DEFAULT_FAR_PLANE : (DEFAULT_FAR_PLANE_FACTOR * sceneDiag));
		StringView type = read<const char*>(m_state, get(m_state, camera, "type"));
		std::vector<ei::Vec3> camPath;
		std::vector<ei::Vec3> camViewDir;
		std::vector<ei::Vec3> camUp;
		read(m_state, get(m_state, camera, "path"), camPath);
		read(m_state, get(m_state, camera, "viewDir"), camViewDir, camPath.size());
		if(auto upIter = get(m_state, camera, "up", false); upIter != camera.MemberEnd())
			read(m_state, get(m_state, camera, "up"), camUp, camPath.size());
		else
			camUp = std::vector<ei::Vec3>{ ei::Vec3{ 0, 1, 0 } };
		if(camViewDir.size() == 1u && (camPath.size() != 1u || camUp.size() != 1u))
			camViewDir = std::vector<ei::Vec3>(camPath.size(), camViewDir.front());
		if(camUp.size() == 1u && (camPath.size() != 1u || camViewDir.size() != 1u))
			camUp = std::vector<ei::Vec3>(camPath.size(), camUp.front());
		if(camPath.size() == 1u && (camViewDir.size() != 1u || camUp.size() != 1u))
			camPath = std::vector<ei::Vec3>(camViewDir.size(), camPath.front());

		if(camPath.size() != camViewDir.size())
			throw std::runtime_error("Mismatched camera path size (view direction)");
		if(camPath.size() != camUp.size())
			throw std::runtime_error("Mismatched camera path size (up direction)");

		const auto maxSize = std::max(camPath.size(), std::max(camViewDir.size(), camUp.size()));
		if(camUp.size() == 1u && maxSize != 1u)
			camUp = std::vector<ei::Vec3>(maxSize, camUp.front());
		if(camViewDir.size() == 1u && maxSize != 1u)
			camViewDir = std::vector<ei::Vec3>(maxSize, camViewDir.front());
		if(camUp.size() == 1u && maxSize != 1u)
			camUp = std::vector<ei::Vec3>(maxSize, camUp.front());

		// Per-camera-model values
		if(type.compare("pinhole") == 0) {
			// Pinhole camera
			const float fovDegree = read_opt<float>(m_state, camera, "fov", 25.f);
			if(world_add_pinhole_camera(m_mffInstHdl, cameraIter->name.GetString(), reinterpret_cast<const Vec3*>(camPath.data()),
										reinterpret_cast<const Vec3*>(camViewDir.data()), reinterpret_cast<const Vec3*>(camUp.data()),
										static_cast<uint32_t>(camPath.size()), near, far, static_cast<Radians>(Degrees(fovDegree))) == nullptr)
				throw std::runtime_error("Failed to add pinhole camera");
		} else if(type.compare("focus") == 0) {
			const float focalLength = read_opt<float>(m_state, camera, "focalLength", 35.f) / 1000.f;
			const float focusDistance = read<float>(m_state, get(m_state, camera, "focusDistance"));
			const float sensorHeight = read_opt<float>(m_state, camera, "chipHeight", 24.f) / 1000.f;
			const float lensRadius = (focalLength / (2.f * read_opt<float>(m_state, camera, "aperture", focalLength)));
			if(world_add_focus_camera(m_mffInstHdl, cameraIter->name.GetString(), reinterpret_cast<const Vec3*>(camPath.data()),
									  reinterpret_cast<const Vec3*>(camViewDir.data()), reinterpret_cast<const Vec3*>(camUp.data()),
									  static_cast<uint32_t>(camPath.size()), near, far, focalLength, focusDistance,
									  lensRadius, sensorHeight) == nullptr)
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
	auto scope = Profiler::loader().start<CpuProfileState>("JsonLoader::load_lights", ProfileLevel::HIGH);
	sprintf(m_loadingStage.data(), "Parsing lights%c", '\0');
	using namespace rapidjson;
	const Value& lights = m_lights->value;
	assertObject(m_state, lights);
	m_state.current = ParserState::Level::LIGHTS;

	m_lightMap = util::FixedHashMap<StringView, LightHdl>{ lights.MemberCount() };
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
			std::vector<ei::Vec3> positions = read_opt_array_array<ei::Vec3>(m_state, light, "position");
			std::vector<float> scales = read_opt_array<float>(m_state, light, "scale", 1.f);
			// For backwards compatibility, we try to read a normal array as fallback
			std::vector<ei::Vec3> intensities;
			if(auto intensityIter = get(m_state, light, "intensity", false); intensityIter != light.MemberEnd()) {
				read(m_state, intensityIter, intensities);
			} else if(auto fluxIter = get(m_state, light, "flux", false); fluxIter != light.MemberEnd()) {
				read(m_state, fluxIter, intensities);
				for(auto& flux : intensities)
					flux /= (4.0f * ei::PI);
			} else {
				std::vector<float> temperatures;
				read(m_state, get(m_state, light, "temperature"), temperatures);
				intensities.reserve(temperatures.size());
				for(const auto temp : temperatures) {
					const auto rgb = spectrum::compute_black_body_color(spectrum::Kelvin{ temp });
					intensities.push_back(rgb);
				}
			}

			const std::size_t maxSize = std::max(positions.size(), std::max(intensities.size(), scales.size()));
			if(intensities.size() == 1u && maxSize != 1u)
				intensities = std::vector<ei::Vec3>(maxSize, intensities.front());
			if(positions.size() == 1u && maxSize != 1u)
				positions = std::vector<ei::Vec3>(maxSize, positions.front());
			if(scales.size() == 1u && maxSize != 1u)
				scales = std::vector<float>(maxSize, scales.front());
			if(positions.size() != intensities.size())
				throw std::runtime_error("Mismatched light path size (intensities)");
			if(positions.size() != scales.size())
				throw std::runtime_error("Mismatched light path size (scales)");

			if(auto hdl = world_add_light(m_mffInstHdl, lightIter->name.GetString(), LIGHT_POINT, static_cast<uint32_t>(positions.size())); hdl.type == LIGHT_POINT) {
				for(u32 i = 0u; i < static_cast<uint32_t>(positions.size()); ++i) {
					world_set_point_light_position(m_mffInstHdl, hdl, util::pun<Vec3>(positions[i]), i);
					world_set_point_light_intensity(m_mffInstHdl, hdl, util::pun<Vec3>(intensities[i] * scales[i]), i);
				}
				m_lightMap.emplace(lightIter->name.GetString(), hdl);
			} else throw std::runtime_error("Failed to add point light");
		} else if(type.compare("spot") == 0) {
			// Spot light
			std::vector<ei::Vec3> positions = read_opt_array_array<ei::Vec3>(m_state, light, "position");
			std::vector<ei::Vec3> directions = read_opt_array_array<ei::Vec3>(m_state, light, "direction");
			// For backwards compatibility, we try to read a normal array as fallback
			std::vector<ei::Vec3> intensities;
			if(auto intensityIter = get(m_state, light, "intensity", false); intensityIter != light.MemberEnd()) {
				read(m_state, intensityIter, intensities);
			} else {
				std::vector<float> temperatures;
				read(m_state, get(m_state, light, "temperature"), temperatures);
				intensities.reserve(temperatures.size());
				for(const auto temp : temperatures) {
					const auto rgb = spectrum::compute_black_body_color(spectrum::Kelvin{ temp });
					intensities.push_back(rgb);
				}
			}
			
			std::vector<float> scales = read_opt_array<float>(m_state, light, "scale", 1.f);
			std::vector<float> angles;
			std::vector<float> falloffStarts;
			// For backwards compatibility, we try to read a normal array as fallback
			if(auto angleIter = get(m_state, light, "cosWidth", false); angleIter != light.MemberEnd())
				read(m_state, angleIter, angles);
			else
				read(m_state, get(m_state, light, "width"), angles);
			
			// This is a bit complex and may need refactoring; we have multiple scenarios to catch depending on what is
			// and isn't given in the JSON but legal according to the file specs
			if(auto falloffIter = get(m_state, light, "cosFalloffStart", false); falloffIter != light.MemberEnd()) {
				read(m_state, falloffIter, falloffStarts);
			} else {
				if(auto angleIter = get(m_state, light, "falloffStart", false); angleIter != light.MemberEnd()) {
					read(m_state, angleIter, falloffStarts);
				} else {
					falloffStarts.reserve(angles.size());
					for(const auto& angle : angles)
						falloffStarts.push_back(Radians{ angle });
				}
			}
			
			const std::size_t maxSize = std::max(positions.size(), std::max(intensities.size(),
												std::max(scales.size(), std::max(directions.size(),
														std::max(angles.size(), falloffStarts.size())))));
			if(intensities.size() == 1u && maxSize != 1u)
				intensities = std::vector<ei::Vec3>(maxSize, intensities.front());
			if(positions.size() == 1u && maxSize != 1u)
				positions = std::vector<ei::Vec3>(maxSize, positions.front());
			if(scales.size() == 1u && maxSize != 1u)
				scales = std::vector<float>(maxSize, scales.front());
			if(directions.size() == 1u && maxSize != 1u)
				directions = std::vector<ei::Vec3>(maxSize, directions.front());
			if(angles.size() == 1u && maxSize != 1u)
				angles = std::vector<float>(maxSize, angles.front());
			if(falloffStarts.size() == 1u && maxSize != 1u)
				falloffStarts = std::vector<float>(maxSize, falloffStarts.front());
			if(positions.size() != intensities.size())
				throw std::runtime_error("Mismatched light path size (intensities)");
			if(positions.size() != directions.size())
				throw std::runtime_error("Mismatched light path size (directions)");
			if(positions.size() != scales.size())
				throw std::runtime_error("Mismatched light path size (scales)");
			if(positions.size() != angles.size())
				throw std::runtime_error("Mismatched light path size (angles)");
			if(positions.size() != falloffStarts.size())
				throw std::runtime_error("Mismatched light path size (falloffStarts)");

			if(auto hdl = world_add_light(m_mffInstHdl, lightIter->name.GetString(), LIGHT_SPOT,
										  static_cast<u32>(positions.size())); hdl.type == LIGHT_SPOT) {
				for(u32 i = 0u; i < static_cast<uint32_t>(positions.size()); ++i) {
					world_set_spot_light_position(m_mffInstHdl, hdl, util::pun<Vec3>(positions[i]), i);
					world_set_spot_light_intensity(m_mffInstHdl, hdl, util::pun<Vec3>(intensities[i] * scales[i]), i);
					world_set_spot_light_direction(m_mffInstHdl, hdl, util::pun<Vec3>(directions[i]), i);
					world_set_spot_light_angle(m_mffInstHdl, hdl, angles[i], i);
					world_set_spot_light_falloff(m_mffInstHdl, hdl, falloffStarts[i], i);
				}
				m_lightMap.emplace(lightIter->name.GetString(), hdl);
			} else throw std::runtime_error("Failed to add spot light");
		} else if(type.compare("directional") == 0) {
			// Directional light
			std::vector<ei::Vec3> directions = read_opt_array_array<ei::Vec3>(m_state, light, "direction");
			// For backwards compatibility, we try to read a normal array as fallback
			std::vector<ei::Vec3> radiances;
			if(auto radianceIter = get(m_state, light, "radiance", false); radianceIter != light.MemberEnd()) {
				read(m_state, radianceIter, radiances);
			} else {
				std::vector<float> temperatures;
				read(m_state, get(m_state, light, "temperature"), temperatures);
				radiances.reserve(temperatures.size());
				for(const auto temp : temperatures) {
					const auto rgb = spectrum::compute_black_body_color(spectrum::Kelvin{ temp });
					radiances.push_back(rgb);
				}
			}
			std::vector<float> scales = read_opt_array<float>(m_state, light, "scale", 1.f);

			const std::size_t maxSize = std::max(directions.size(), std::max(radiances.size(), scales.size()));
			if(radiances.size() == 1u && maxSize != 1u)
				radiances = std::vector<ei::Vec3>(maxSize, radiances.front());
			if(directions.size() == 1u && maxSize != 1u)
				directions = std::vector<ei::Vec3>(maxSize, directions.front());
			if(scales.size() == 1u && maxSize != 1u)
				scales = std::vector<float>(maxSize, scales.front());
			if(directions.size() != radiances.size())
				throw std::runtime_error("Mismatched light path size (radiances)");
			if(directions.size() != scales.size())
				throw std::runtime_error("Mismatched light path size (scales)");

			if(auto hdl = world_add_light(m_mffInstHdl, lightIter->name.GetString(), LIGHT_DIRECTIONAL,
										  static_cast<u32>(directions.size())); hdl.type == LIGHT_DIRECTIONAL) {
				for(u32 i = 0u; i < static_cast<uint32_t>(directions.size()); ++i) {
					world_set_dir_light_direction(m_mffInstHdl, hdl, util::pun<Vec3>(directions[i]), i);
					world_set_dir_light_irradiance(m_mffInstHdl, hdl, util::pun<Vec3>(radiances[i] * scales[i]), i);
				}
				m_lightMap.emplace(lightIter->name.GetString(), hdl);
			} else throw std::runtime_error("Failed to add directional light");
		} else if(type.compare("envmap") == 0) {
			ei::Vec3 color{ 1.0f };
			if(const auto scaleIter = get(m_state, light, "scale", false); scaleIter != light.MemberEnd()) {
				if(scaleIter->value.IsArray())
					color = read<ei::Vec3>(m_state, scaleIter);
				else
					color = ei::Vec3{ read<float>(m_state, scaleIter) };
			}

			// Check if we have a proper envmap or a simple monochrome background
			if(const auto mapIter = get(m_state, light, "map", false); mapIter != light.MemberEnd()) {
				TextureHdl texture = load_texture(read<const char*>(m_state, get(m_state, light, "map")), TextureSampling::SAMPLING_NEAREST);
				if(auto hdl = world_add_background_light(m_mffInstHdl, lightIter->name.GetString(), BackgroundType::BACKGROUND_ENVMAP);
				   hdl.type == LightType::LIGHT_ENVMAP) {
					world_set_env_light_map(m_mffInstHdl, hdl, texture);
					world_set_env_light_scale(m_mffInstHdl, hdl, util::pun<Vec3>(color));
					m_lightMap.emplace(lightIter->name.GetString(), hdl);
				} else throw std::runtime_error("Failed to add environment light");
			} else {
				if(auto hdl = world_add_background_light(m_mffInstHdl, lightIter->name.GetString(), BackgroundType::BACKGROUND_MONOCHROME);
				   hdl.type == LightType::LIGHT_ENVMAP) {
					world_set_env_light_color(m_mffInstHdl, hdl, util::pun<Vec3>(color));
					m_lightMap.emplace(lightIter->name.GetString(), hdl);
				} else throw std::runtime_error("Failed to add monochromatic environment light");
			}
		} else if(type.compare("sky") == 0) {
			const auto turbidity = read_opt<float>(m_state, light, "turbidity", 1.f);
			const auto albedo = read_opt<float>(m_state, light, "albedo", 0.f);
			const auto solarRadius = read_opt<float>(m_state, light, "solarRadius", 0.00445059f);
			const auto sunDir = read_opt<ei::Vec3>(m_state, light, "sunDir", ei::Vec3{ 0.f, 1.f, 0.f });
			const auto modelName = read_opt<const char*>(m_state, light, "model", "hosek");
			ei::Vec3 scale{ 1.0f };
			if(const auto scaleIter = get(m_state, light, "scale", false); scaleIter != light.MemberEnd()) {
				if(scaleIter->value.IsArray())
					scale = read<ei::Vec3>(m_state, scaleIter);
				else
					scale = ei::Vec3{ read<float>(m_state, scaleIter) };
			}
			
			BackgroundType type;
			if(std::strncmp(modelName, "hosek", 5u) == 0)
				type = BackgroundType::BACKGROUND_SKY_HOSEK;
			else // TODO: Preetham?
				throw std::runtime_error("Unknown sky model '" + std::string(modelName) + "'");

			if(auto hdl = world_add_background_light(m_mffInstHdl, lightIter->name.GetString(), type);
			   hdl.type == LightType::LIGHT_ENVMAP) {
				world_set_sky_light_turbidity(m_mffInstHdl, hdl, turbidity);
				world_set_sky_light_albedo(m_mffInstHdl, hdl, albedo);
				world_set_sky_light_solar_radius(m_mffInstHdl, hdl, solarRadius);
				world_set_sky_light_sun_direction(m_mffInstHdl, hdl, util::pun<Vec3>(sunDir));
				world_set_env_light_scale(m_mffInstHdl, hdl, util::pun<Vec3>(scale));
				m_lightMap.emplace(lightIter->name.GetString(), hdl);
			} else throw std::runtime_error("Failed to add sky light");
		} else if(type.compare("goniometric") == 0) {
			// TODO: Goniometric light
			std::vector<ei::Vec3> positions = read_opt_array_array<ei::Vec3>(m_state, light, "position");
			std::vector<float> scales = read_opt_array<float>(m_state, light, "scale", 1.f);
			(void)load_texture(read<const char*>(m_state, get(m_state, light, "map")));
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
	auto scope = Profiler::loader().start<CpuProfileState>("JsonLoader::load_materials", ProfileLevel::HIGH);
	sprintf(m_loadingStage.data(), "Parsing materials%c", '\0');
	using namespace rapidjson;
	const Value& materials = m_materials->value;
	assertObject(m_state, materials);
	m_state.current = ParserState::Level::MATERIALS;

	for(auto matIter = materials.MemberBegin(); matIter != materials.MemberEnd(); ++matIter) {
		if(m_abort)
			return false;
		MaterialParams* mat = load_material(matIter);
		if(mat != nullptr) {
			auto hdl = world_add_material(m_mffInstHdl, matIter->name.GetString(), mat);
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

bool JsonLoader::load_scenarios(const std::vector<std::string>& binMatNames,
								const mufflon::util::FixedHashMap<StringView, binary::InstanceMapping>& instances) {
	auto scope = Profiler::loader().start<CpuProfileState>("JsonLoader::load_scenarios", ProfileLevel::HIGH);
	sprintf(m_loadingStage.data(), "Parsing scenarios%c", '\0');
	using namespace rapidjson;
	const Value& scenarios = m_scenarios->value;
	assertObject(m_state, scenarios);
	m_state.current = ParserState::Level::SCENARIOS;

	world_reserve_scenarios(m_mffInstHdl, static_cast<u32>(scenarios.MemberCount()));
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

		CameraHdl camHdl = world_get_camera(m_mffInstHdl, camera);
		if(camHdl == nullptr)
			throw std::runtime_error("Camera '" + std::string(camera) + "' does not exist");
		ScenarioHdl scenarioHdl = world_create_scenario(m_mffInstHdl, scenarioIter->name.GetString());
		if(scenarioHdl == nullptr)
			throw std::runtime_error("Failed to create scenario");

		if(!scenario_set_camera(m_mffInstHdl, scenarioHdl, camHdl))
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
					if(!scenario_add_light(m_mffInstHdl, scenarioHdl, nameIter->second))
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
			scenario_reserve_custom_object_properties(scenarioHdl, objectsIter->value.MemberCount());
			assertObject(m_state, objectsIter->value);
			for(auto objIter = objectsIter->value.MemberBegin(); objIter != objectsIter->value.MemberEnd(); ++objIter) {
				StringView objectName = objIter->name.GetString();
				m_state.objectNames.push_back(&objectName[0u]);
				const Value& object = objIter->value;
				assertObject(m_state, object);
				ObjectHdl objHdl = world_get_object(m_mffInstHdl, &objectName[0u]);
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
				if(const auto tessIter = get(m_state, object, "tessellation", false); tessIter != object.MemberEnd()) {
					const bool adaptive = read_opt<bool>(m_state, tessIter->value, "adaptive", true);
					const bool usePhong = read_opt<bool>(m_state, tessIter->value, "usePhong", true);
					if(!scenario_set_object_adaptive_tessellation(scenarioHdl, objHdl, adaptive))
						throw std::runtime_error("Failed to set adaptive tesselation of object '" + std::string(objectName) + "'");
					if(!scenario_set_object_phong_tessellation(scenarioHdl, objHdl, usePhong))
						throw std::runtime_error("Failed to set phong tesselation of object '" + std::string(objectName) + "'");
					if(const auto levelIter = get(m_state, tessIter->value, "level", false); levelIter != tessIter->value.MemberEnd()) {
						const auto level = read<float>(m_state, levelIter);
						if(!scenario_set_object_tessellation_level(scenarioHdl, objHdl, level))
							throw std::runtime_error("Failed to set tesselation level of object '" + std::string(objectName) + "'");
					}
				}

				m_state.objectNames.pop_back();
			}
			m_state.objectNames.pop_back();
		}

		// Read Instance LOD and masking information
		if(auto instancesIter = get(m_state, scenario, "instanceProperties", false);
		   instancesIter != scenario.MemberEnd()) {
			m_state.objectNames.push_back(instancesIter->name.GetString());
			scenario_reserve_custom_instance_properties(scenarioHdl, instancesIter->value.MemberCount());
			assertObject(m_state, instancesIter->value);
			for(auto instIter = instancesIter->value.MemberBegin(); instIter != instancesIter->value.MemberEnd(); ++instIter) {
				StringView instName = instIter->name.GetString();
				m_state.objectNames.push_back(&instName[0u]);
				const Value& instance = instIter->value;
				assertObject(m_state, instance);

				// Set masking for both animated and non-animated instances
				u32 frameCount;
				world_get_frame_count(m_mffInstHdl, &frameCount);
				if(const auto instIter = instances.find(&instName[0u]); instIter != instances.end()) {
					InstanceHdl instHdl = instIter->second.handle;
					if(instHdl != nullptr) {
						for(u32 frame = 0; frame < frameCount; ++frame) {
							// Read LoD
							if(auto lodIter = get(m_state, instance, "lod", false); lodIter != instance.MemberEnd())
								if(!scenario_set_instance_lod(scenarioHdl, instHdl, read<u32>(m_state, lodIter)))
									throw std::runtime_error("Failed to set LoD level of instance '" + std::string(instName) + "'");
							// Read masking information
							if(auto maskIter = get(m_state, instance, "mask", false); maskIter != instance.MemberEnd()
							   && read<bool>(m_state, maskIter))
								if(!scenario_mask_instance(scenarioHdl, instHdl))
									throw std::runtime_error("Failed to set mask of instance '" + std::string(instName) + "'");
						}

						// Read LoD
						if(auto lodIter = get(m_state, instance, "lod", false); lodIter != instance.MemberEnd())
							if(!scenario_set_instance_lod(scenarioHdl, instHdl, read<u32>(m_state, lodIter)))
								throw std::runtime_error("Failed to set LoD level of instance '" + std::string(instName) + "'");
						// Read masking information
						if(auto maskIter = get(m_state, instance, "mask", false); maskIter != instance.MemberEnd()
						   && read<bool>(m_state, maskIter))
							if(!scenario_mask_instance(scenarioHdl, instHdl))
								throw std::runtime_error("Failed to set mask of instance '" + std::string(instName) + "'");
					}
				}
				m_state.objectNames.pop_back();
			}
			m_state.objectNames.pop_back();
		}

		// Associate binary with JSON material names
		auto materialsIter = get(m_state, scenario, "materialAssignments");
		m_state.objectNames.push_back(materialsIter->name.GetString());
		assertObject(m_state, materialsIter->value);
		scenario_reserve_material_slots(scenarioHdl, binMatNames.size());
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
		if(!world_finalize_scenario(m_mffInstHdl, scenarioHdl, &sanityMsg))
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


bool JsonLoader::load_file(fs::path& binaryFile) {
	using namespace rapidjson;
	auto scope = Profiler::loader().start<CpuProfileState>("JsonLoader::load_file");

	this->clear_state();
	logInfo("[JsonLoader::load_file] Parsing scene file '", m_filePath.u8string(), "'");

	sprintf(m_loadingStage.data(), "Loading JSON%c", '\0');
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
	bool hasWorldToInstTrans = true;
	auto versionIter = get(m_state, document, "version", false);
	if(versionIter == document.MemberEnd()) {
		logWarning("[JsonLoader::load_file] Scene file: no version specified (current one assumed)");
	} else {
		m_version = FileVersion{ read<const char*>(m_state, versionIter) };
		if(m_version > CURRENT_FILE_VERSION)
			logWarning("[JsonLoader::load_file] Scene file: version mismatch (",
					   m_version, "(file) vs ", CURRENT_FILE_VERSION, "(current))");
		hasWorldToInstTrans = m_version >= INVERTEX_TRANSMAT_FILE_VERSION;
		m_absoluteCamNearFar = m_version >= ABSOLUTE_CAM_NEAR_FAR_FILE_VERSION;
		logInfo("[JsonLoader::load_file] Detected file version '", m_version, "'");
	}
	// Binary file path
	binaryFile = read<const char*>(m_state, get(m_state, document, "binary"));
	if(binaryFile.empty())
		throw std::runtime_error("Scene file has an empty binary file path");
	logInfo("[JsonLoader::load_file] Detected binary file path '", binaryFile.u8string(), "'");
	// Make the file path absolute
	if(binaryFile.is_relative())
		binaryFile = fs::canonical(m_filePath.parent_path() / binaryFile);
	if(!fs::exists(binaryFile)) {
		logError("[JsonLoader::load_file] Scene file: specifies a binary file that doesn't exist ('",
				 binaryFile.u8string(), "'");
		throw std::runtime_error("Binary file '" + binaryFile.u8string() + "' does not exist");
	}
	// Tessellation level
	const float initTessLevel = read_opt<float>(m_state, document, "initTessellationLevel", 0u);
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
	logInfo("[JsonLoader::load_file] Detected global LoD '", defaultGlobalLod, "'");

	sprintf(m_loadingStage.data(), "Parsing object properties%c", '\0');
	
	// First we have to parse how many object/instance properties there are
	std::size_t objPropCount = 0u;
	std::size_t instPropCount = 0u;
	for(auto scenIter = m_scenarios->value.MemberBegin(); scenIter != m_scenarios->value.MemberEnd(); ++scenIter) {
		const auto& scenario = scenIter->value;
		if(const auto objPropsIter = get(m_state, scenario, "objectProperties", false);
		   objPropsIter != scenario.MemberEnd())
			objPropCount += objPropsIter->value.MemberCount();
		if(const auto instPropsIter = get(m_state, scenario, "instanceProperties", false);
		   instPropsIter != scenario.MemberEnd())
			instPropCount += instPropsIter->value.MemberCount();
	}
	util::FixedHashMap<StringView, u32> defaultObjectLods{ objPropCount };
	util::FixedHashMap<StringView, binary::InstanceMapping> defaultInstanceLods{ instPropCount };
	// Now we can parse the properties themselves
	// We have to start with the default scenario, since it defines the LoD level in case of ambiguity
	m_state.objectNames.push_back(m_defaultScenario.data());
	parse_object_instance_properties(m_state, defScen, defaultObjectLods, defaultInstanceLods, defaultGlobalLod);
	m_state.objectNames.pop_back();
	// Now the rest of the scenarios
	for(auto scenIter = m_scenarios->value.MemberBegin(); scenIter != m_scenarios->value.MemberEnd(); ++scenIter) {
		m_state.objectNames.push_back(scenIter->name.GetString());
		parse_object_instance_properties(m_state, scenIter->value, defaultObjectLods, defaultInstanceLods, defaultGlobalLod);
		m_state.objectNames.pop_back();
	}
	const bool deinstance = read_opt<bool>(m_state, document, "deinstance", false);
	const bool noDefaultInstances = read_opt<bool>(m_state, document, "noDefaultInstances", false);
	// Load the binary file before we load the rest of the JSON
	if(!m_binLoader.load_file(binaryFile, defaultGlobalLod, defaultObjectLods, defaultInstanceLods,
							  deinstance, hasWorldToInstTrans, noDefaultInstances))
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
		sprintf(m_loadingStage.data(), "Checking world sanity%c", '\0');
		if(!world_finalize(m_mffInstHdl, util::pun<Vec3>(m_binLoader.get_bounding_box().min),
						   util::pun<Vec3>(m_binLoader.get_bounding_box().max), &sanityMsg))
			throw std::runtime_error("World did not pass sanity check: " + std::string(sanityMsg));
		// Scenarios
		m_state.current = ParserState::Level::ROOT;
		if(!load_scenarios(m_binLoader.get_material_names(), defaultInstanceLods))
			return false;
		// Load the default scenario
		m_state.reset();
		ScenarioHdl defScenHdl = world_find_scenario(m_mffInstHdl, &m_defaultScenario[0u]);
		if(defScenHdl == nullptr)
			throw std::runtime_error("Cannot find the default scenario '" + std::string(m_defaultScenario) + "'");

		sprintf(m_loadingStage.data(), "Loading initial scenario%c", '\0');
		auto scope = Profiler::loader().start<CpuProfileState>("JsonLoader::load_file - load default scenario", ProfileLevel::LOW);
		if(!world_load_scenario(m_mffInstHdl, defScenHdl))
			throw std::runtime_error("Cannot load the default scenario '" + std::string(m_defaultScenario) + "'");
		// Check if we should tessellate initially, indicated by a non-zero max. level
		if(initTessLevel > 0.f) {
			sprintf(m_loadingStage.data(), "Performing initial tessellation%c", '\0');
			world_set_tessellation_level(m_mffInstHdl, initTessLevel);
			scene_request_retessellation(m_mffInstHdl);
		}
	} catch(const std::runtime_error& e) {
		throw std::runtime_error(m_state.get_parser_level() + ": " + e.what());
	}
	return true;
}


} // namespace mff_loader::json
