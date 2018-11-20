#include "util/assert.hpp"
#include "util/log.hpp"
#include "util/punning.hpp"
#include "util/degrad.hpp"
#include "core/export/interface.h"
#include <rapidjson/document.h>
#include <rapidjson/rapidjson.h>
#include <ei/vector.hpp>
#include <fstream>
#include <vector>

// Make use of filepaths
#if !defined(__cpp_lib_filesystem)
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;
#else // !defined(__cpp_lib_filesystem)
#include <filesystem>
namespace fs = std::filesystem;
#endif // !defined(__cpp_lib_filesystem)

#define FUNCTION_NAME __func__

inline constexpr const char FILE_VERSION[] = "1.0";

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

bool read_path(const rapidjson::Value::ConstMemberIterator& object,
			   std::vector<ei::Vec3>& path,
			   std::string_view objectName, const char* pathName) {
	using namespace rapidjson;
	Value::ConstMemberIterator pathIter = object->value.FindMember(pathName);
	if(pathIter == object->value.MemberEnd()) {
		logWarning("[", FUNCTION_NAME, "] Scene file: ", objectName, " '",
				   object->name.GetString(), "' is missing ", pathName);
		return false;
	}
	if(!pathIter->value.IsArray()) {
		logWarning("[", FUNCTION_NAME, "] Scene file: ", pathName, " of ", objectName, " '",
				   object->name.GetString(), "' has invalid type (must be array or array of array)");
		return false;
	}
	for(auto pathValIter = pathIter->value.MemberBegin(); pathValIter != pathIter->value.MemberEnd(); ++pathValIter) {
		if(!pathValIter->value.IsArray()) {
			logWarning("[", FUNCTION_NAME, "] Scene file: ignoring non-array path segment of ", objectName, " '",
					   object->name.GetString(), "'s ", pathName, "");
			continue;
		}
		if(pathValIter->value.Size() != 3u) {
			logWarning("[", FUNCTION_NAME, "] Scene file: ignoring path segment of ", objectName, " '",
					   object->name.GetString(), "'s ", pathName, " with invalid number of elements (",
					   pathValIter->value.Size(), ")");
			continue;
		}
		if(!pathValIter->value[0u].IsNumber()) {
			logWarning("[", FUNCTION_NAME, "] Scene file: ignoring path segment of ", objectName, " '",
					   object->name.GetString(), "'s ", pathName, " with invalid element type ");
			continue;
		}
		path.emplace_back(pathValIter->value[0u].GetFloat(), pathValIter->value[1u].GetFloat(),
							 pathValIter->value[2u].GetFloat());
	}
	return true;
}

bool read_vec3(const rapidjson::Value::ConstMemberIterator& object,
			   ei::Vec3& vec, std::string_view objectName, const char* vecName) {
	using namespace rapidjson;
	Value::ConstMemberIterator vecIter = object->value.FindMember(vecName);
	if(!vecIter->value.IsArray()) {
		logWarning("[", FUNCTION_NAME, "] Scene file: ", vecName, " of ", objectName, " '",
				   object->name.GetString(), "' is not an array");
		return false;
	}
	if(vecIter->value.Size() != 3u) {
		logWarning("[", FUNCTION_NAME, "] Scene file: ", vecName, " of ", objectName, " '",
				   object->name.GetString(), "' has invalid number of elements (",
				   vecIter->value.Size()), ")";
		return false;
	}
	if(!vecIter->value[0u].IsNumber()) {
		logWarning("[", FUNCTION_NAME, "] Scene file: ", vecName, " of ", objectName, " '",
				   object->name.GetString(), "' has invalid element type");
		return false;
	}
	vec.x = vecIter->value[0u].GetFloat();
	vec.y = vecIter->value[1u].GetFloat();
	vec.z = vecIter->value[2u].GetFloat();
	return true;
}

bool parse_cameras(const rapidjson::Document& document) {
	using namespace rapidjson;
	Value::ConstMemberIterator cameras = document.FindMember("cameras");
	if(cameras != document.MemberEnd()) {
		if(!cameras->value.IsObject()) {
			logError("[", FUNCTION_NAME, "] Scene file: doesn't specify cameras as key:value pairs");
			return false;
		}
		// Iterate all cameras and add them to the world container
		for(auto camera = cameras->value.MemberBegin(); camera != cameras->value.MemberEnd(); ++camera) {
			// Validity check
			if(camera->value.IsObject()) {
				logWarning("[", FUNCTION_NAME, "] Scene file: skipping non-object camera value '",
						   camera->name.GetString(), "'");
				continue;
			}
			// Get the camera type
			Value::ConstMemberIterator camType = camera->value.FindMember("type");
			if(camType == camera->value.MemberEnd()) {
				logWarning("[", FUNCTION_NAME, "] Scene file camera object '",
						   camera->name.GetString(), "' doesn't specify a type");
				continue;
			}
			if(!camType->value.IsString()) {
				logWarning("[", FUNCTION_NAME, "] Scene file: type of camera object '",
						   camera->name.GetString(), "' is not a string");
				continue;
			}

			// Parse common camera values
			std::vector<ei::Vec3> posPath;
			std::vector<ei::Vec3> viewDirPath;
			std::vector<ei::Vec3> upDirPath;
			// First is path
			if(!read_path(camera, posPath, "camera", "path"))
				continue;
			if(posPath.size() == 0u) {
				logWarning("[", FUNCTION_NAME, "] Scene file: camera '",
						   camera->name.GetString(), "' has empty or no valid elements in 'path'");
				continue;
			}
			// Second is viewdir
			if(!read_path(camera, viewDirPath, "camera", "viewDir"))
				continue;
			if(viewDirPath.size() == 0u) {
				logWarning("[", FUNCTION_NAME, "] Scene file: camera '",
						   camera->name.GetString(), "' has empty or no valid elements in 'viewDir'");
				continue;
			}
			if(viewDirPath.size() != posPath.size()) {
				logWarning("[", FUNCTION_NAME, "] Scene file: 'path' and 'viewDir' of camera '",
						   camera->name.GetString(), "' do not match in size (", posPath.size(), "(path) vs ",
						   viewDirPath.size(), ")");
				continue;
			}
			// Third is (optional) up
			if(camera->value.HasMember("up")) {
				if(!read_path(camera, posPath, "camera", "up"))
					continue;
				if(upDirPath.size() == 0u) {
					logWarning("[", FUNCTION_NAME, "] Scene file: camera '",
							   camera->name.GetString(), "' has empty or no valid elements in 'up'");
					continue;
				}
				if(upDirPath.size() != posPath.size()) {
					logWarning("[", FUNCTION_NAME, "] Scene file: 'path' and 'up' of camera '",
							   camera->name.GetString(), "' do not match in size (", posPath.size(), "(path) vs ",
							   upDirPath.size(), ")");
					continue;
				}
			} else {
				upDirPath.push_back(ei::Vec3{ 0, 1, 0 });
			}
			// Then come near and far plane (both optional)
			// The min value indicates later that they need to be multiplied with
			// the scene AABB diagonal
			float near = std::numeric_limits<float>::min();
			float far = std::numeric_limits<float>::min();
			Value::ConstMemberIterator nearIter = camera->value.FindMember("near");
			if(nearIter != camera->value.MemberEnd()) {
				if(!nearIter->value.IsNumber())
					logWarning("[", FUNCTION_NAME, "] Scene file: near plane of camera '",
							   camera->name.GetString(), "' is not a number");
				else
					near = nearIter->value.GetFloat();
			}
			Value::ConstMemberIterator farIter = camera->value.FindMember("far");
			if(farIter != camera->value.MemberEnd()) {
				if(!farIter->value.IsNumber())
					logWarning("[", FUNCTION_NAME, "] Scene file: far plane of camera '",
							   camera->name.GetString(), "' is not a number");
				else
					far = farIter->value.GetFloat();
			}

			// Check the type
			const char* type = camType->value.GetString();
			if(std::strncmp(type, "pinhole", 7) == 0) {
				// Pinhole camera
				float fovDegree = 25.f;
				// Default for FoV is 25 degrees
				Value::ConstMemberIterator fov = camera->value.FindMember("fov");
				if(fov != camera->value.MemberEnd()) {
					if(!fov->value.IsNumber())
						logWarning("[", FUNCTION_NAME, "] Scene file: fov of camera '",
								   camera->name.GetString(), "' is not a number");
					else
						fovDegree = fov->value.GetFloat();
				}

				// TODO: add entire path!
				if(posPath.size() > 1u)
					logWarning("[", FUNCTION_NAME, "] Scene file: camera paths are not supported yet");
				world_add_pinhole_camera(camera->name.GetString(), util::pun<Vec3>(posPath[0u]),
										 util::pun<Vec3>(viewDirPath[0u]),
										 util::pun<Vec3>(upDirPath[0u]), near, far,
										 static_cast<Radians>(Degrees(fovDegree)));
			} else if(std::strncmp(type, "focus", 5) == 0) {
				// TODO: Focus camera
				logWarning("[", FUNCTION_NAME, "] Scene file: Focus cameras are not supported yet");
			} else if(std::strncmp(type, "ortho", 5) == 0) {
				// TODO: Orthogonal camera
				logWarning("[", FUNCTION_NAME, "] Scene file: Focus cameras are not supported yet");
			} else {
				logWarning("[", FUNCTION_NAME, "] Scene file: camera object '",
						   camera->name.GetString(), "' has unknown type '", type, "'");
			}
		}
	} else {
		logError("[", FUNCTION_NAME, "] Scene file: is missing cameras");
		return false;
	}

	return true;
}

bool parse_lights(const rapidjson::Document& document) {
	using namespace rapidjson;

	Value::ConstMemberIterator lights = document.FindMember("lights");
	if(lights != document.MemberEnd()) {
		if(!lights->value.IsObject()) {
			logError("[", FUNCTION_NAME, "] Scene file: doesn't specify lights as key:value pairs");
			return false;
		}
		// Iterate all cameras and add them to the world container
		for(auto light = lights->value.MemberBegin(); light != lights->value.MemberEnd(); ++light) {
			// Validity check
			if(light->value.IsObject()) {
				logWarning("[", FUNCTION_NAME, "] Scene file: skipping non-object light value '",
						   light->name.GetString(), "'");
				continue;
			}
			// Get the camera type
			Value::ConstMemberIterator lightType = light->value.FindMember("type");
			if(lightType == light->value.MemberEnd()) {
				logWarning("[", FUNCTION_NAME, "] Scene file light object '",
						   light->name.GetString(), "' doesn't specify a type");
				continue;
			}
			if(!lightType->value.IsString()) {
				logWarning("[", FUNCTION_NAME, "] Scene file: type of light object '",
						   light->name.GetString(), "' is not a string");
				continue;
			}

			// Check the type
			const char* type = lightType->value.GetString();
			if(std::strncmp(type, "point", 5) == 0) {
				// Point light
				ei::Vec3 position;
				ei::Vec3 intensity;
				if(!read_vec3(light, position, "point light", "position"))
					continue;
				if(light->value.HasMember("flux")) {
					if(!read_vec3(light, intensity, "point light", "flux"))
						continue;
					// Convert flux to intensity
					intensity *= 1.f / (4.f * ei::PI);
				} else {
					if(!read_vec3(light, intensity, "point light", "intensity"))
						continue;
				}
				Value::ConstMemberIterator scale = light->value.FindMember("scale");
				if(scale != light->value.MemberEnd()) {
					if(!scale->value.IsNumber())
						logWarning("[", FUNCTION_NAME, "] Scene file: scale of point light '",
								   light->name.GetString(), "' is not a number");
					else
						intensity *= scale->value.GetFloat();
				}
				if(world_add_point_light(light->name.GetString(), util::pun<Vec3>(position),
										 util::pun<Vec3>(intensity)) == nullptr) {
					logError("[", FUNCTION_NAME, "] Scene file: failed to add point light '",
							 light->name.GetString(), "'");
					return false;
				}
			} else if(std::strncmp(type, "directional", 11) == 0) {
				// Directional light
				ei::Vec3 direction;
				ei::Vec3 radiance;
				if(!read_vec3(light, direction, "directional light", "direction"))
					continue;
				if(!read_vec3(light, radiance, "directional light", "radiance"))
					continue;
				Value::ConstMemberIterator scale = light->value.FindMember("scale");
				if(scale != light->value.MemberEnd()) {
					if(!scale->value.IsNumber())
						logWarning("[", FUNCTION_NAME, "] Scene file: scale of directional light '",
								   light->name.GetString(), "' is not a number");
					else
						radiance *= scale->value.GetFloat();
				}
				if(world_add_directional_light(light->name.GetString(), util::pun<Vec3>(direction),
											   util::pun<Vec3>(radiance)) == nullptr) {
					logError("[", FUNCTION_NAME, "] Scene file: failed to add directional light '",
							 light->name.GetString(), "'");
					return false;
				}
			} else if(std::strncmp(type, "spot", 4) == 0) {
				// Spot light
				ei::Vec3 position;
				ei::Vec3 direction;
				ei::Vec3 intensity;
				float angle;
				float falloff;
				if(!read_vec3(light, position, "spot light", "position"))
					continue;
				if(!read_vec3(light, direction, "spot light", "direction"))
					continue;
				if(!read_vec3(light, intensity, "spot light", "intensity"))
					continue;
				Value::ConstMemberIterator scale = light->value.FindMember("scale");
				if(scale != light->value.MemberEnd()) {
					if(!scale->value.IsNumber())
						logWarning("[", FUNCTION_NAME, "] Scene file: scale of spot light '",
								   light->name.GetString(), "' is not a number");
					else
						intensity *= scale->value.GetFloat();
				}
				// Opening angle
				if(light->value.HasMember("cosWidth")) {
					Value::ConstMemberIterator angleIter = light->value.FindMember("cosWidth");
					if(!angleIter->value.IsNumber()) {
						logWarning("[", FUNCTION_NAME, "] Scene file: opening angle of spot light '",
								   light->name.GetString(), "' is not a number");
						continue;
					}
					angle = std::acos(angleIter->value.GetFloat());
				} else {
					Value::ConstMemberIterator angleIter = light->value.FindMember("width");
					if(angleIter == light->value.MemberEnd()) {
						logWarning("[", FUNCTION_NAME, "] Scene file: spot light '",
								   light->name.GetString(), "' is missing opening angle");
						continue;
					}
					if(!angleIter->value.IsNumber()) {
						logWarning("[", FUNCTION_NAME, "] Scene file: opening angle of spot light '",
								   light->name.GetString(), "' is not a number");
						continue;
					}
					angle = angleIter->value.GetFloat();
				}
				// Falloff start
				if(light->value.HasMember("cosFalloffStart")) {
					Value::ConstMemberIterator falloffIter = light->value.FindMember("cosFalloffStart");
					if(!falloffIter->value.IsNumber()) {
						logWarning("[", FUNCTION_NAME, "] Scene file: falloff start of spot light '",
								   light->name.GetString(), "' is not a number");
						continue;
					}
					falloff = std::acos(falloffIter->value.GetFloat());
				} else {
					Value::ConstMemberIterator falloffIter = light->value.FindMember("falloffStart");
					if(falloffIter == light->value.MemberEnd()) {
						logWarning("[", FUNCTION_NAME, "] Scene file: spot light '",
								   light->name.GetString(), "' is missing falloff start");
						continue;
					}
					if(!falloffIter->value.IsNumber()) {
						logWarning("[", FUNCTION_NAME, "] Scene file: falloff start of spot light '",
								   light->name.GetString(), "' is not a number");
						continue;
					}
					falloff = falloffIter->value.GetFloat();
				}

				if(world_add_spot_light(light->name.GetString(), util::pun<Vec3>(position),
										util::pun<Vec3>(direction), util::pun<Vec3>(intensity),
										angle, falloff) == nullptr) {
					logError("[", FUNCTION_NAME, "] Scene file: failed to add spot light '",
							 light->name.GetString(), "'");
					return false;
				}
			} else if(std::strncmp(type, "envmap", 6) == 0) {
				// Environment-map light
				const char* texPath = nullptr;
				float texScale = 1.f;
				Value::ConstMemberIterator envmap = light->value.FindMember("map");
				if(envmap == light->value.MemberEnd()) {
					logWarning("[", FUNCTION_NAME, "] Scene file: envmap light '",
							   light->name.GetString(), "' is missing an envmap");
					continue;
				}
				if(!envmap->value.IsString()) {
					logWarning("[", FUNCTION_NAME, "] Scene file: envmap of envmap light '",
							   light->name.GetString(), "' is not a string");
					continue;
				}
				texPath = envmap->value.GetString();
				if(std::strlen(texPath) == 0u) {
					logWarning("[", FUNCTION_NAME, "] Scene file: envmap of envmap light '",
							   light->name.GetString(), "' is empty");
					continue;
				}
				// TODO: load the texture
				TextureHdl texture = world_add_texture(texPath, 0u, 0u, 0u, TextureFormat::FORMAT_R8U,
													   TextureSampling::SAMPLING_NEAREST, false, nullptr);
				if(texture == nullptr) {
					logError("[", FUNCTION_NAME, "] Scene file: could not load envmap of envmap light '",
							 light->name.GetString(), "'");
					return false;
				}
				// TODO: incorporate scale
				if(world_add_envmap_light(light->name.GetString(), texture) == nullptr) {
					logError("[", FUNCTION_NAME, "] Scene file: failed to add envmap light '",
							 light->name.GetString(), "'");
					return false;
				}
			} else if(std::strncmp(type, "goniometric", 11) == 0) {
				// TODO: goniometric light
				logWarning("[", FUNCTION_NAME, "] Scene file: Goniometric lights are not supported yet");
			} else {
				logWarning("[", FUNCTION_NAME, "] Scene file: light object '",
						   light->name.GetString(), "' has unknown type '", type, "'");
			}
		}
	}

	return true;
}

bool parse_materials(const rapidjson::Document& document) {
	using namespace rapidjson;

	Value::ConstMemberIterator scenarios = document.FindMember("materials");
	if(scenarios != document.MemberEnd()) {
		// TODO
	} else {
		logError("[", FUNCTION_NAME, "] Scene file: is missing scenarios");
		return false;
	}
	return true;
}

bool parse_scenarios(const rapidjson::Document& document) {
	using namespace rapidjson;
	Value::ConstMemberIterator scenarios = document.FindMember("scenarios");
	if(scenarios != document.MemberEnd()) {
		if(!scenarios->value.IsObject()) {
			logError("[", FUNCTION_NAME, "] Scene file: doesn't specify scenarios as key:value pairs");
			return false;
		}
		// Iterate all cameras and add them to the world container
		for(auto scenario = scenarios->value.MemberBegin(); scenario != scenarios->value.MemberEnd(); ++scenario) {
			// Validity check
			if(scenario->value.IsObject()) {
				logWarning("[", FUNCTION_NAME, "] Scene file: skipping non-object scenario value '",
						   scenario->name.GetString(), "'");
				continue;
			}
			CameraHdl camHdl = nullptr;
			IVec2 resolution{};
			std::size_t globalLod = 0u;
			// Read the camera
			Value::ConstMemberIterator camera = scenario->value.FindMember("camera");
			if(camera == scenario->value.MemberEnd()) {
				logWarning("[", FUNCTION_NAME, "] Scene file: scenario '",
						   scenario->name.GetString(), "' is missing a camera");
				continue;
			}
			if(!camera->value.IsString()) {
				logWarning("[", FUNCTION_NAME, "] Scene file: camera for scenario '",
						   scenario->name.GetString(), "' is not a string");
				continue;
			}
			camHdl = world_get_camera(camera->value.GetString());
			if(camHdl == nullptr) {
				logWarning("[", FUNCTION_NAME, "] Scene file: camera for scenario '",
						   scenario->name.GetString(), "' is invalid");
				continue;
			}
			// Read the resolution
			Value::ConstMemberIterator res = scenario->value.FindMember("resolution");
			if(res == scenario->value.MemberEnd()) {
				logWarning("[", FUNCTION_NAME, "] Scene file: scenario '",
						   scenario->name.GetString(), "' is missing a resolution");
				continue;
			}
			if(!res->value.IsArray()) {
				logWarning("[", FUNCTION_NAME, "] Scene file: resolution for scenario '",
						   scenario->name.GetString(), "' is not an array");
				continue;
			}
			if(res->value.Size() != 2u) {
				logWarning("[", FUNCTION_NAME, "] Scene file: resolution for scenario '",
						   scenario->name.GetString(), "' has invalid number of elements (",
						   res->value.Size(), ")");
				continue;
			}
			if(!res->value[0u].IsNumber()) {
				logWarning("[", FUNCTION_NAME, "] Scene file: resolution for scenario '",
						   scenario->name.GetString(), "' has invalid element type");
				continue;
			}
			resolution.x = res->value[0u].GetInt();
			resolution.y = res->value[1u].GetInt();
			// TODO: check for negative res?
			// Read the global LoD
			Value::ConstMemberIterator lod = scenario->value.FindMember("lod");
			if(lod != scenario->value.MemberEnd()) {
				if(!lod->value.IsNumber()) {
					logWarning("[", FUNCTION_NAME, "] Scene file: global LoD for scenario '",
							   scenario->name.GetString(), "' is not a number");
				} else {
					globalLod = lod->value.GetInt();
				}
			}

			// Create the scenario and insert into world
			ScenarioHdl scenHdl = world_create_scenario(scenario->name.GetString());
			if(scenHdl == nullptr) {
				logError("[", FUNCTION_NAME, "] Scene file: failed to create scenario '",
						 scenario->name.GetString(), "'");
				return false;
			}
			if(!scenario_set_camera(scenHdl, camHdl)) {
				logError("[", FUNCTION_NAME, "] Scene file: failed to set camera of scenario '",
						 scenario->name.GetString(), "'");
				return false;
			}
			if(!scenario_set_resolution(scenHdl, resolution.x, resolution.y)) {
				logError("[", FUNCTION_NAME, "] Scene file: failed to set resolution of scenario '",
						 scenario->name.GetString(), "'");
				return false;
			}
			if(!scenario_set_global_lod_level(scenHdl, globalLod)) {
				logError("[", FUNCTION_NAME, "] Scene file: failed to set LoD of scenario '",
						 scenario->name.GetString(), "'");
				return false;
			}
			// Parse and set the lights
			Value::ConstMemberIterator lights = scenario->value.FindMember("lights");
			if(lights != scenario->value.MemberEnd()) {
				if(!lights->value.IsArray()) {
					logWarning("[", FUNCTION_NAME, "] Scene file: lights for scenario '",
							   scenario->name.GetString(), "' is not an array");
				} else if(lights->value.Size() > 0) {
					if(!lights->value[0u].IsString()) {
						logWarning("[", FUNCTION_NAME, "] Scene file: lights for scenario '",
								   scenario->name.GetString(), "' has invalid element type");
					} else {
						for(auto lightIter = lights->value.MemberBegin(); lightIter != lights->value.MemberEnd(); ++lightIter) {
							const char* lightName = lightIter->value.GetString();
							if(!scenario_add_light(scenHdl, lightName)) {
								logWarning("[", FUNCTION_NAME, "] Scene file: failed to add light '",
										   lightName, "' to scenario '", scenario->name.GetString(), "'");
							}
						}
					}
				}
			}
			// TODO Read the material/object names/props
		}
	} else {
		logError("[", FUNCTION_NAME, "] Scene file: is missing scenarios");
		return false;
	}

	return true;
}

bool parse_json_file(fs::path filePath) {
	using namespace rapidjson;

	logInfo("[", FUNCTION_NAME, "] Parsing scene file '", filePath.string(), "'");

	// Binary file path
	fs::path binaryFile;
	// JSON text
	std::string jsonString = read_file(filePath);
	
	Document document;
	document.Parse(jsonString.c_str());

	// Parse our file specification
	if(!document.IsObject()) {
		logError("[", FUNCTION_NAME, "] Scene file: doesn't have root object");
		return false;
	}

	// Version
	Value::ConstMemberIterator version = document.FindMember("version");
	if(version != document.MemberEnd()) {
		if(!version->value.IsString() || version->value.GetString() != FILE_VERSION) {
			logWarning("[", FUNCTION_NAME, "] Scene file: doesn't match current version (",
					   version->value.GetString(), "(file) vs ", FILE_VERSION, "(current))");
		}
		// TODO: do something with the version
	} else {
		logWarning("[", FUNCTION_NAME, "] Scene file: is missing a version");
	}
	// Binary file path
	Value::ConstMemberIterator binary = document.FindMember("binary");
	if(binary != document.MemberEnd()) {
		if(!binary->value.IsString()) {
			logError("[", FUNCTION_NAME, "] Scene file: contains a non-string binary file path");
			return false;
		}
		binaryFile = fs::path(binary->value.GetString());
		if(binaryFile.empty()) {
			logError("[", FUNCTION_NAME, "] Scene file: has an empty binary file path");
			return false;
		}
		// Make the file path absolute
		if(binaryFile.is_relative())
			binaryFile = filePath.parent_path() / binaryFile;
		if(!fs::exists(binaryFile)) {
			logError("[", FUNCTION_NAME, "] Scene file: specifies a binary file that doesn't exist");
			return false;
		}
	} else {
		logError("[", FUNCTION_NAME, "] Scene file: is missing a binary file path");
		return false;
	}
	// Cameras
	if(!parse_cameras(document))
		return false;
	// Lights
	if(!parse_lights(document))
		return false;
	// Materials
	if(!parse_materials(document))
		return false;
	// Scenarios
	if(!parse_scenarios(document))
		return false;

	// TODO: parse binary file

	return true;
}

} // namespace

bool load_scene_file(const char* path) {
	fs::path filepath(path);

	// Perform some error checking
	if (!fs::exists(filepath)) {
		logError("[", FUNCTION_NAME, "] File '", fs::canonical(path).string(), "' does not exist");
		return false;
	}
	if (fs::is_directory(filepath)) {
		logError("[", FUNCTION_NAME, "] Path '", fs::canonical(path).string(), "' is a directory, not a file");
		return false;
	}
	if (filepath.extension() != ".json")
		logWarning("[", FUNCTION_NAME, "] Scene file does not end with '.json'; attempting to parse it anyway");


	return true;
}