#pragma once

#include "util/filesystem.hpp"
#include <rapidjson/document.h>
#include "core_interface.h"
#include <rapidjson/prettywriter.h>
#include <vector>

namespace mff_loader::exprt {

class SceneExporter
{
public:
	static constexpr const char FILE_VERSION[] = "1.0";

	SceneExporter(MufflonInstanceHdl mffInstHdl, fs::path fileDestinationPath, fs::path mffPath) :
		m_mffInstHdl{ mffInstHdl },
		m_fileDestinationPath(fileDestinationPath),
		m_mffPath(mffPath)
	{}

	bool save_scene() const;
private:
	bool save_cameras(rapidjson::Document& document) const;
	bool save_lights(rapidjson::Document& document) const;
	bool save_materials(rapidjson::Document& document) const;
	rapidjson::Value save_material(const MaterialParams&, rapidjson::Document& document ) const;
	bool save_scenarios(rapidjson::Document& document) const;

	bool add_member_from_texture_handle(const TextureHdl& textureHdl, const std::string& memberName, rapidjson::Value& saveIn, rapidjson::Document& document) const;

	rapidjson::Value store_in_array(Vec3 value, rapidjson::Document& document) const;
	rapidjson::Value store_in_array(Vec2 value, rapidjson::Document& document) const;
	template<typename T>
	rapidjson::Value store_in_array(std::vector<T> value, rapidjson::Document& document) const;
	rapidjson::Value store_in_array_from_float_string(std::string floatString, rapidjson::Document& document) const;

	rapidjson::Value store_in_string_relative_to_destination_path(const fs::path& path, rapidjson::Document& document) const;

	MufflonInstanceHdl m_mffInstHdl;
	const fs::path m_fileDestinationPath;
	const fs::path m_mffPath;
};
}// namespace mff_loader::exprt
