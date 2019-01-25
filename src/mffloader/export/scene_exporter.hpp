#pragma once

#include "util/filesystem.hpp"
#include <rapidjson/document.h>
#include "core/export/interface.h"

namespace mff_loader::exprt {

class SceneExporter
{
public:
	static constexpr const char FILE_VERSION[] = "1.0";

	SceneExporter(fs::path fileDestinationPath, fs::path mffPath) :
		m_fileDestinationPath(fs::canonical(fileDestinationPath)),
		m_mffPath(mffPath)
	{}

	bool save_scene() const;
private:
	bool save_cameras(rapidjson::Document& document) const;
	bool save_lights(rapidjson::Document& document) const;
	bool save_materials(rapidjson::Document& document) const;
	rapidjson::Value save_material( MaterialParams materialParams, rapidjson::Document& document ) const;
	bool save_scenarios(rapidjson::Document& document) const;

	rapidjson::Value store_in_array(Vec3 value, rapidjson::Document& document) const;
	rapidjson::Value store_in_array(Vec2 value, rapidjson::Document& document) const;

	rapidjson::Value store_in_string_relative_to_destination_path(const fs::path& path, rapidjson::Document& document) const;


	const fs::path m_fileDestinationPath;
	const fs::path m_mffPath;
};

}// namespace mff_loader::exprt