#pragma once

#include "util/filesystem.hpp"
#include <rapidjson/document.h>
#include "core/export/interface.h"

namespace mff_loader::exprt {

class SceneExporter
{
public:
	static constexpr const char FILE_VERSION[] = "1.0";

	SceneExporter(fs::path file) :
		m_filePath(fs::canonical(file))
	{}

	bool save_scene() const;
private:
	bool save_cameras(rapidjson::Document& document) const;
	bool save_lights(rapidjson::Document& document) const;
	bool save_materials(rapidjson::Document& document) const;
	bool save_material(rapidjson::Document& document, MaterialParams materialParams) const;
	bool save_scenarios(rapidjson::Document& document) const;

	rapidjson::Value save_in_array(Vec3 value, rapidjson::Document& document) const;

	const fs::path m_filePath;
	
};

}// namespace mff_loader::exprt