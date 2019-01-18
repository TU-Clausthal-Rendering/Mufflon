#pragma once

#include "util/filesystem.hpp"
#include <rapidjson/document.h>
namespace mff_loader::exprt {

class SceneExporter
{
public:
	static constexpr const char FILE_VERSION[] = "1.0";

	SceneExporter(fs::path file) :
		m_filePath(fs::canonical(file))
	{}

	bool save_scene();
private:
	bool save_cameras(rapidjson::Document& document);
	bool save_lights(rapidjson::Document& document);
	bool save_materials(rapidjson::Document& document);
	bool save_scenarios(rapidjson::Document& document);

	const fs::path m_filePath;
	
};

}// namespace mff_loader::exprt