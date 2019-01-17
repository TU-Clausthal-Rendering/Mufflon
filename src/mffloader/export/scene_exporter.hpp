#pragma once

#include "util/filesystem.hpp"
#include <map>
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

	bool save_scene();
private:

	const fs::path m_filePath;
	rapidjson::Value::ConstMemberIterator m_cameras;
	rapidjson::Value::ConstMemberIterator m_lights;
	rapidjson::Value::ConstMemberIterator m_materials;
	rapidjson::Value::ConstMemberIterator m_scenarios;
	std::string_view m_defaultScenario;
	std::map<std::string, MaterialHdl, std::less<>> m_materialMap;
};

}// namespace mff_loader::exprt