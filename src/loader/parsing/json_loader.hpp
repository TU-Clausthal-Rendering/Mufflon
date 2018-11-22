#pragma once

#include "json_helper.hpp"
#include "util/filesystem.hpp"
#include <rapidjson/document.h>
#include <string>

namespace loader::json {

class JsonLoader {
public:
	static constexpr const char FILE_VERSION[] = "1.0";

	JsonLoader(fs::path file) :
		m_filePath(fs::canonical(file))
	{
		if(!fs::exists(m_filePath))
			throw std::runtime_error("JSON file '" + m_filePath.string() + "' doesn't exist");
	}

	void load_file();
	void clear_state();

private:
	void load_cameras();
	void load_lights();
	void load_materials();
	void load_scenarios();

	const fs::path m_filePath;
	std::string m_jsonString;
	rapidjson::Document m_document;
	rapidjson::Value::ConstMemberIterator m_cameras;
	rapidjson::Value::ConstMemberIterator m_lights;
	rapidjson::Value::ConstMemberIterator m_materials;
	rapidjson::Value::ConstMemberIterator m_scenarios;
	std::string_view m_version;
	fs::path m_binaryFile;
	std::string_view m_defaultScenario;
	ParserState m_state;
};

} // namespace loader::json