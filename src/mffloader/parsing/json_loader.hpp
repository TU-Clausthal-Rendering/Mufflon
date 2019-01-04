#pragma once

#include "util/filesystem.hpp"
#include "json_helper.hpp"
#include "binary.hpp"
#include "core/export/interface.h"
#include <ei/3dtypes.hpp>
#include <rapidjson/document.h>
#include <atomic>
#include <string>
#include <map>

namespace loader::json {

class JsonException : public std::exception {
public:
	JsonException(const std::string& str, rapidjson::ParseResult res);
	virtual const char* what() const override {
		return m_error.c_str();
	}

private:
	std::string m_error;
};


class JsonLoader {
public:
	static constexpr const char FILE_VERSION[] = "1.0";
	static constexpr float DEFAULT_NEAR_PLANE = 1.e-4f;
	static constexpr float DEFAULT_FAR_PLANE = 2.f;

	JsonLoader(fs::path file) :
		m_filePath(fs::canonical(file))
	{
		if(!fs::exists(m_filePath))
			throw std::runtime_error("JSON file '" + m_filePath.string() + "' doesn't exist");
	}

	// This may be called from a different thread and leads to the current load being cancelled
	void abort_load() { m_abort = true; m_binLoader.abort_load(); }
	bool was_aborted() { return m_abort; }

	bool load_file();
	void clear_state();

private:
	TextureHdl load_texture(const char* name);
	MaterialParams* load_material(rapidjson::Value::ConstMemberIterator matIter);
	void free_material(MaterialParams* mat);
	bool load_cameras(const ei::Box& aabb);
	bool load_lights();
	bool load_materials();
	bool load_scenarios(const std::vector<std::string>& binMatNames);

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
	std::map<std::string, MaterialHdl, std::less<>> m_materialMap;
	ParserState m_state;

	binary::BinaryLoader m_binLoader;

	// These are for aborting a load and keeping track of progress
	std::atomic_bool m_abort = false;
};

} // namespace loader::json