#pragma once

#include "util/filesystem.hpp"
#include "util/string_view.hpp"
#include "json_helper.hpp"
#include "binary.hpp"
#include "core_interface.h"
#include "util/fixed_hashmap.hpp"
#include <ei/3dtypes.hpp>
#include <rapidjson/document.h>
#include <atomic>
#include <string>
#include <map>
#include <optional>
#include <tuple>

namespace mff_loader::json {

class JsonException : public std::exception {
public:
	JsonException(const std::string& str, rapidjson::ParseResult res);
	virtual const char* what() const noexcept override {
		return m_error.c_str();
	}

private:
	std::string m_error;
};

struct FileVersion {
	unsigned major = 0u;
	unsigned minor = 0u;

	constexpr FileVersion() = default;

	constexpr FileVersion(unsigned major, unsigned minor) noexcept :
		major{ major },
		minor{ minor }
	{}

	explicit FileVersion(std::string_view str) {
		std::sscanf(str.data(), "%u.%u", &major, &minor);
	}

	constexpr bool operator<(const FileVersion& rhs) const noexcept {
		return (major < rhs.major) || ((major == rhs.major) && (minor < rhs.minor));
	}

	constexpr bool operator==(const FileVersion& rhs) const noexcept {
		return (major == rhs.major) && (minor == rhs.minor);
	}

	constexpr bool operator<=(const FileVersion& rhs) const noexcept {
		return this->operator<(rhs) || this->operator==(rhs);
	}

	constexpr bool operator>(const FileVersion& rhs) const noexcept {
		return !this->operator<=(rhs);
	}

	constexpr bool operator>=(const FileVersion& rhs) const noexcept {
		return !this->operator<(rhs);
	}

	constexpr bool operator!=(const FileVersion& rhs) const noexcept {
		return !this->operator==(rhs);
	}
};

class JsonLoader {
public:

	static constexpr ei::Mat2x2 TEST{ 0.f, 0.f, 0.f, 0.f };
	static constexpr FileVersion CURRENT_FILE_VERSION{ 1u, 7u };
	static constexpr FileVersion INVERTEX_TRANSMAT_FILE_VERSION{ 1u, 4u };
	static constexpr FileVersion ABSOLUTE_CAM_NEAR_FAR_FILE_VERSION{ 1u, 5u };
	static constexpr FileVersion PER_OBJECT_UNIQUE_MAT_INDICES{ 1u, 7u };

	static constexpr float DEFAULT_NEAR_PLANE_FACTOR = 1.e-4f;
	static constexpr float DEFAULT_FAR_PLANE_FACTOR = 2.f;
	static constexpr float DEFAULT_NEAR_PLANE = 0.01f;
	static constexpr float DEFAULT_FAR_PLANE = 500.f;

	JsonLoader(MufflonInstanceHdl mffInstHdl, fs::path file) :
		m_mffInstHdl{ mffInstHdl },
		m_filePath(fs::canonical(file)),
		m_binLoader{ m_mffInstHdl, m_loadingStage }
	{
		if(!fs::exists(m_filePath))
			throw std::runtime_error("JSON file '" + m_filePath.u8string() + "' doesn't exist");
		m_loadingStage.resize(1024);
	}

	// This may be called from a different thread and leads to the current load being cancelled
	void abort_load() { m_abort = true; m_binLoader.abort_load(); }
	bool was_aborted() { return m_abort; }

	const std::string& get_loading_stage() const noexcept { return m_loadingStage; }

	bool load_file(fs::path& binaryFile, FileVersion* version = nullptr);
	FileVersion get_version() const noexcept { return m_version; }
	void clear_state();

private:
	TextureHdl load_texture(const char* name, TextureSampling sampling = TextureSampling::SAMPLING_LINEAR,
							MipmapType mipmapType = MipmapType::MIPMAP_NONE, std::optional<TextureFormat> targetFormat = std::nullopt,
							TextureCallback callback = nullptr, void* userParams = nullptr);
	std::pair<TextureHdl, TextureHdl> load_displacement_map(const char* name);
	MaterialParams* load_material(rapidjson::Value::ConstMemberIterator matIter);
	void free_material(MaterialParams* mat);
	bool load_cameras(const ei::Box& aabb);
	bool load_lights();
	bool load_materials();
	bool load_scenarios(const std::vector<std::string>& binMatNames,
						const mufflon::util::FixedHashMap<mufflon::StringView, binary::InstanceMapping>& instances);
	rapidjson::Value load_scenario(const rapidjson::GenericMemberIterator<true, rapidjson::UTF8<>, rapidjson::MemoryPoolAllocator<>>& scenarioIter, int maxRecursionDepth);

	void selective_replace_keys(const rapidjson::Value& objectToCopy, rapidjson::Value& objectToCopyIn);


	MufflonInstanceHdl m_mffInstHdl;
	const fs::path m_filePath;
	std::string m_jsonString;
	rapidjson::Document m_document;
	rapidjson::Value::ConstMemberIterator m_cameras;
	rapidjson::Value::ConstMemberIterator m_lights;
	rapidjson::Value::ConstMemberIterator m_materials;
	rapidjson::Value::ConstMemberIterator m_scenarios;
	FileVersion m_version;
	mufflon::StringView m_defaultScenario;
	std::map<std::string, MaterialHdl, std::less<>> m_materialMap;
	mufflon::util::FixedHashMap<mufflon::StringView, LightHdl> m_lightMap;
	ParserState m_state;

	binary::BinaryLoader m_binLoader;

	bool m_absoluteCamNearFar = false;

	// These are for aborting a load and keeping track of progress
	std::atomic_bool m_abort = false;
	std::string m_loadingStage;
};

} // namespace mff_loader::json

namespace std {

inline std::string to_string(const mff_loader::json::FileVersion& version) {
	return std::to_string(version.major) + "." + std::to_string(version.minor);
}

} // namespace std
