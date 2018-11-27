#pragma once

#include "plugin.hpp"
#include "core/export/texture_data.h"

namespace mufflon {

class TextureLoaderPlugin : public Plugin {
public:
	TextureLoaderPlugin(fs::path path);
	TextureLoaderPlugin(const TextureLoaderPlugin&) = delete;
	TextureLoaderPlugin(TextureLoaderPlugin&&) = default;
	TextureLoaderPlugin& operator=(const TextureLoaderPlugin&) = delete;
	TextureLoaderPlugin& operator=(TextureLoaderPlugin&&) = delete;
	~TextureLoaderPlugin() = default;

	bool can_load_format(std::string_view extension) const {
		if(!m_canLoadFunc)
			throw std::runtime_error("No function bound for 'can_load_format'");
		return m_canLoadFunc(&extension[0u]);
	}

	bool load(std::string_view filePath, TextureData* texData) const {
		if(!m_loadFunc)
			throw std::runtime_error("No function bound for 'load'");
		return m_loadFunc(&filePath[0u], texData);
	}
	
private:
	bool(*m_canLoadFunc)(const char*);
	bool(*m_loadFunc)(const char*, TextureData* texData);
};

} // namespace mufflon