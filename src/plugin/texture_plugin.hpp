#pragma once

#include "plugin.hpp"

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

	void load(std::string_view filePath) const {
		if(!m_loadFunc)
			throw std::runtime_error("No function bound for 'load'");
		m_loadFunc(&filePath[0u]);
	}
	
private:
	bool(*m_canLoadFunc)(const char*);
	void(*m_loadFunc)(const char*);
};

} // namespace mufflon