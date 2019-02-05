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

	bool can_load_format(StringView extension) const;
	bool load(StringView filePath, TextureData* texData) const;
	void set_logger(void(*logCallback)(const char*, int));
	
private:
	bool(*m_canLoadFunc)(const char*) = nullptr;
	bool(*m_loadFunc)(const char*, TextureData* texData) = nullptr;
	void(*m_setLogger)(void(*logCallback)(const char*, int)) = nullptr;
};

} // namespace mufflon