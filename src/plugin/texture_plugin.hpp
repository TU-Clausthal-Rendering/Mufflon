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
	bool can_store_format(StringView extension) const;
	bool load(StringView filePath, TextureData* texData) const;
	bool store(StringView filePath, const TextureData* texData) const;
	
private:
	bool(*m_canLoadFunc)(const char*) = nullptr;
	bool(*m_canStoreFunc)(const char*) = nullptr;
	bool(*m_loadFunc)(const char*, TextureData* texData) = nullptr;
	bool(*m_storeFunc)(const char*, const TextureData* texData) = nullptr;
};

} // namespace mufflon