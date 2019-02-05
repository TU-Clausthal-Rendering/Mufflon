#include "texture_plugin.hpp"

namespace mufflon {

TextureLoaderPlugin::TextureLoaderPlugin(fs::path path) :
	Plugin(std::move(path)),
	m_canLoadFunc(nullptr),
	m_loadFunc(nullptr)
{
	if(is_loaded()) {
		// Check if it supports the desired operations
		if(has_function("can_load_texture_format")
		   && has_function("load_texture")
		   && has_function("set_logger")) {
			m_canLoadFunc = load_function<bool, const char*>("can_load_texture_format");
			m_loadFunc = load_function<bool, const char*, TextureData*>("load_texture");
			m_setLogger = load_function<void, void(*)(const char*, int)>("set_logger");
		}

		// If not (or something went wrong), we immediately close it again
		if(m_canLoadFunc == nullptr || m_loadFunc == nullptr || m_setLogger == nullptr)
			close();
	}
}

bool TextureLoaderPlugin::can_load_format(StringView extension) const {
	if(!m_canLoadFunc)
		throw std::runtime_error("No function bound for 'can_load_format'");
	return m_canLoadFunc(&extension[0u]);
}

bool TextureLoaderPlugin::load(StringView filePath, TextureData* texData) const {
	if(!m_loadFunc)
		throw std::runtime_error("No function bound for 'load'");
	return m_loadFunc(&filePath[0u], texData);
}

void TextureLoaderPlugin::set_logger(void(*logCallback)(const char*, int)) {
	if(!m_setLogger)
		throw std::runtime_error("No function bound for 'set_logger'");
	return m_setLogger(logCallback);
}

} // namespace mufflon