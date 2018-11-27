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
		   && has_function("load_texture")) {
			m_canLoadFunc = load_function<bool, const char*>("can_load_texture_format");
			m_loadFunc = load_function<void, const char*>("load_texture");
		}

		// If not (or something went wrong), we immediately close it again
		if(m_canLoadFunc == nullptr || m_loadFunc == nullptr)
			close();
	}
}

} // namespace mufflon