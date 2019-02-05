#include "plugin.hpp"
#include "util/log.hpp"
#ifdef _WIN32
#include <windows.h>
#else // _WIN32
#include <dlfcn.h>
#endif // _WIN32

namespace mufflon {

#ifdef _WIN32
using HandleType = HINSTANCE;
#else // _WIN32
using HandleType = void*;
#endif // _WIN32

Plugin::Plugin(fs::path path) :
	m_pluginPath(path),
	m_handle(nullptr)
{
	if(fs::exists(m_pluginPath) && !fs::is_directory(m_pluginPath)) {
#ifdef _WIN32
		m_handle = ::LoadLibrary(m_pluginPath.c_str());
#else // _WIN32
		m_handle = dlopen();
#endif // _WIN32
		if(!is_loaded())
			logError("[Plugin::Plugin] Failed to load plugin '",
					 m_pluginPath.string(), "': ", get_last_error_message());
	}
}

Plugin::Plugin(Plugin&& plugin) :
	m_pluginPath(std::move(plugin.m_pluginPath)),
	m_handle(plugin.m_handle)
{
	plugin.m_handle = nullptr;
}

Plugin::~Plugin() {
	this->close();
}


bool Plugin::is_loaded() const {
	return m_handle != nullptr;
}

void Plugin::close() {
	if(is_loaded()) {
#ifdef _WIN32
		if(!::FreeLibrary(static_cast<HandleType>(m_handle))) {
#else // _WIN32
		if(::dlcose(static_cast<HandleType>(m_handle)) != 0) {
#endif // _WIN32
			logError("[Plugin::close] Failed to free plugin '",
					 m_pluginPath.string(), "': ", get_last_error_message());
		}
		m_handle = nullptr;
	}
}

bool Plugin::has_function(StringView name) const {
	if(is_loaded())
		return load_procedure(&name[0u]) != nullptr;
	return false;
}

void* Plugin::load_procedure(const char* name) const {
#ifdef _WIN32
	return ::GetProcAddress(static_cast<HandleType>(m_handle), name);
#else // _WIN32
	return ::dlsym(static_cast<HandleType>(m_handle), name);
#endif // _WIN32
}

std::string Plugin::get_last_error_message() {
#ifdef _WIN32
	DWORD errorId = ::GetLastError();
	if(errorId == 0) {
		return std::string();
	} else {
		LPSTR msgBuffer = nullptr;
		std::size_t msgSize = ::FormatMessageA(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM
												| FORMAT_MESSAGE_IGNORE_INSERTS,
												nullptr, errorId, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
												(LPSTR)&msgBuffer, 0, nullptr);
		std::string msg(msgBuffer, msgSize);
		::LocalFree(msgBuffer);
		return msg;
	}
#else // _WIN32
	return std::string(::dlerror());
#endif // _WIN32
}

} // namespace mufflon