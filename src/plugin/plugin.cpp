#include "plugin.hpp"
#include "util/log.hpp"
#ifdef _MSC_VER
#include <windows.h>
#else // _MSC_VER
#include <dlfcn.h>
#endif // _MSC_VER

namespace mufflon {

#ifdef _MSC_VER
using HandleType = HINSTANCE;
#else // _MSC_VER
using HandleType = void*;
#endif // _MSC_VER

Plugin::Plugin(fs::path path) :
	m_pluginPath(path),
	m_handle(nullptr)
{
	if(fs::exists(m_pluginPath) && !fs::is_directory(m_pluginPath)) {
#ifdef _MSC_VER
		m_handle = ::LoadLibrary(m_pluginPath.c_str());
#else // _MSC_VER
		m_handle = dlopen();
#endif // _MSC_VER
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
#ifdef _MSC_VER
		if(!::FreeLibrary(static_cast<HandleType>(m_handle))) {
#else // _MSC_VER
		if(::dlcose(static_cast<HandleType>(m_handle)) != 0) {
#endif // _MSC_VER
			logError("[Plugin::close] Failed to free plugin '",
					 m_pluginPath.string(), "': ", get_last_error_message());
		}
		m_handle = nullptr;
	}
}

bool Plugin::has_function(std::string_view name) const {
	if(is_loaded())
		return load_procedure(&name[0u]) != nullptr;
	return false;
}

void* Plugin::load_procedure(const char* name) const {
#ifdef _MSC_VER
	return ::GetProcAddress(static_cast<HandleType>(m_handle), name);
#else // _MSC_VER
	return ::dlsym(static_cast<HandleType>(m_handle), name);
#endif // _MSC_VER
}

std::string Plugin::get_last_error_message() {
#ifdef _MSC_VER
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
#else // _MSC_VER
	return std::string(::dlerror());
#endif // _MSC_VER
}

} // namespace mufflon