#pragma once

#include "util/filesystem.hpp"
#include "util/log.hpp"
#include <string>
#include <string_view>

namespace mufflon {

class Plugin {
public:
	Plugin(fs::path path);
	Plugin(const Plugin&) = delete;
	Plugin(Plugin&&);
	Plugin& operator=(const Plugin&) = delete;
	Plugin& operator=(Plugin&&) = delete;
	virtual ~Plugin();

	bool is_loaded() const;
	void close();
	bool has_function(std::string_view name) const;

	const fs::path& get_path() const noexcept {
		return m_pluginPath;
	}

protected:
	template < class R, class... Args >
	using FunctionPtr = R(*)(Args...);

	template < class R, class... Args >
	FunctionPtr<R, Args...> load_function(std::string_view name) const {
		using FunctionType = FunctionPtr<R, Args...>;
		if(is_loaded()) {
			FunctionType proc = static_cast<FunctionType>(this->load_procedure(&name[0u]));
			if(proc == nullptr)
				logError("[Plugin::resolve_function] Failed to load function address '",
						 name, "': ", get_last_error_message());
			return proc;
		}
		return static_cast<FunctionType>(nullptr);
	}

private:
	void* load_procedure(const char* name) const;
	static std::string get_last_error_message();

	fs::path m_pluginPath;
	void* m_handle;
};

} // namespace mufflon