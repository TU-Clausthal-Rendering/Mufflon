#pragma once

#include <string>

/// Log levels from 0 (all things including pendantic) to 3 (only errors and fatal errors)
#ifndef CA_LOG_LEVEL
#define CA_LOG_LEVEL 1
#endif

namespace mufflon {
	
enum class LogSeverity {
	PEDANTIC,
	INFO,
	WARNING,
	ERROR,
	FATAL_ERROR,
};
	
typedef void (*MessageHandlerFunc)(LogSeverity _severity, const std::string& _message);
	
/// Add a function which is called for each occuring message.
/// \details All message handler functions are called in order of insertion.
void registerMessageHandler(MessageHandlerFunc _func);
	
/// Remove the default output to std::cerr. If a message is send when no handler
/// is registered, the default handler is enabled again.
void disableStdHandler();
	
namespace details {
		
	// This one calls all the callbacks
	void logMessage(LogSeverity _severity, const std::string& _msg);

	// Dummy conversion methods to make all types compatible
	inline const char* to_string(const char* _str) { return _str; }
	inline const std::string& to_string(const std::string& _str) { return _str; }
	using std::to_string;

	// This one builds the message string
	template<typename T, typename... ArgTs>
	void logMessage(LogSeverity _severity, std::string& _msg, T&& _head, ArgTs&&... _tail)
	{
		using namespace std;
		_msg += to_string(_head);
		logMessage(_severity, _msg, std::forward<ArgTs>(_tail)...);
	}
}

#if CA_LOG_LEVEL <= 0
template<typename... ArgTs>
void logPedantic(ArgTs&&... _args)
{
	std::string msg;
	msg.reserve(500);
	details::logMessage(LogSeverity::PEDANTIC, msg, std::forward<ArgTs>(_args)...);
}
#else
template<typename... ArgTs>
void logPedantic(ArgTs&&... _args) {}
#endif

#if CA_LOG_LEVEL <= 1
template<typename... ArgTs>
void logInfo(ArgTs&&... _args)
{
	std::string msg;
	msg.reserve(500);
	details::logMessage(LogSeverity::INFO, msg, std::forward<ArgTs>(_args)...);
}
#else
template<typename... ArgTs>
void logInfo(ArgTs&&... _args) {}
#endif

#if CA_LOG_LEVEL <= 2
template<typename... ArgTs>
void logWarning(ArgTs&&... _args)
{
	std::string msg;
	msg.reserve(500);
	details::logMessage(LogSeverity::WARNING, msg, std::forward<ArgTs>(_args)...);
}
#else
template<typename... ArgTs>
void logWarning(ArgTs&&... _args) {}
#endif

template<typename... ArgTs>
void logError(ArgTs&&... _args)
{
	std::string msg;
	msg.reserve(500);
	details::logMessage(LogSeverity::ERROR, msg, std::forward<ArgTs>(_args)...);
}
	
template<typename... ArgTs>
void logFatal(ArgTs&&... _args)
{
	std::string msg;
	msg.reserve(500);
	details::logMessage(LogSeverity::FATAL_ERROR, msg, std::forward<ArgTs>(_args)...);
}

} // namespace mufflon