#include "log.hpp"

#include <vector>
#include <iostream>
#if defined(_WINDOWS) || defined(_WIN64) || defined(_WIN32)
#include <windows.h>
#ifndef ENABLE_VIRTUAL_TERMINAL_PROCESSING
#define ENABLE_VIRTUAL_TERMINAL_PROCESSING 0x0004
#endif
#undef ERROR
/*inline std::ostream& win_green(std::ostream &s)
{
	HANDLE hStdout = GetStdHandle(STD_OUTPUT_HANDLE);
	SetConsoleTextAttribute(hStdout, 
		FOREGROUND_GREEN);
	return s;
}
inline std::ostream& win_white(std::ostream &s)
{
	HANDLE hStdout = GetStdHandle(STD_OUTPUT_HANDLE); 
	SetConsoleTextAttribute(hStdout, 
		FOREGROUND_RED|FOREGROUND_GREEN|FOREGROUND_BLUE);
	return s;
}
inline std::ostream& win_red(std::ostream &s)
{
	HANDLE hStdout = GetStdHandle(STD_OUTPUT_HANDLE); 
	SetConsoleTextAttribute(hStdout, 
		FOREGROUND_RED|FOREGROUND_INTENSITY);
	return s;
}*/
#endif

namespace mufflon {

	// Defaults the log level to 'Info', but can be changed at runtime
	LogSeverity s_logLevel = LogSeverity::INFO;

	namespace details {

		void defaultHandler(LogSeverity _severity, const std::string& _message)
		{
			switch(_severity)
			{
			case LogSeverity::PEDANTIC:
				std::cerr << "\033[38;2;128;128;128mINF: " << _message << "\033[0m\n";
				break;
			case LogSeverity::INFO:
				std::cerr << "\033[32mINF: " << _message << "\033[0m\n";
				break;
			case LogSeverity::WARNING:
				std::cerr << "\033[38;2;200;200;0mWAR: " << _message << "\033[0m\n";
				break;
			case LogSeverity::ERROR:
				std::cerr << "\033[38;2;255;40;25mERR: " << _message << "\033[0m\n";
				break;
			case LogSeverity::FATAL_ERROR:
				std::cerr << "\033[38;2;255;0;255mFATAL: " << _message << "\033[0m\n";
				throw std::bad_exception();
				break;
			}
		}

		MessageHandlerFunc s_msgHandler = &defaultHandler;

		bool s_initialized = false; // assert s_notInitialized == false because static memory is 0
		void logMessage(LogSeverity _severity, const std::string& _msg)
		{
			if(!s_initialized)
			{
#if defined(_WINDOWS) || defined(_WIN64) || defined(_WIN32)
				HANDLE hStdout = GetStdHandle(STD_OUTPUT_HANDLE); 
				SetConsoleMode(hStdout, ENABLE_VIRTUAL_TERMINAL_PROCESSING | ENABLE_PROCESSED_OUTPUT);
			//	SetConsoleMode(hStdout, ENABLE_VIRTUAL_TERMINAL_INPUT);
#endif
				s_initialized = true;
			}

			s_msgHandler(_severity, _msg);
		}

	} // namespace details

	void setMessageHandler(MessageHandlerFunc _func)
	{
		details::s_msgHandler = _func;
	}

} // namespace mufflon