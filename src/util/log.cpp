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

	namespace details {

		static std::vector<MessageHandlerFunc> s_msgHandlers;

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

		bool s_initialized = false; // assert s_notInitialized == false because static memory is 0
		void logMessage(LogSeverity _severity, const std::string& _msg)
		{
			if(!s_initialized)
			{
				if(s_msgHandlers.empty())
					s_msgHandlers.push_back(defaultHandler);
#if defined(_WINDOWS) || defined(_WIN64) || defined(_WIN32)
				HANDLE hStdout = GetStdHandle(STD_OUTPUT_HANDLE); 
				SetConsoleMode(hStdout, ENABLE_VIRTUAL_TERMINAL_PROCESSING | ENABLE_PROCESSED_OUTPUT);
			//	SetConsoleMode(hStdout, ENABLE_VIRTUAL_TERMINAL_INPUT);
#endif
				s_initialized = true;
			}
			if(s_msgHandlers.empty())
				defaultHandler(_severity, _msg);
			for(auto it : s_msgHandlers)
			{
				it(_severity, _msg);
			}
		}

	} // namespace details

	void registerMessageHandler(MessageHandlerFunc _func)
	{
		details::s_msgHandlers.push_back(_func);
	}

	void disableStdHandler()
	{
		for(size_t i = 0; i < details::s_msgHandlers.size(); ++i)
			if(details::s_msgHandlers[i] == details::defaultHandler)
				details::s_msgHandlers.erase(details::s_msgHandlers.begin() + i);
	}

} // namespace mufflon