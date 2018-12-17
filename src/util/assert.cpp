#include "assert.hpp"
#include <iostream>

#ifdef _MSC_VER
#include <intrin.h>
#endif // _MSC_VER

namespace mufflon {

void debug_break() {
#ifdef _MSC_VER
	__debugbreak();
#else
	__builtin_trap();
#endif // _MSC_VER
}

void check_assert(bool condition, const char* file, int line, const char* condStr) {
	if(!condition) {
		std::cerr << "Assertion '" << condStr << "' (" << file << ", line " << line << ") failed" << std::endl;
#ifndef NO_BREAK_ON_ASSERT
		debug_break();
#endif // NO_BREAK_ON_ASSERT
	}
}

void check_assert(bool condition, const char* file, int line, const char* condStr, const char* msg) {
	if(!condition) {
		std::cerr << "Assertion '" << condStr << "' (" << file << ", line " << line << ") failed: " << msg << std::endl;
#ifndef NO_BREAK_ON_ASSERT
		debug_break();
#endif // NO_BREAK_ON_ASSERT
	}
}

} // namespace mufflon