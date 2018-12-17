#pragma once

// Defines whether an assert should automatically trigger a breakpoint in a debugger or not
//#define NO_BREAK_ON_ASSERT

namespace mufflon {

void debug_break();
void check_assert(bool condition, const char* file, int line, const char* condStr);
void check_assert(bool condition, const char* file, int line, const char* condStr, const char* msg);

} // namespace mufflon

#ifdef DEBUG_ENABLED
#ifdef __CUDACC__
#ifndef NO_BREAK_ON_ASSERT
#define cudaDebugBreak asm("brkpt;");
#else // NO_BREAK_ON_ASSERT
#define cudaDebugBreak ((void)0)
#endif // NO_BREAK_ON_ASSERT
#define mAssert(cond)																													\
	do {																																\
		if(!(cond))	{																													\
			printf("Assertion '" #cond "' (%s, line %d) failed\n", __FILE__, __LINE__);													\
			cudaDebugBreak();																											\
		}																																\
	} while(0)
#define mAssertMsg(cond, msg)																											\
	do {																																\
		if(!(cond)) {																													\
			printf("Assertion '" #cond "' (%s, line %d) failed: %s\n", __FILE__, __LINE__, msg);										\
			cudaDebugBreak();																											\
		}																																\
	} while(0)
#else // __CUDACC__
#define mAssert(cond) check_assert((cond), __FILE__, __LINE__, #cond)
#define mAssertMsg(cond, msg) check_assert((cond), __FILE__, __LINE__, #cond, msg)
#endif // __CUDACC__
#else // NDEBUG
#define mAssert(cond) ((void)0)
#define mAssertMsg(cond, msg) ((void)0)
#endif // NDEBUG
