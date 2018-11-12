#pragma once

#include <cassert>
#include <intrin.h>
#include <iostream>

//#define NO_BREAK_ON_ASSERT

#ifdef _MSC_VER
#ifdef _DEBUG
#define DEBUG_ENABLED
#endif // _DEBUG
#else // _MSC_VER
#ifndef NDEBUG
#define DEBUG_ENABLED
#endif // NDEBUG
#endif // _MSC_VER

#ifdef DEBUG_ENABLED
#ifndef NO_BREAK_ON_ASSERT
#define mDebugBreak __debugbreak
#else // NO_BREAK_ON_ASSERT
#define mDebugBreak ((void)0)
#endif // NO_BREAK_ON_ASSERT
#ifdef __CUDACC__
#define mAssert(cond)																													\
	do {																																\
		if(!(cond))																														\
			printf("Assertion '" #cond "' (%s, line %d) failed\n", __FILE__, __LINE__);													\
	} while(0)
#define mAssertMsg(cond, msg)																											\
	do {																																\
		if(!(cond)) {																													\
			printf("Assertion '" #cond "' (%s, line %d) failed: %s\n", __FILE__, __LINE__, msg);										\
			mDebugBreak();																												\
		}																																\
	} while(0)
#else // __CUDACC__
#define mAssert(cond)																													\
	do {																																\
		if(!(cond)) {																													\
			std::cerr << "Assertion '" << #cond << "' (" << __FILE__ << ", line " << __LINE__ << ") failed" << std::endl;				\
			mDebugBreak();																												\
		}																																\
	} while(0)
#define mAssertMsg(cond, msg)																											\
	do {																																\
		if(!(cond)) {																													\
			std::cerr << "Assertion '" << #cond << "' (" << __FILE__ << ", line " << __LINE__ << ") failed: " << msg << std::endl;		\
			mDebugBreak();																												\
		}																																\
	} while(0)
#endif // __CUDACC__
#else // NDEBUG
#define mAssert(cond) ((void)0)
#define mAssertMsg(cond, msg) ((void)0)
#endif // NDEBUG