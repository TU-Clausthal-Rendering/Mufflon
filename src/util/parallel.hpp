#pragma once

#if defined(DEBUG_ENABLED) || defined(_DEBUG) || defined(DEBUG)

#define PARALLEL_FOR
#else

#define PARALLEL_FOR omp parallel for

#endif