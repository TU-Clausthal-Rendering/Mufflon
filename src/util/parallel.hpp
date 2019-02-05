#pragma once

#include "util/assert.hpp"

#ifndef DEBUG_ENABLED
#define PARALLEL_FOR omp parallel for
#else // DEBUG_ENABLED
#define PARALLEL_FOR
#endif // DEBUG_ENABLED