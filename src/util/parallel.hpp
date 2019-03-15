#pragma once

#include "util/assert.hpp"

#ifndef DEBUG_ENABLED
#define PARALLEL_FOR omp parallel for
#else // DEBUG_ENABLED
#define PARALLEL_FOR
#endif // DEBUG_ENABLED

namespace mufflon {

int get_thread_num();
int get_current_thread_idx();

} // namespace mufflon
