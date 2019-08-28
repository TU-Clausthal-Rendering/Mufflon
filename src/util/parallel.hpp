#pragma once

#include "util/assert.hpp"

#ifndef DEBUG_ENABLED
// The chunk size and scheduling type is an "best guess";
// the default values are abysmal for our case
#define PARALLEL_FOR omp parallel for schedule(static, 1000)
#define PARALLEL_REDUCTION(op, var) omp parallel for schedule(static, 1000) reduction (op:var)
#else // DEBUG_ENABLED
#define PARALLEL_FOR
#endif // DEBUG_ENABLED

namespace mufflon {

int get_thread_num();
int get_current_thread_idx();

} // namespace mufflon
