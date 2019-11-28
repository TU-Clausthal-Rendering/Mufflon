#pragma once

#include "util/assert.hpp"

#ifndef DEBUG_ENABLED
// The chunk size and scheduling type is an "best guess";
// the default values are abysmal for our case
#define PARALLEL_FOR omp parallel for schedule(static, 1000)
#define PARALLEL_FOR_COND(cond) omp parallel for schedule(static) if(cond)
#define PARALLEL_FOR_COND_DYNAMIC(cond) omp parallel for schedule(dynamic) if(cond)
#else // DEBUG_ENABLED
#define PARALLEL_FOR
#define PARALLEL_FOR_COND(cond)
#define PARALLEL_FOR_COND_DYNAMIC(cond)
#endif // DEBUG_ENABLED

namespace mufflon {

int get_thread_num();
int get_current_thread_idx();
int get_max_thread_num();

} // namespace mufflon
