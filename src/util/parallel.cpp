#include "parallel.hpp"
#include <omp.h>

namespace mufflon {

int get_thread_num() {
#ifndef DEBUG_ENABLED
	// omp_get_thread_num() always returns 1 in sequential code.
	// https://stackoverflow.com/a/13328691/1913512
	int n = 0;
	#pragma omp parallel reduction(+:n)
	n += 1;
	return n;
#else
	return 1;
#endif
}

int get_max_thread_num() {
#ifndef DEBUG_ENABLED
	return omp_get_max_threads();
#else
	return 1;
#endif
}

int get_current_thread_idx() {
#ifndef DEBUG_ENABLED
	return omp_get_thread_num();
#else
	return 0;
#endif
}

} // namespace mufflon