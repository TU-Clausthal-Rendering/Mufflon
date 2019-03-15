#include "parallel.hpp"
#include <omp.h>

namespace mufflon {

int get_thread_num() {
	return ::omp_get_num_threads();
}

int get_current_thread_idx() {
	return ::omp_get_thread_num();
}

} // namespace mufflon