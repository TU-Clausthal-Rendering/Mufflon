#pragma once

#include "util/log.hpp"
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

namespace mufflon { namespace cuda {

class CudaException : public std::exception {
public:
	CudaException(cudaError_t error) :
		m_errCode(error) {
		logError("[Unknown function] CUDA exception: ", cudaGetErrorName(m_errCode),
				 ": ", cudaGetErrorString(m_errCode));
	}

	virtual const char *what() const noexcept override {
		return (std::string(cudaGetErrorName(m_errCode)) + std::string(": ") + std::string(cudaGetErrorString(m_errCode))).c_str();
	}

private:
	cudaError_t m_errCode;
};

inline void check_error(cudaError_t err) {
	if(err != cudaSuccess)
		throw CudaException(err);
}

}} // mufflon::cuda