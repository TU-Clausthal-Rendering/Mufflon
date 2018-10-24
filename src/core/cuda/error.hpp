#pragma once

#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

namespace mufflon::cuda {

class CudaException : public std::exception {
public:
	CudaException(cudaError_t error) :
		m_errCode(error)
	{}

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

} // mufflon::cuda