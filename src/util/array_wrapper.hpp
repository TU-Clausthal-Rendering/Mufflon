#pragma once

#include "assert.hpp"

namespace mufflon::util {

/**
 * Wrapper class for a C-style array.
 * Uses asserts to ensure bounds during runtime if debug-mode is enabled.
 */
template < class T >
class ArrayWrapper {
public:
	ArrayWrapper(T* ptr) :
		m_ptr(ptr)
	{}

	ArrayWrapper(T* ptr, std::size_t length) :
#ifdef DEBUG_ENABLED
		m_length(length),
#endif // DEBUG_ENABLED
		m_ptr(ptr)
	{}

	T& operator*() {
		mAssert(m_ptr != nullptr);
		return *m_ptr;
	}

	const T& operator*() const {
		mAssert(m_ptr != nullptr);
		return *m_ptr;
	}

	T* operator->() {
		mAssert(m_ptr != nullptr);
		return m_ptr;
	}

	const T* operator->() const {
		mAssert(m_ptr != nullptr);
		return m_ptr;
	}

	T& operator[](std::size_t idx) {
		mAssert(m_ptr != nullptr);
		mAssert(idx < m_length);
		return m_ptr[idx];
	}

	const T& operator[](std::size_t idx) const {
		mAssert(m_ptr != nullptr);
		mAssert(idx < m_length);
		return m_ptr[idx];
	}

	explicit operator T*() {
		return m_ptr;
	}

	explicit operator const T*() const {
		return m_ptr;
	}

private:
#ifdef DEBUG_ENABLED
	std::size_t m_length;
#endif // DEBUG_ENABLED
	T* m_ptr;
};

template < class T >
class ConstArrayWrapper {
public:
	ConstArrayWrapper(const T* ptr) :
		m_ptr(ptr)
	{}

	ConstArrayWrapper(const T* ptr, std::size_t length) :
#ifdef DEBUG_ENABLED
		m_length(length),
#endif // DEBUG_ENABLED
		m_ptr(ptr)
	{}

	const T& operator*() const {
		mAssert(m_ptr != nullptr);
		return *m_ptr;
	}

	const T* operator->() const {
		mAssert(m_ptr != nullptr);
		return m_ptr;
	}

	const T& operator[](std::size_t idx) const {
		mAssert(m_ptr != nullptr);
		mAssert(idx < m_length);
		return m_ptr[idx];
	}

	explicit operator const T*() const {
		return m_ptr;
	}

private:
#ifdef DEBUG_ENABLED
	std::size_t m_length;
#endif // DEBUG_ENABLED
	const T* m_ptr;
};

} // namespace mufflon::util