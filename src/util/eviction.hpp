#pragma once

#include "assert.hpp"
#include <cstdlib>

namespace mufflon::util {

template < class T >
class Evictable {
public:
	Evictable() :
		m_value{ static_cast<T*>(std::malloc(sizeof(T))) },
		m_present{ false }
	{}

	Evictable(const Evictable&) = delete;
	Evictable(Evictable&& other) :
		m_present{ other.m_present },
		m_value{ other.m_value }
	{
		other.m_present = false;
		other.m_value = nullptr;
	}
	Evictable& operator=(const Evictable&) = delete;
	Evictable& operator=(Evictable&& other) {
		std::swap(m_present, other.m_present);
		std::swap(m_value, other.m_value);
		return *this;
	}

	~Evictable() {
		if(m_present)
			m_value->~T();
		if(m_value != nullptr)
			std::free(m_value);
	}

	T& operator*() noexcept {
		mAssert(m_present);
		return *m_value;
	}
	const T& operator*() const noexcept {
		mAssert(m_present);
		return *m_value;
	}
	T* operator->() noexcept {
		mAssert(m_present);
		return m_value;
	}
	const T* operator->() const noexcept {
		mAssert(m_present);
		return m_value;
	}

	template < class... Args >
	T& admit(Args&& ...args) {
		this->evict();
		m_present = true;
		return *new (m_value) T(std::forward<Args>(args)...);
	}

	void evict() noexcept {
		if(m_present)
			m_value->~T();
		m_present = false;
	}
	bool is_admitted() const noexcept { return m_present; }

private:
	T* m_value;
	bool m_present;
};

} // namespace mufflon::util