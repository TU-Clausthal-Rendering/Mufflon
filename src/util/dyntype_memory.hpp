#pragma once

#include "util/assert.hpp"
#include <memory>

namespace mufflon {

/*
 * Basic size prediction for trivial types.
 * Specialize for the specific needs of more complex types.
 * The difference to a sizeof() is that the type may have additional (runtime)
 * parameter dependent memory. I.e. if the size of an instance is sizeof(T)+x.
 */
template<typename T, typename... Args> inline std::size_t predict_size(Args...) {
	return sizeof(T);
}

/*
 * Helper function for a systactical more convienient reinterpretation of memory.
 * The nice trick with overloading is that you don't need to bother with const/ref/pointer.
 */
template<typename TTarget, typename T> inline TTarget& as(T& t) {
	return reinterpret_cast<TTarget&>(t);
}
template<typename TTarget, typename T> inline const TTarget& as(const T& t) {
	return reinterpret_cast<const TTarget&>(t);
}
template<typename TTarget, typename T> inline TTarget* as(T* t) {
	return reinterpret_cast<TTarget*>(t);
}
template<typename TTarget, typename T> inline const TTarget* as(const T* t) {
	return reinterpret_cast<const TTarget*>(t);
}

/*
 * At different points of the renderer we store 'tagged unions' of very heterogenious
 * types. These can have different sizes which even change on runtime. To simplyfy working
 * with these dynamic changing types the following helper allocates some memory and grants
 * eseay casted access.
 * Instances of DyntypeMemory should be created as seldom as possible and then reused.
 * The creation of the memory includes a dynamic allocation, while all its methods only
 * operate inplace.
 */
template<typename BaseT>
class DyntypeMemory {
public:
	// Create the memory (no initilization)
	DyntypeMemory(std::size_t upperBoundSize) :
		// TODO: use custom allocation for GPU memory, or allow DyntypeMemory arrays?
		m_mem(std::make_unique<unsigned char[]>(upperBoundSize)),
		m_maxSize(upperBoundSize)
	{}

	/*
	 * Create an instance of whatever type in this memory
	 * T: The type to create. Must always be a type with a trivial destructor.
	 *		Otherwise memory leaks will appear!
	 *		Additionally, there must be a predict_size<T>(args) function, which
	 *		take the same arguments as the constructor and returns the size of
	 *		the created instance. This mechanism handles the runtime dependent
	 *		size variation.
	 * Args: constructor arguments for the type which is created
	 */
	template<typename T, typename... Args>
	void initialize(Args... args) {
		mAssert(predict_size<T>(args...) <= m_maxSize);
		// Placement new
		new (m_mem.get()) T {args...};
	}

	// Simple access in form of the base type
	BaseT& operator () () {
		return *reinterpret_cast<BaseT*>(m_mem.get());
	}
	const BaseT& operator () () const {
		return *reinterpret_cast<const BaseT*>(m_mem.get());
	}

	// Simple access in form of a different type
	template<typename T> T& as() {
		static_assert(std::is_base_of<BaseT, T>::value, "Base type and advanced type are likely to be incompatible.");
		return *reinterpret_cast<T*>(m_mem.get());
	}
	template<typename T> const T& as() const {
		static_assert(std::is_base_of<BaseT, T>::value, "Base type and advanced type are likely to be incompatible.");
		return *reinterpret_cast<const T*>(m_mem.get());
	}
private:
	std::unique_ptr<unsigned char[]> m_mem;		// The block of memory.
	std::size_t m_maxSize;
};

} // namespace mufflon