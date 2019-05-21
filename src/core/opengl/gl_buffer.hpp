#pragma once
#include "gl_wrapper.hpp"

namespace mufflon {
namespace gl {

	template<class T>
	struct BufferHandle
	{
		Handle id;
		// element offset
		size_t offset;

		BufferHandle() = default;
		BufferHandle(Handle id, size_t offset = 0) :
			id(id),
			offset(offset)
		{}
		size_t get_byte_offset() const {
			return offset * sizeof(T);
		}
		// pointer arithmetic helper
		BufferHandle& operator+=(size_t o) {
			offset += o;
			return *this;
		}
		BufferHandle& operator-=(size_t o) {
			offset -= o;
			return *this;
		}
		BufferHandle operator+(size_t o) const {
			return BufferHandle(*this) += o;
		}
		BufferHandle operator-(size_t o) const {
			return BufferHandle(*this) -= o;
		}
		bool operator==(std::nullptr_t) const {
			return id == 0;
		}
		bool operator!=(std::nullptr_t) const {
			return id != 0;
		}
		bool operator==(const BufferHandle<T>& o) const {
			return id == o.id && offset == o.offset;
		}
		bool operator!=(const BufferHandle<T>& o) const {
			return !(*this == o);
		}
		// bool conversion (removed because it conflicts with + and - operator)
		//operator bool() const
		//{
		//	return id != 0;
		//}
	};
}
}