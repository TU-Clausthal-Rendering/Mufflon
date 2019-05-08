#pragma once
#include <cstdint>

// soft wrapper for opengl that avoids the windows include

namespace mufflon {
namespace gl {
	using Handle = uint32_t;

	template<class T>
	struct BufferHandle
	{
		Handle id;
		// element offset
		size_t offset;

		size_t get_byte_offset() const
		{
			return offset * sizeof(T);
		}
		// pointer arithmetic helper
		BufferHandle& operator+=(size_t o)
		{
			offset += o;
			return *this;
		}
		BufferHandle& operator-=(size_t o)
		{
			offset -= o;
			return *this;
		}
		BufferHandle operator+(size_t o)
		{
			return BufferHandle(*this) += o;
		}
		BufferHandle operator-(size_t o)
		{
			return BufferHandle(*this) -= o;
		}
		// bool conversion
		operator bool() const
		{
			return id != 0;
		}
	};

	enum class BufferType
	{
		ShaderStorage = 0x90D2
	};

	enum class StorageFlags
	{
		None = 0,
		DynamicStorage = 0x0100,
	};

	Handle genBuffer();
	void bindBuffer(BufferType target, Handle id);
	void bufferStorage(Handle id, size_t size, const void* data, StorageFlags flags);
	void copyBufferSubData(Handle src, Handle dst, size_t srcOffset, size_t dstOffset, size_t size);
	void deleteBuffer(Handle h);
	void bufferSubData(Handle h, size_t offset, size_t size, const void* data);
	void clearBufferData(Handle h, size_t clearValueSize, size_t numValues, const void* clearValue);
	// numValues: number of values in the buffer
	// clearValue: the reference clear value
	// This function tries to use the build in clearBufferData function.
	// However, if the size of the clear value is not supported by opengl
	// (e.g. bigger than 16 byte) a more expensive sub data update will be
	// used
	template < class T >
	void clearBufferData(Handle h, size_t numValues, const T* clearValue) {
		clearBufferData(h, sizeof(T), numValues, clearValue);
	}
	void getBufferSubData(Handle h, size_t offset, size_t size, void* dstData);
}}