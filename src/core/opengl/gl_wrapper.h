#pragma once
#include <cstdint>

// soft wrapper for opengl that avoids the windows include

namespace mufflon {
namespace gl {
	using Handle = uint32_t;

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
	void clearBufferData(Handle h, size_t clearValueSize, const void* clearValue);
	template < class T >
	void clearBufferData(Handle h, const T* data) {
		static_assert(
			sizeof(T) == 1 ||
			sizeof(T) == 2 ||
			sizeof(T) == 3 ||
			sizeof(T) == 4 ||
			sizeof(T) == 6 ||
			sizeof(T) == 8 ||
			sizeof(T) == 12 ||
			sizeof(T) == 16,
			"gl::clearBufferData is not supported for this data type");

		clearBufferData(h, sizeof(T), data);
	}
	void getBufferSubData(Handle h, size_t offset, size_t size, void* dstData);
}}