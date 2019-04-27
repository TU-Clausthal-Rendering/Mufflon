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
}}