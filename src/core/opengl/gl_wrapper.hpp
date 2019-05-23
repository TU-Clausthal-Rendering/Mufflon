#pragma once
#include <cstdint>
#include <cstddef>

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

	enum class BufferFormat {
		R8 = 0x8229,
		R16 = 0x822A,
		R16F = 0x822D,
		R32F = 0x822E,
		R8I = 0x8231,
		R16I = 0x8233,
		R32I = 0x8235,
		R8UI = 0x8232,
		R16UI = 0x8234,
		R32UI = 0x8236,
		RG8 = 0x822B,
		RG16 = 0x822C,
		RG16F = 0x822F,
		RG32F = 0x8230,
		RG8I = 0x8237,
		RG16I = 0x8239,
		RG32I = 0x823B,
		RG8UI = 0x8238,
		RG16UI = 0x823A,
		RG32UI = 0x823C,
		RGB32F = 0x8815,
		RGB32I = 0x8D83,
		RGB32UI = 0x8D71,
		RGBA8 = 0x8058,
		RGBA16 = 0x805B,
		RGBA16F = 0x881A,
		RGBA32F = 0x8814,
		RGBA8I = 0x8D8E,
		RGBA16I = 0x8D88,
		RGBA32I = 0x8D82,
		RGBA8UI = 0x8D7C,
		RGBA16UI = 0x8D76,
		RGBA32UI = 0x8D70
	};

	enum class Type {
		BYTE = 0x1400,
		UNSIGNED_BYTE = 0x1401,
		SHORT = 0x1402,
		UNSIGNED_SHORT = 0x1403,
		INT = 0x1404,
		UNSIGNED_INT = 0x1405,
		FLOAT = 0x1406
	};

	Handle genBuffer();
	void bindBuffer(BufferType target, Handle id);
	void bufferStorage(Handle id, size_t size, const void* data, StorageFlags flags);
	void copyBufferSubData(Handle src, Handle dst, size_t srcOffset, size_t dstOffset, size_t size);
	void clearBufferSubData(Handle dst, BufferFormat bufferFormat, size_t dstOffset, size_t size, BufferFormat dataFormat,
							Type dataType, const void* data);
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