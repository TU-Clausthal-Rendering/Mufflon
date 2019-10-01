#pragma once
#include <cstdint>
#include <cstddef>
// soft wrapper for opengl that avoids the windows include

namespace mufflon {
namespace gl {
	using Handle = uint32_t;
	using TextureHandle = uint64_t;

	enum class BufferType {
		ShaderStorage = 0x90D2
	};

    enum class TextureType {
        Texture2DArray = 0x8C1A,
    };

	enum class StorageFlags {
		None = 0,
		DynamicStorage = 0x0100,
		ClientStorage = 0x0200
	};

    enum class TextureInternal {
        R8U = 0x8229,
        RG8U = 0x822B,
        RGBA8U = 0x8058,
        R16U = 0x822A,
        RG16U = 0x822C,
        RGBA16U = 0x805B,
        R16F = 0x822D,
        RG16F = 0x822F,
        RGBA16F = 0x881A,
        R32F = 0x822E,
        RG32F = 0x8230,
        RGBA32F = 0x8814,

        SRGBA8U = 0x8C43,
    };

    enum class TextureSetFormat {
        R = 0x1903,
        RG = 0x8227,
        RGBA = 0x1908
    };

    enum class TextureSetType {
        U8 = 0x1401, // unsigned byte
        U16 = 0x1403, // unsigned short
        F16 = 0x140B, // half float
        F32 = 0x1406, // float
    };

	Handle genBuffer();
	void bindBuffer(BufferType target, Handle id);
	void bufferStorage(Handle id, size_t size, const void* data, StorageFlags flags);
	void copyBufferSubData(Handle src, Handle dst, size_t srcOffset, size_t dstOffset, size_t size);
	void clearBufferSubData(Handle dst, size_t offset, size_t size, int value);
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

	Handle genTexture();
	void bindTexture(TextureType type, Handle id);
	void deleteTexture(Handle h);
	void clearTexImage(Handle h, int level);
	void texStorage3D(Handle h, int levels, TextureInternal format, size_t width, size_t height, size_t depth);
	void texSubImage3D(Handle h, int level, size_t offsetX, size_t offsetY, size_t offsetZ, size_t width, size_t height, size_t depth, TextureSetFormat setFormat, TextureSetType setType, const void* data);
	TextureHandle getTextureHandle(Handle h);
	TextureHandle getTextureSamplerHandle(Handle tex, Handle sampler);
	void makeTextureHandleResident(TextureHandle h);
	void makeTextureHandleNonResident(TextureHandle h);

    enum class SamplerParameterI {
        WrapR = 0x8072,
        WrapS = 0x2802,
        WrapT = 0x2803,
        MinFilter = 0x2801,
        MagFilter = 0x2800,
    };

	constexpr int WRAP_REPEAT = 0x2901;
	constexpr int FILTER_NEAREST = 0x2600;
	constexpr int FILTER_LINEAR = 0x2601;
	constexpr int FILTER_NEAREST_MIPMAP_NEAREST = 0x2700;
	constexpr int FILTER_NEAREST_MIPMAP_LINEAR = 0x2702;
	constexpr int FILTER_LINEAR_MIPMAP_NEAREST = 0x2701;
	constexpr int FILTER_LINEAR_MIPMAP_LINEAR = 0x2703;

	Handle genSampler();
	void samplerParameter(Handle h, SamplerParameterI param, int value);
}}