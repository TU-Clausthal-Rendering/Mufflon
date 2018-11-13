#include "texture.hpp"
#include "cputexture.hpp"

namespace mufflon { namespace scene { namespace textures {

Texture::Texture(std::string_view fileName) :
	m_srcFileName(fileName)
{
	// A file loader provides an array with pixel data. This is loaded into
	// a CPUTexture per default.
	// TODO create texture resource

	// Switch format and use an image loader library to fill the texture resource.
	// TODO load


	// REMOVE (this dump is only for compile checks)
	using t = decltype(m_handles.get<TextureDevHandle_t<Device::CPU>>());
	auto x = *aquireConst<Device::CPU>();
	auto y = *aquireConst<Device::CUDA>();
	auto z = *aquireConst<Device::OPENGL>();
	auto x2 = *aquire<Device::CPU>();
	auto y2 = *aquire<Device::CUDA>();
	auto z2 = *aquire<Device::OPENGL>();
}

}}} // namespace mufflon::scene::textures