#pragma once

#include <string>

namespace mufflon { namespace scene { namespace textures {

/*
 * The texture class handles the resource. Sampling and accessing the data
 * is up to device specific needs.
 */
class Texture {
public:
	// Loads a texture into the CPU-RAM
	void Texture(const std::string_view& fileName);

	void make_resident(Device);
};

}}} // namespace mufflon::scene::textures