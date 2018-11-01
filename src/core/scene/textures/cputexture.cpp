#include "cputexture.hpp"

namespace mufflon { namespace scene { namespace textures {

ei::Vec4 CpuTexture::sample(const UvCoordinate& uv) const {
	return ei::Vec4{1.0f};
}

}}} // namespace mufflon::scene::texturesnst UvCoordinate& uv);