#include "lambert.hpp"
#include "util/log.hpp"
#include "core/scene/textures/cputexture.hpp"

namespace mufflon { namespace scene { namespace material {

std::size_t Lambert::get_handle_pack_size(Device device) const {
	switch(device) {
		case Device::CPU: return sizeof(LambertHandlePack<Device::CPU>);
		case Device::CUDA: return sizeof(LambertHandlePack<Device::CUDA>);
		case Device::OPENGL: return sizeof(LambertHandlePack<Device::OPENGL>);
	}
	logError("[Lambert::get_handle_pack_size] Unknown device type.");
	return 0;
}

void Lambert::get_handle_pack(Device device, HandlePack* outBuffer) const {
	HandlePack matProps{ Materials::LAMBERT, get_property_flags(), m_innerMedium, m_outerMedium };
	switch(device) {
		case Device::CPU: {
			*reinterpret_cast<LambertHandlePack<Device::CPU>*>(outBuffer) =
				LambertHandlePack{ matProps, m_albedo->aquireConst<Device::CPU>() };
		} break;
		case Device::CUDA: {
			*reinterpret_cast<LambertHandlePack<Device::CUDA>*>(outBuffer) =
				LambertHandlePack{ matProps, m_albedo->aquireConst<Device::CUDA>() };
		} break;
		case Device::OPENGL: {
			*reinterpret_cast<LambertHandlePack<Device::OPENGL>*>(outBuffer) =
				LambertHandlePack{ matProps, m_albedo->aquireConst<Device::OPENGL>() };
		} break;
	}
}

void Lambert::get_parameter_pack_cpu(const HandlePack* handles, const UvCoordinate& uvCoordinate, ParameterPack* outBuffer) const {
	auto* in = reinterpret_cast<const LambertHandlePack<Device::CPU>*>(handles);
	auto* out = reinterpret_cast<LambertParameterPack*>(outBuffer);
	*out = LambertParameterPack{
		ParameterPack{ Materials::LAMBERT, in->innerMedium, in->outerMedium },
		Spectrum{(*in->albedoTex)->sample(uvCoordinate)}
	};
}


}}} // namespace mufflon::scene::material