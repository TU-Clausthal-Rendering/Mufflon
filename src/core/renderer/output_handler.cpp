#include "output_handler.hpp"
#include "util/log.hpp"


using namespace mufflon::scene::textures;

namespace mufflon::renderer {

OutputHandler::OutputHandler(u16 width, u16 height, OutputValue targets) :
	// If a texture is created without data, it does not allocate data yet.
	m_cumulativeTex{
		Texture{width, height, 1, Format::RGBA32F, SamplingMode::NEAREST, false},
		Texture{width, height, 1, Format::RGBA32F, SamplingMode::NEAREST, false},
		Texture{width, height, 1, Format::RGBA32F, SamplingMode::NEAREST, false},
		Texture{width, height, 1, Format::RGBA32F, SamplingMode::NEAREST, false},
		Texture{width, height, 1, Format::R32F, SamplingMode::NEAREST, false}
	},
	m_iterationTex{
		Texture{width, height, 1, Format::RGBA32F, SamplingMode::NEAREST, false},
		Texture{width, height, 1, Format::RGBA32F, SamplingMode::NEAREST, false},
		Texture{width, height, 1, Format::RGBA32F, SamplingMode::NEAREST, false},
		Texture{width, height, 1, Format::RGBA32F, SamplingMode::NEAREST, false},
		Texture{width, height, 1, Format::R32F, SamplingMode::NEAREST, false}
	},
	m_cumulativeVarTex{
		Texture{width, height, 1, Format::RGBA32F, SamplingMode::NEAREST, false},
		Texture{width, height, 1, Format::RGBA32F, SamplingMode::NEAREST, false},
		Texture{width, height, 1, Format::RGBA32F, SamplingMode::NEAREST, false},
		Texture{width, height, 1, Format::RGBA32F, SamplingMode::NEAREST, false},
		Texture{width, height, 1, Format::R32F, SamplingMode::NEAREST, false}
	},
	m_targets(targets),
	m_iteration(-1), // if begin_iteration is called without reset=true, this will still work
	m_width(width),
	m_height(height)
{
	if(width <= 0 || height <= 0) {
		logError("[OutputHandler::OutputHandler] Invalid resolution (<= 0)");
	}
}

template < Device dev >
RenderBuffer<dev> OutputHandler::begin_iteration(bool reset) {
	// Count the iteration
	if(reset)
		m_iteration = 0;
	else ++m_iteration;

	RenderBuffer<dev> rb;
	int i = 0;
	for(u32 flag : OutputValue::iterator) {
		// Is this atttribute recorded at all?
		if(m_targets.is_set(flag)) {
			if(m_targets.is_set(flag << 8)) { // Variance flag set?
				// Variance case: needs to cumulate samples per iteration
				rb.m_radiance = m_iterationTex[i].aquire<dev>();	// Allocates if necessary
				m_iterationTex[i].mark_changed(dev);
				m_iterationTex[i].clear<dev>();
				if(reset) {
					m_cumulativeTex[i].aquire<dev>();		// Allocates if necessary
					m_cumulativeVarTex[i].aquire<dev>();	// Allocates if necessary
					m_cumulativeTex[i].clear<dev>();
					m_cumulativeVarTex[i].clear<dev>(); // TODO: async
				}
			} else {
				rb.m_radiance = m_cumulativeTex[i].aquire<dev>();	// Allocates if necessary
				m_cumulativeTex[i].mark_changed(dev);
				if(reset) m_cumulativeTex[i].clear<dev>();
			}
		}
		++i;
	}
	rb.m_resolution = ei::IVec2{m_width, m_height};

	return std::move(rb);
}

template RenderBuffer<Device::CPU> OutputHandler::begin_iteration<Device::CPU>(bool reset);
template RenderBuffer<Device::CUDA> OutputHandler::begin_iteration<Device::CUDA>(bool reset);



template < Device dev >
void OutputHandler::end_iteration() {
	int i = 0;
	for(u32 flag : OutputValue::iterator) {
		// If variances are computed we need some post processing (the renderer
		// provides the iteration sample only).
		if(m_targets.is_set(flag) && m_targets.is_set(flag << 8)) {
			if constexpr(dev == Device::CUDA) {
				// Looks like inavitable redundancy. update_variance<<<>>> call is only valid
				// for one type. Without 'if constexpr' this branch would not compile. if 
				// handles are initialized TextureDevHandle_t<dev> iterTex...
				TextureDevHandle_t<Device::CUDA> iterTex = m_iterationTex[i].aquire<Device::CUDA>(); // TODO: aquire const
				TextureDevHandle_t<Device::CUDA> cumTex = m_cumulativeTex[i].aquire<Device::CUDA>(); m_cumulativeTex[i].mark_changed(Device::CUDA);
				TextureDevHandle_t<Device::CUDA> varTex = m_cumulativeVarTex[i].aquire<Device::CUDA>(); m_cumulativeVarTex[i].mark_changed(Device::CUDA);
				update_variance_cuda(iterTex, cumTex, varTex);
			} else {
				TextureDevHandle_t<Device::CPU> iterTex = m_iterationTex[i].aquire<Device::CPU>(); // TODO: aquire const
				TextureDevHandle_t<Device::CPU> cumTex = m_cumulativeTex[i].aquire<Device::CPU>(); m_cumulativeTex[i].mark_changed(Device::CPU);
				TextureDevHandle_t<Device::CPU> varTex = m_cumulativeVarTex[i].aquire<Device::CPU>(); m_cumulativeVarTex[i].mark_changed(Device::CPU);
				// TODO: openmp
				for(int y = 0; y < m_height; ++y) for(int x = 0; x < m_width; ++x)
					update_variance(iterTex, cumTex, varTex, x, y, float(m_iteration));
			}
			// TODO: opengl
		}
		++i;
	}
}

template void OutputHandler::end_iteration<Device::CPU>();
template void OutputHandler::end_iteration<Device::CUDA>();



CpuTexture OutputHandler::get_data(OutputValue which, Format exportFormat, bool exportSRgb) {
	// Is the current flag, and in case of variance its basic value, set?
	if(!m_targets.is_set(which) || (which.is_variance() && !m_targets.is_set(which >> 8))) {
		logError("[OutputHandler::get_data] The desired quantity cannot be exported, because it is not recorded!");
		return std::move(CpuTexture{1,1,1,exportFormat,SamplingMode::NEAREST,exportSRgb});
	}

	// Allocate the memory for the output and get basic properties of the quantity to export.
	// TODO: if format has the same pixel size it is possible to avoid one memcopy by directly
	// syncing into the new texture (and then convert if necessary).
	//Texture data(m_width, m_height, 1, exportFormat, SamplingMode::NEAREST, exportSRgb);
	CpuTexture data(m_width, m_height, 1, exportFormat, SamplingMode::NEAREST, exportSRgb);
	int quantity = which.is_variance() ? ei::ilog2(which >> 8) : ei::ilog2(int(which));
	bool isNormalized = !which.is_variance() && m_targets.is_set(which << 8);
	float normalizer = isNormalized ? 1.0f : ((which & 0xff) ?
								1.0f / ei::max(1, m_iteration+1) :
								1.0f / ei::max(1, m_iteration));

	// Upload to CPU / synchronize if necessary
	ConstTextureDevHandle_t<Device::CPU> tex = which.is_variance() ?
		m_cumulativeVarTex[quantity].aquireConst<Device::CPU>() :
		m_cumulativeTex[quantity].aquireConst<Device::CPU>();

	// TODO: openmp
	for(int y = 0; y < m_height; ++y) for(int x = 0; x < m_width; ++x) {
		ei::Vec4 value = read(tex, Pixel{x,y});
		value *= normalizer;
		data.write(value, Pixel{x,y});
	}

	return std::move(data);
}

} // namespace mufflon::renderer