#include "output_handler.hpp"
#include "util/log.hpp"
#include "util/parallel.hpp"


using namespace mufflon::scene::textures;

namespace mufflon::renderer {

OutputHandler::OutputHandler(u16 width, u16 height, OutputValue targets) :
	// If a texture is created without data, it does not allocate data yet.
	m_cumulativeTex{
		Texture{"Output###Cum_Radiance", width, height, 1, Format::RGBA32F, SamplingMode::NEAREST, false},
		Texture{"Output###Cum_Position", width, height, 1, Format::RGBA32F, SamplingMode::NEAREST, false},
		Texture{"Output###Cum_Albedo", width, height, 1, Format::RGBA32F, SamplingMode::NEAREST, false},
		Texture{"Output###Cum_Normal", width, height, 1, Format::RGBA32F, SamplingMode::NEAREST, false},
		Texture{"Output###Cum_Lightness", width, height, 1, Format::R32F, SamplingMode::NEAREST, false}
	},
	m_iterationTex{
		Texture{"Output###Iter_Radiance", width, height, 1, Format::RGBA32F, SamplingMode::NEAREST, false},
		Texture{"Output###Iter_Position", width, height, 1, Format::RGBA32F, SamplingMode::NEAREST, false},
		Texture{"Output###Iter_Albedo", width, height, 1, Format::RGBA32F, SamplingMode::NEAREST, false},
		Texture{"Output###Iter_Normal", width, height, 1, Format::RGBA32F, SamplingMode::NEAREST, false},
		Texture{"Output###Iter_Lightness", width, height, 1, Format::R32F, SamplingMode::NEAREST, false}
	},
	m_cumulativeVarTex{
		Texture{"Output###CumVar_Radiance", width, height, 1, Format::RGBA32F, SamplingMode::NEAREST, false},
		Texture{"Output###CumVar_Position", width, height, 1, Format::RGBA32F, SamplingMode::NEAREST, false},
		Texture{"Output###CumVar_Albedo", width, height, 1, Format::RGBA32F, SamplingMode::NEAREST, false},
		Texture{"Output###CumVar_Normal", width, height, 1, Format::RGBA32F, SamplingMode::NEAREST, false},
		Texture{"Output###CumVar_Lightness", width, height, 1, Format::R32F, SamplingMode::NEAREST, false}
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
				rb.m_targets[i] = m_iterationTex[i].acquire<dev>();	// Allocates if necessary
				m_iterationTex[i].mark_changed(dev);
				m_iterationTex[i].clear<dev>();
				if(reset) {
					m_cumulativeTex[i].acquire<dev>();		// Allocates if necessary
					m_cumulativeVarTex[i].acquire<dev>();	// Allocates if necessary
					m_cumulativeTex[i].clear<dev>();
					m_cumulativeVarTex[i].clear<dev>(); // TODO: async
				}
			} else {
				rb.m_targets[i] = m_cumulativeTex[i].acquire<dev>();	// Allocates if necessary
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
				TextureDevHandle_t<Device::CUDA> iterTex = m_iterationTex[i].acquire<Device::CUDA>(); // TODO: aquire const
				TextureDevHandle_t<Device::CUDA> cumTex = m_cumulativeTex[i].acquire<Device::CUDA>(); m_cumulativeTex[i].mark_changed(Device::CUDA);
				TextureDevHandle_t<Device::CUDA> varTex = m_cumulativeVarTex[i].acquire<Device::CUDA>(); m_cumulativeVarTex[i].mark_changed(Device::CUDA);
				update_variance_cuda(iterTex, cumTex, varTex);
			} else {
				TextureDevHandle_t<Device::CPU> iterTex = m_iterationTex[i].acquire<Device::CPU>(); // TODO: aquire const
				TextureDevHandle_t<Device::CPU> cumTex = m_cumulativeTex[i].acquire<Device::CPU>(); m_cumulativeTex[i].mark_changed(Device::CPU);
				TextureDevHandle_t<Device::CPU> varTex = m_cumulativeVarTex[i].acquire<Device::CPU>(); m_cumulativeVarTex[i].mark_changed(Device::CPU);
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


void OutputHandler::set_targets(OutputValue targets) {
	if(targets != m_targets) {
		int i = 0;
		for (u32 flag : OutputValue::iterator) {
			// Is this atttribute recorded at all?
			if(!targets.is_set(flag) && m_targets.is_set(flag)) {
				m_cumulativeTex[i].unload<Device::CPU>();
				m_cumulativeTex[i].unload<Device::CUDA>();
			}
			if(!targets.is_set(flag << 8) && m_targets.is_set(flag << 8)) {
				m_cumulativeVarTex[i].unload<Device::CPU>();
				m_cumulativeVarTex[i].unload<Device::CUDA>();
				m_iterationTex[i].unload<Device::CPU>();
				m_iterationTex[i].unload<Device::CUDA>();
			}
			++i;
		}
		m_targets = targets;
	}
}

scene::textures::Format OutputHandler::get_target_format(OutputValue which) {
	switch(which) {
		case OutputValue::LIGHTNESS:
		case OutputValue::LIGHTNESS_VAR:
			return Format::R32F;
		default:
			return Format::RGBA32F;
	}
}

CpuTexture OutputHandler::get_data(OutputValue which, Format exportFormat, bool exportSRgb) {
	// Is the current flag, and in case of variance its basic value, set?
	if(!m_targets.is_set(which) || (which.is_variance() && !m_targets.is_set(which >> 8))) {
		logError("[OutputHandler::get_data] The desired quantity cannot be exported, because it is not recorded!");
		return CpuTexture{1,1,1,exportFormat,SamplingMode::NEAREST,exportSRgb};
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
		m_cumulativeVarTex[quantity].acquire_const<Device::CPU>() :
		m_cumulativeTex[quantity].acquire_const<Device::CPU>();

	const int PIXELS = m_width * m_height;
#pragma PARALLEL_FOR
	for(int i = 0; i < PIXELS; ++i) {
		const Pixel pixel{ i % m_width, i / m_width };
		ei::Vec4 value = read(tex, pixel);
		value *= normalizer;
		data.write(value, pixel);
	}

	return data;
}

ei::Vec4 OutputHandler::get_pixel_value(OutputValue which, Pixel pixel) {
	// Is the current flag, and in case of variance its basic value, set?
	if(!m_targets.is_set(which) || (which.is_variance() && !m_targets.is_set(which >> 8))) {
		logError("[OutputHandler::get_data] The desired quantity cannot be exported, because it is not recorded!");
		return ei::Vec4{};
	}

	int quantity = which.is_variance() ? ei::ilog2(which >> 8) : ei::ilog2(int(which));
	bool isNormalized = !which.is_variance() && m_targets.is_set(which << 8);
	float normalizer = isNormalized ? 1.0f : ((which & 0xff) ?
											  1.0f / ei::max(1, m_iteration + 1) :
											  1.0f / ei::max(1, m_iteration));
	// Upload to CPU / synchronize if necessary
	ConstTextureDevHandle_t<Device::CPU> tex = which.is_variance() ?
		m_cumulativeVarTex[quantity].acquire_const<Device::CPU>() :
		m_cumulativeTex[quantity].acquire_const<Device::CPU>();

	return read(tex, pixel) * normalizer;
}

} // namespace mufflon::renderer