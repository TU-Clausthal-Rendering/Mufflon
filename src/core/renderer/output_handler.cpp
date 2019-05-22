#include "output_handler.hpp"
#include "util/log.hpp"
#include "util/parallel.hpp"


using namespace mufflon::scene::textures;

namespace mufflon::renderer {

constexpr size_t ATOMIC_F32_SIZE = ei::max(sizeof(cuda::Atomic<Device::CPU, float>), sizeof(cuda::Atomic<Device::CUDA, float>));

OutputHandler::OutputHandler(u16 width, u16 height, OutputValue targets) :
	// If a texture is created without data, it does not allocate data yet.
	m_cumulativeTarget{
		GenericResource{width * height * 3 * ATOMIC_F32_SIZE},	// Radiance
		GenericResource{width * height * 3 * ATOMIC_F32_SIZE},	// Position
		GenericResource{width * height * 3 * ATOMIC_F32_SIZE},	// Albedo
		GenericResource{width * height * 3 * ATOMIC_F32_SIZE},	// Normal
		GenericResource{width * height * 1 * ATOMIC_F32_SIZE}	// Lightness
	},
	m_iterationTarget{
		GenericResource{width * height * 3 * ATOMIC_F32_SIZE},	// Radiance
		GenericResource{width * height * 3 * ATOMIC_F32_SIZE},	// Position
		GenericResource{width * height * 3 * ATOMIC_F32_SIZE},	// Albedo
		GenericResource{width * height * 3 * ATOMIC_F32_SIZE},	// Normal
		GenericResource{width * height * 1 * ATOMIC_F32_SIZE}	// Lightness
	},
	m_cumulativeVarTarget{
		GenericResource{width * height * 3 * ATOMIC_F32_SIZE},	// Radiance
		GenericResource{width * height * 3 * ATOMIC_F32_SIZE},	// Position
		GenericResource{width * height * 3 * ATOMIC_F32_SIZE},	// Albedo
		GenericResource{width * height * 3 * ATOMIC_F32_SIZE},	// Normal
		GenericResource{width * height * 1 * ATOMIC_F32_SIZE}	// Lightness
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
				rb.m_targets[i] = (RenderTarget<dev>)m_iterationTarget[i].acquire<dev>();	// Allocates if necessary
				m_iterationTarget[i].mark_changed(dev);
				mem_set<dev>(rb.m_targets[i], 0, m_iterationTarget[i].size());
				if(reset) {
					auto cumT = (RenderTarget<dev>)m_cumulativeTarget[i].acquire<dev>();		// Allocates if necessary
					mem_set<dev>(cumT, 0, m_cumulativeTarget[i].size());
					auto cumVarT = (RenderTarget<dev>)m_cumulativeVarTarget[i].acquire<dev>();		// Allocates if necessary
					mem_set<dev>(cumVarT, 0, m_cumulativeVarTarget[i].size());
				}
			} else {
				rb.m_targets[i] = (RenderTarget<dev>)m_cumulativeTarget[i].acquire<dev>();	// Allocates if necessary
				m_cumulativeTarget[i].mark_changed(dev);
				if(reset) mem_set<dev>(rb.m_targets[i], 0, m_cumulativeTarget[i].size());
			}
		}
		++i;
	}
	rb.m_resolution = ei::IVec2{m_width, m_height};

	return std::move(rb);
}

template RenderBuffer<Device::CPU> OutputHandler::begin_iteration<Device::CPU>(bool reset);
template RenderBuffer<Device::CUDA> OutputHandler::begin_iteration<Device::CUDA>(bool reset);
template RenderBuffer<Device::OPENGL> OutputHandler::begin_iteration<Device::OPENGL>(bool reset);

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
				ConstRenderTarget<Device::CUDA> iterTarget = (ConstRenderTarget<Device::CUDA>)m_iterationTarget[i].acquire_const<Device::CUDA>();
				RenderTarget<Device::CUDA> cumTarget = (RenderTarget<Device::CUDA>)m_cumulativeTarget[i].acquire<Device::CUDA>(); m_cumulativeTarget[i].mark_changed(Device::CUDA);
				RenderTarget<Device::CUDA> varTarget = (RenderTarget<Device::CUDA>)m_cumulativeVarTarget[i].acquire<Device::CUDA>(); m_cumulativeVarTarget[i].mark_changed(Device::CUDA);
				update_variance_cuda(iterTarget, cumTarget, varTarget, RenderTargets::NUM_CHANNELS[i]);
			} else {
				ConstRenderTarget<Device::CPU> iterTarget = (ConstRenderTarget<Device::CPU>)m_iterationTarget[i].acquire_const<Device::CPU>();
				RenderTarget<Device::CPU> cumTarget = (RenderTarget<Device::CPU>)m_cumulativeTarget[i].acquire<Device::CPU>(); m_cumulativeTarget[i].mark_changed(Device::CPU);
				RenderTarget<Device::CPU> varTarget = (RenderTarget<Device::CPU>)m_cumulativeVarTarget[i].acquire<Device::CPU>(); m_cumulativeVarTarget[i].mark_changed(Device::CPU);
				// TODO: openmp
				for(int y = 0; y < m_height; ++y) for(int x = 0; x < m_width; ++x)
					update_variance(iterTarget, cumTarget, varTarget, x, y,
						RenderTargets::NUM_CHANNELS[i], m_width, float(m_iteration));
			}
			// TODO: opengl
		}
		++i;
	}
}

template void OutputHandler::end_iteration<Device::CPU>();
template void OutputHandler::end_iteration<Device::CUDA>();
template void OutputHandler::end_iteration<Device::OPENGL>();


void OutputHandler::set_targets(OutputValue targets) {
	if(targets != m_targets) {
		int i = 0;
		for (u32 flag : OutputValue::iterator) {
			// Is this atttribute recorded at all?
			if(!targets.is_set(flag) && m_targets.is_set(flag)) {
				m_cumulativeTarget[i].unload<Device::CPU>();
				m_cumulativeTarget[i].unload<Device::CUDA>();
			}
			if(!targets.is_set(flag << 8) && m_targets.is_set(flag << 8)) {
				m_cumulativeVarTarget[i].unload<Device::CPU>();
				m_cumulativeVarTarget[i].unload<Device::CUDA>();
				m_iterationTarget[i].unload<Device::CPU>();
				m_iterationTarget[i].unload<Device::CUDA>();
			}
			++i;
		}
		m_targets = targets;
	}
}


std::unique_ptr<float[]> OutputHandler::get_data(OutputValue which) {
	// Is the current flag, and in case of variance its basic value, set?
	if(!m_targets.is_set(which) || (which.is_variance() && !m_targets.is_set(which >> 8))) {
		logError("[OutputHandler::get_data] The desired quantity cannot be exported, because it is not recorded!");
		return nullptr;
	}

	int quantity = which.get_quantity();

	// Allocate the memory for the output and get basic properties of the quantity to export.
	// TODO: if format has the same pixel size it is possible to avoid one memcopy by directly
	// syncing into the new texture (and then convert if necessary).
	auto data = std::make_unique<float[]>(m_width * m_height * RenderTargets::NUM_CHANNELS[quantity]);
	//Texture data(m_width, m_height, 1, exportFormat, SamplingMode::NEAREST, exportSRgb);
	//CpuTexture data(m_width, m_height, 1, exportFormat, SamplingMode::NEAREST, exportSRgb);
	bool isNormalized = !which.is_variance() && m_targets.is_set(which << 8);
	float normalizer = isNormalized ? 1.0f : ((which & 0xff) ?
								1.0f / ei::max(1, m_iteration+1) :
								1.0f / ei::max(1, m_iteration));

	// Upload to CPU / synchronize if necessary
	ConstRenderTarget<Device::CPU> src
		= which.is_variance() ?
			(ConstRenderTarget<Device::CPU>)m_cumulativeVarTarget[quantity].acquire_const<Device::CPU>() :
			(ConstRenderTarget<Device::CPU>)m_cumulativeTarget[quantity].acquire_const<Device::CPU>();

	float* dst = data.get();
	int numValues = m_width * m_height * RenderTargets::NUM_CHANNELS[quantity];
#pragma PARALLEL_FOR
	for(int i = 0; i < numValues; ++i) {
		float value = cuda::atomic_load<Device::CPU, float>(src[i]);
		dst[i] = value * normalizer;
	}

	return data;
}

ei::Vec4 OutputHandler::get_pixel_value(OutputValue which, Pixel pixel) {
	// Is the current flag, and in case of variance its basic value, set?
	if(!m_targets.is_set(which) || (which.is_variance() && !m_targets.is_set(which >> 8))) {
		logError("[OutputHandler::get_pixel_value] The desired quantity cannot be exported, because it is not recorded!");
		return ei::Vec4{0.0f};
	}
	if(pixel.x < 0 || pixel.x >= m_width || pixel.y < 0 || pixel.y >= m_height) {
		logError("[OutputHandler::get_pixel_value] The query pixel is out of range.");
		return ei::Vec4{0.0f};
	}

	int quantity = which.get_quantity();
	bool isNormalized = !which.is_variance() && m_targets.is_set(which << 8);
	float normalizer = isNormalized ? 1.0f : ((which & 0xff) ?
											  1.0f / ei::max(1, m_iteration + 1) :
											  1.0f / ei::max(1, m_iteration));
	// Upload to CPU / synchronize if necessary
	ConstArrayDevHandle_t<Device::CPU, cuda::Atomic<Device::CPU, float>> src
		= which.is_variance() ?
			(ConstArrayDevHandle_t<Device::CPU, cuda::Atomic<Device::CPU, float>>)m_cumulativeVarTarget[quantity].acquire_const<Device::CPU>() :
			(ConstArrayDevHandle_t<Device::CPU, cuda::Atomic<Device::CPU, float>>)m_cumulativeTarget[quantity].acquire_const<Device::CPU>();

	int idx = pixel.x + pixel.y * m_width;
	if(RenderTargets::NUM_CHANNELS[quantity] == 3) {
		idx *= 3;
		float r = cuda::atomic_load<Device::CPU, float>(src[idx]);
		float g = cuda::atomic_load<Device::CPU, float>(src[idx+1]);
		float b = cuda::atomic_load<Device::CPU, float>(src[idx+2]);
		return {r * normalizer, g * normalizer, b * normalizer, 0.0f};
	} else {
		float l = cuda::atomic_load<Device::CPU, float>(src[idx]);
		return {l * normalizer, 0.0f, 0.0f, 0.0f};
	}
}

} // namespace mufflon::renderer