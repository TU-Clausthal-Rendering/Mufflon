#include "output_handler.hpp"

using namespace mufflon::scene::textures;

namespace mufflon { namespace renderer {

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
	m_iteration(0)
{
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
				rb.m_radiance = *m_iterationTex[i].aquire<dev>();	// Allocates if necessary
				m_iterationTex[i].clear<dev>();
				if(reset) {
					m_cumulativeTex[i].aquire<dev>();		// Allocates if necessary
					m_cumulativeVarTex[i].aquire<dev>();	// Allocates if necessary
					m_cumulativeTex[i].clear<dev>();
					m_cumulativeVarTex[i].clear<dev>(); // TODO: async
				}
			} else {
				rb.m_radiance = *m_cumulativeTex[i].aquire<dev>();	// Allocates if necessary
				if(reset) m_cumulativeTex[i].clear<dev>();
			}
		}
		++i;
	}

	return std::move(rb);
}

template RenderBuffer<Device::CPU> OutputHandler::begin_iteration<Device::CPU>(bool);
template RenderBuffer<Device::CUDA> OutputHandler::begin_iteration<Device::CUDA>(bool);



template < Device dev > __host__ __device__ void
update_variance(const TextureDevHandle_t<dev>& iterTex,
				const TextureDevHandle_t<dev>& cumTex,
				const TextureDevHandle_t<dev>& varTex,
				int x, int y, float iteration
) {
	ei::Vec3 cum { read(cumTex, Pixel{x,y}) };
	ei::Vec3 iter { read(iterTex, Pixel{x,y}) };
	ei::Vec3 var { read(varTex, Pixel{x,y}) };
	// Use a stable addition scheme for the variance
	ei::Vec3 diff = iter - cum;
	cum += diff / ei::max(1.0f, iteration);
	var += diff * (iter - cum);
	write(cumTex, Pixel{x,y}, {cum, 0.0f});
	write(varTex, Pixel{x,y}, {var, 0.0f});
}

__global__ void update_variance(TextureDevHandle_t<Device::CUDA> iterTex,
								TextureDevHandle_t<Device::CUDA> cumTex,
								TextureDevHandle_t<Device::CUDA> varTex,
								float iteration
) {
	int x = int(blockIdx.x * blockDim.x + threadIdx.x);
	int y = int(blockIdx.y * blockDim.y + threadIdx.y);
	update_variance<Device::CUDA>(iterTex, cumTex, varTex, x, y, iteration);
}

template < Device dev >
void OutputHandler::end_iteration() {
	int i = 0;
	for(u32 flag : OutputValue::iterator) {
		// If variances are computed we need some post processing (the renderer
		// provides the iteration sample only).
		if(m_targets.is_set(flag) && m_targets.is_set(flag << 8)) {
			if(dev == Device::CUDA) {
				// Looks like inavitable redundancy. update_variance<<<>>> call is only valid
				// for one type. Without 'if constexpr' this branch would not compile. if 
				// handles are initialized TextureDevHandle_t<dev> iterTex...
				TextureDevHandle_t<Device::CUDA> iterTex = *m_iterationTex[i].aquire<Device::CUDA>();
				TextureDevHandle_t<Device::CUDA> cumTex = *m_cumulativeTex[i].aquire<Device::CUDA>();
				TextureDevHandle_t<Device::CUDA> varTex = *m_cumulativeVarTex[i].aquire<Device::CUDA>();
				dim3 dimBlock(16,16);
				dim3 dimGrid((m_iterationTex->get_width() + dimBlock.x-1) / dimBlock.x,
							 (m_iterationTex->get_height() + dimBlock.y-1) / dimBlock.y);
				update_variance<<<dimGrid,dimBlock>>>(iterTex, cumTex, varTex, float(m_iteration));
			} else {
				TextureDevHandle_t<Device::CPU> iterTex = *m_iterationTex[i].aquire<Device::CPU>();
				TextureDevHandle_t<Device::CPU> cumTex = *m_cumulativeTex[i].aquire<Device::CPU>();
				TextureDevHandle_t<Device::CPU> varTex = *m_cumulativeVarTex[i].aquire<Device::CPU>();
				// TODO: openmp
				for(int y = 0; y < m_iterationTex->get_height(); ++y)
					for(int x = 0; x < m_iterationTex->get_width(); ++x)
						update_variance<Device::CPU>(iterTex, cumTex, varTex, x, y, float(m_iteration));
			}
			// TODO: opengl
		}
		++i;
	}
}

template void OutputHandler::end_iteration<Device::CPU>();
template void OutputHandler::end_iteration<Device::CUDA>();

}} // namespace mufflon::renderer