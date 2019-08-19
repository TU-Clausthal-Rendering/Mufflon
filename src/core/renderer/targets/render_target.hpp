#pragma once

#include "util/int_types.hpp"
#include "util/tagged_tuple.hpp"
#include "core/cuda/cuda_utils.hpp"
#include "core/export/api.h"
#include "core/memory/residency.hpp"
#include <ei/vector.hpp>
#include <type_traits>

namespace mufflon { namespace renderer {

template < class... >
class OutputHandler;
template < Device, class... >
class RenderBuffer;

// This is a helper you can use to shorten your definitions and share the same target list
// across different renderer (variations)
template < class... Ts >
struct TargetList {
	using OutputHandlerType = OutputHandler<Ts...>;
	template < Device dev >
	using RenderBufferType = RenderBuffer<dev, Ts...>;
	static constexpr std::size_t TARGET_COUNT = sizeof...(Ts);
};

// Helper typedef's to shorten some code
// Takes the device and the pixel type (float, double, ...)
template < Device dev, class T >
using RenderTargetBuffer = ArrayDevHandle_t<dev, cuda::Atomic<dev, T>>;
template < Device dev, class T >
using ConstRenderTargetBuffer = ConstArrayDevHandle_t<dev, cuda::Atomic<dev, T>>;

// Takes the device and the Target type (RadianceTarget, NormalTarget, ...)
template < Device dev, class TargetType >
using RenderTarget = RenderTargetBuffer<dev, typename TargetType::PixelType>;
template < Device dev, class TargetType >
using ConstRenderTarget = ConstRenderTargetBuffer<dev, typename TargetType::PixelType>;

template < Device dev, class... Ts >
class RenderBuffer {
public:
	static constexpr std::size_t TARGET_COUNT = sizeof...(Ts);

	RenderBuffer() = default;

	RenderBuffer(ei::IVec2 resolution, ArrayDevHandle_t<dev, cuda::Atomic<dev, u32>> nanCounterPtr) :
		m_resolution(resolution),
		m_nanCounter(nanCounterPtr) {}

	/*template < class T, class... Args >
	CUDA_FUNCTION void contribute(Pixel pixel, Args&& ...args) {
		using PixelType = typename T::PixelType;
		auto& target = m_targets.template get<Target<T>>();
		auto& pixels = reinterpret_cast<cuda::Atomic<dev, PixelType>(*)[T::NUM_CHANNELS]>(target.pixels)[pixel.x + pixel.y * m_resolution.x];
		// TODO: NaN-check isn't really possible here if we don't know which of
		// the targets we're supposed to check!
		T::template contribute<dev>(pixels, static_cast<Args&&>(args)...);
	}*/

	template < class T >
	CUDA_FUNCTION void contribute(const Pixel pixel, const ei::Vec<typename T::PixelType, T::NUM_CHANNELS>& value) {
		static_assert(TargetTupleType::template has<Target<T>>(), "Requested render target doesn't exist");
		mAssert(pixel.x < m_resolution.x && pixel.y < m_resolution.y);
		using PixelType = typename T::PixelType;
		auto& target = m_targets.template get<Target<T>>();
		if(target.pixels) {
			auto& pixels = reinterpret_cast<cuda::Atomic<dev, PixelType>(*)[T::NUM_CHANNELS]>(target.pixels)[pixel.x + pixel.y * m_resolution.x];
			for(u32 i = 0u; i < T::NUM_CHANNELS; ++i) {
				cuda::atomic_add<dev>(pixels[i], check_nan(value[i]));
			}
		}
	}
	// Overload for single-value targets
	template < class T >
	CUDA_FUNCTION void contribute(const Pixel pixel, const typename T::PixelType& value) {
		this->contribute<T>(pixel, ei::Vec<typename T::PixelType, T::NUM_CHANNELS>{ value });
	}

	template < class T >
	CUDA_FUNCTION void set(Pixel pixel, const ei::Vec<typename T::PixelType, T::NUM_CHANNELS>& value) {
		static_assert(TargetTupleType::template has<Target<T>>(), "Requested render target doesn't exist");
		mAssert(pixel.x < m_resolution.x && pixel.y < m_resolution.y);
		using PixelType = typename T::PixelType;
		auto& target = m_targets.template get<Target<T>>();
		if(target.pixels) {
			auto& pixels = reinterpret_cast<cuda::Atomic<dev, PixelType>(*)[T::NUM_CHANNELS]>(target.pixels)[pixel.x + pixel.y * m_resolution.x];
			for(u32 i = 0u; i < T::NUM_CHANNELS; ++i)
				cuda::atomic_exchange<dev>(pixels[i], check_nan(value[i]));
		}
	}
	// Overload for single-value targets
	template < class T >
	CUDA_FUNCTION void set(const Pixel pixel, const typename T::PixelType& value) {
		this->set<T>(pixel, ei::Vec<typename T::PixelType, T::NUM_CHANNELS>{ value });
	}

	template < class T >
	__host__ void set_target_buffer(RenderTarget<dev, T> buffer) {
		m_targets.template get<Target<T>>().pixels = buffer;
	}

	// Retrieves the value; if target is not enabled, default value is returned
	// CAREFUL: no clamping is performed, so you have to make sure that the coordinates
	// are within screen limits
	template < class T >
	CUDA_FUNCTION std::enable_if_t<T::NUM_CHANNELS != 1, ei::Vec<typename T::PixelType, T::NUM_CHANNELS>> get(const Pixel pixel) const {
		static_assert(TargetTupleType::template has<Target<T>>(), "Requested render target doesn't exist");
		mAssert(pixel.x < m_resolution.x && pixel.y < m_resolution.y);
		using PixelType = typename T::PixelType;
		const auto& target = m_targets.template get<Target<T>>();
		ei::Vec<typename T::PixelType, T::NUM_CHANNELS> res{ typename T::PixelType{0} };
		if(target.pixels) {
			const auto& pixels = reinterpret_cast<const cuda::Atomic<dev, PixelType>(*)[T::NUM_CHANNELS]>(target.pixels)[pixel.x + pixel.y * m_resolution.x];
			for(u32 i = 0u; i < T::NUM_CHANNELS; ++i)
				res[i] = cuda::atomic_load<dev, typename T::PixelType>(pixels[i]);
		}
		return res;
	}
	// Overload for single-value targets
	template < class T >
	CUDA_FUNCTION std::enable_if_t<T::NUM_CHANNELS == 1, typename T::PixelType> get(const Pixel pixel) const {
		static_assert(TargetTupleType::template has<Target<T>>(), "Requested render target doesn't exist");
		mAssert(pixel.x < m_resolution.x && pixel.y < m_resolution.y);
		using PixelType = typename T::PixelType;
		const auto& target = m_targets.template get<Target<T>>();
		typename T::PixelType res{ 0 };
		if(target.pixels) {
			const auto& pixels = reinterpret_cast<const cuda::Atomic<dev, PixelType>*>(target.pixels)[pixel.x + pixel.y * m_resolution.x];
			res = cuda::atomic_load<dev, typename T::PixelType>(pixels);
		}
		return res;
	}

	template < class T >
	CUDA_FUNCTION bool is_target_enabled() const noexcept {
		return m_targets.template get<Target<T>>().pixels;
	}

	CUDA_FUNCTION bool is_target_enabled(const u32 index) const noexcept {
		if(index >= TARGET_COUNT)
			return false;
		u32 i = 0u;
		bool enabled = false;
		m_targets.for_each([index, &i, &enabled](auto& target) {
			// Since tagged_tuple's for_each iterates front to back we have to invert the index
			if(TARGET_COUNT - i - 1 == index)
				enabled = target.pixels == nullptr;
			++i;
		});
		return enabled;
	}

	CUDA_FUNCTION ConstRenderTargetBuffer<dev, char> get_target(const u32 index) {
		if(index >= TARGET_COUNT)
			return {};

		ConstRenderTargetBuffer<dev, char> targetBuffer;
		u32 i = 0u;
		m_targets.for_each([index, &i, &targetBuffer](auto& target) {
			if(i == index)
				targetBuffer = (ConstRenderTargetBuffer<dev, char>)target.pixels;
			++i;
		});
		return targetBuffer;
	}

	CUDA_FUNCTION int get_width() const { return m_resolution.x; }
	CUDA_FUNCTION int get_height() const { return m_resolution.y; }
	CUDA_FUNCTION ei::IVec2 get_resolution() const { return m_resolution; }
	CUDA_FUNCTION int get_num_pixels() const { return m_resolution.x * m_resolution.y; }

private:
	template < class T >
	struct Target {
		using TargetType = T;
		using PixelType = typename T::PixelType;
		RenderTarget<dev, TargetType> pixels = {};
	};

	using TargetTupleType = util::TaggedTuple<Target<Ts>...>;

	// Two versions with enable_if so we don't have to use helper structs again: one
	// for floating-point types, one for the rest (which doesn't have NaN-checks)
	template < class T >
	CUDA_FUNCTION std::enable_if_t<std::is_floating_point<T>::value, T> check_nan(const T x) {
		if(isnan(x)) {
			cuda::atomic_add<dev>(*m_nanCounter, 1u);
			return T{ 0 };
		}
		return x;
	}
	template < class T >
	__forceinline__ __host__ __device__ std::enable_if_t<!std::is_same<T, float>::value, T> check_nan(const T x) {
		// No NaN possible for non-FP types
		return x;
	}

	// The following texture handles may contain iteration only or all iterations summed
	// information. The meaning of the variables is only known to the OutputHandler.
	// The renderbuffer only needs to add values to all defined handles.
	TargetTupleType m_targets;
	ei::IVec2 m_resolution;
	// This holds the address of an atomic used to count the number of NaNs in an iteration
	// TODO: this doesn't play nicely with OpenGL
	template < Device d >
	using NanPtr = ArrayDevHandle_t<d, cuda::Atomic<d, u32>>;
	ArrayDevHandle_t<dev, cuda::Atomic<dev, u32>> m_nanCounter;

	//std::conditional_t<dev == Device::OPENGL, NanPtr<Device::CPU>, NanPtr<dev>> m_nanCounter;
};

}} // namespace mufflon::renderer