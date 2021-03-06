#pragma once

#include "render_target.hpp"
#include "output_handler_variance.hpp"
#include "util/parallel.hpp"
#include "util/string_view.hpp"
#include "util/tagged_tuple.hpp"
#include "util/type_helpers.hpp"
#include "core/cuda/cuda_utils.hpp"
#include "core/math/sample_types.hpp"
#include "core/memory/generic_resource.hpp"
#include "core/memory/synchronize.hpp"
#include "core/scene/textures/texture.hpp"
#include "core/scene/textures/cputexture.hpp"
#include "core/scene/textures/interface.hpp"
#include <type_traits>

namespace mufflon { namespace renderer {

namespace output_handler_details {

u32* get_cuda_nan_counter_ptr_and_set_zero();
u32 get_cuda_nan_counter_value();

template < class PixelType, bool ReduceMoments >
void update_iter_cuda(ConstRenderTargetBuffer<Device::CUDA, PixelType> iterTarget,
					  RenderTargetBuffer<Device::CUDA, float> cumTarget,
					  RenderTargetBuffer<Device::CUDA, float> varTarget,
					  int numChannels, int width, int height, int iteration);

// Helper structs to detect whether a render target is required (always on and not disable-able)
template < class P, class Enable = void >
struct IsRequired : std::false_type {};
template < class P >
struct IsRequired<P, std::enable_if_t<P::REQUIRED>> : std::true_type {};
template < class P, class Enable = void >
struct IsVarianceRequired : std::false_type {};
template < class P >
struct IsVarianceRequired<P, std::enable_if_t<P::VARIANCE_REQUIRED>> : std::true_type {};
template < class P, class Enable = void >
struct IsReduceMomentsEnabled : std::false_type {};
template < class P >
struct IsReduceMomentsEnabled<P, std::enable_if_t<P::REDUCE_MOMENTS>> : std::true_type {};

} // namespace output_handler_details 

// Interface class for all output handlers
class IOutputHandler {
public:
	virtual ~IOutputHandler() = default;

	virtual std::size_t get_render_target_count() const noexcept = 0;
	virtual StringView get_render_target_name(std::size_t index) const = 0;
	virtual std::unique_ptr<float[]> get_data(StringView name, const bool variance) = 0;
	virtual ei::Vec4 get_pixel_value(Pixel pixel, StringView name, const bool variance) = 0;
	virtual void resize(int width, int height) = 0;

	virtual void enable_render_target(StringView name, const bool variance) = 0;
	virtual void disable_render_target(StringView name, const bool variance) = 0;
	virtual void enable_all_render_targets(const bool includeVariance) noexcept = 0;
	virtual void disable_all_render_targets(const bool varianceOnly) noexcept = 0;
	virtual bool has_render_target(StringView name) const = 0;
	virtual bool is_render_target_enabled(StringView name, const bool variance) const = 0;
	virtual bool is_render_target_required(StringView name, const bool variance) const = 0;
	virtual u32 get_num_channels(StringView targetName) const = 0;
	virtual int get_width() const noexcept = 0;
	virtual int get_height() const noexcept = 0;
	virtual int get_current_iteration() const noexcept = 0;
};

template < class... Ts >
class OutputHandler final : public IOutputHandler {
public:
	static_assert(have_distinct_names<Ts...>(),
				  "All render targets for an output handler must have distinct names");

	template < Device dev >
	using RenderBufferType = RenderBuffer<dev, Ts...>;
	static constexpr std::size_t TARGET_COUNT = sizeof...(Ts);

	OutputHandler(u16 width, u16 height) :
		m_iteration(-1),
		m_width(width),
		m_height(height) {
		if(width <= 0 || height <= 0) {
			logError("[OutputHandler::OutputHandler] Invalid resolution (<= 0)");
		} else {
			m_targets.for_each([width, height](auto& elem) {
				using Type = std::decay_t<decltype(elem)>;
				using TargetType = typename Type::TargetType;
				using PixelType = typename TargetType::PixelType;
				constexpr std::size_t PIXEL_TYPE_SIZE = std::max(sizeof(cuda::Atomic<Device::CPU, PixelType>),
																 sizeof(cuda::Atomic<Device::CUDA, PixelType>));
				const std::size_t bytes = width * height * TargetType::NUM_CHANNELS * PIXEL_TYPE_SIZE;
				elem.cumulative = GenericResource{ bytes };
				elem.iteration = GenericResource{ bytes };
				elem.cumulativeVariance = GenericResource{ bytes };
				if(output_handler_details::IsRequired<TargetType>::value)
					elem.record = true;
				if(output_handler_details::IsVarianceRequired<TargetType>::value) {
					elem.record = true;
					elem.recordVariance = true;
				}
			});
		}
	}

	void resize(int width, int height) override {
		if(width <= 0 || height <= 0) {
			logError("[OutputHandler::OutputHandler] Invalid resolution (<= 0)");
		} else {
			m_targets.for_each([width, height](auto& elem) {
				using Type = std::decay_t<decltype(elem)>;
				using TargetType = typename Type::TargetType;
				using PixelType = typename TargetType::PixelType;
				constexpr std::size_t PIXEL_TYPE_SIZE = std::max(sizeof(cuda::Atomic<Device::CPU, PixelType>),
																 sizeof(cuda::Atomic<Device::CUDA, PixelType>));
				const std::size_t bytes = width * height * TargetType::NUM_CHANNELS * PIXEL_TYPE_SIZE;
				elem.cumulative.resize(bytes);
				elem.iteration.resize(bytes);
				elem.cumulativeVariance.resize(bytes);
				if(output_handler_details::IsRequired<TargetType>::value)
					elem.record = true;
				if(output_handler_details::IsVarianceRequired<TargetType>::value) {
					elem.record = true;
					elem.recordVariance = true;
				}
			});
			m_width = width;
			m_height = height;
			m_iteration = -1;
		}
	}

	// Allocate and clear the buffers. Which buffers are returned depends on the
	// 'targets' which where set in the constructor.
	template < Device dev >
	RenderBuffer<dev, Ts...> begin_iteration(bool reset) {
		m_iteration = reset ? 0 : m_iteration + 1;

		RenderBuffer<dev, Ts...> rb{ ei::IVec2{m_width, m_height}, get_and_reset_nan_counter<dev>() };

		m_targets.for_each([&rb, reset](auto& target) {
			using Type = std::decay_t<decltype(target)>;
			using TargetType = typename Type::TargetType;
			if(target.record) {
				// Always accumulate per iteration and use a numerical more robust
				// solution for summation afterwards.
				auto targetBuffer = (RenderTarget<dev, TargetType>)target.iteration.template acquire<dev>();
				rb.template set_target_buffer<TargetType>(targetBuffer);
				target.iteration.mark_changed(dev);
				mem_set<dev>(targetBuffer, 0, target.iteration.size());
				if(reset) {
					auto cumTarget = target.cumulative.template acquire<dev>();
					mem_set<dev>(cumTarget, 0, target.cumulative.size());
					if(target.recordVariance) {
						auto cumVarTarget = target.cumulativeVariance.template acquire<dev>();
						mem_set<dev>(cumVarTarget, 0, target.cumulativeVariance.size());
					}
				}
			}
		});

		return rb;
	}
	
	template < Device dev1, Device dev2 >
	std::pair<RenderBuffer<dev1, Ts...>, RenderBuffer<dev2, Ts...>> begin_iteration_hybrid(bool reset) {
		m_iteration = reset ? 0 : m_iteration + 1;

		RenderBuffer<dev1, Ts...> rb1{ ei::IVec2{m_width, m_height}, get_and_reset_nan_counter<dev1>() };
		RenderBuffer<dev2, Ts...> rb2{ ei::IVec2{m_width, m_height}, get_and_reset_nan_counter<dev2>() };

		m_targets.for_each([&rb1, &rb2, reset](auto& target) {
			using Type = std::decay_t<decltype(target)>;
			using TargetType = typename Type::TargetType;
			if(target.record) {
				auto targetBuffer1 = reinterpret_cast<RenderTarget<dev1, TargetType>>(target.iteration.template acquire<dev1>(false));
				target.iteration.mark_changed(dev1);
				auto targetBuffer2 = reinterpret_cast<RenderTarget<dev2, TargetType>>(target.iteration.template acquire<dev2>(false));
				rb1.template set_target_buffer<TargetType>(targetBuffer1);
				rb2.template set_target_buffer<TargetType>(targetBuffer2);
				// TODO: for this a "mark-out-of-sync without unloading" would be perfect
				mem_set<dev1>(targetBuffer1, 0, target.iteration.size());
				mem_set<dev2>(targetBuffer2, 0, target.iteration.size());
				if(reset) {
					auto cumTarget1 = target.cumulative.template acquire<dev1>();
					auto cumTarget2 = target.cumulative.template acquire<dev2>();
					mem_set<dev1>(cumTarget1, 0, target.cumulative.size());
					mem_set<dev2>(cumTarget2, 0, target.cumulative.size());
					if(target.recordVariance) {
						auto cumVarTarget1 = target.cumulativeVariance.template acquire<dev1>();
						auto cumVarTarget2 = target.cumulativeVariance.template acquire<dev2>();
						mem_set<dev1>(cumVarTarget1, 0, target.cumulativeVariance.size());
						mem_set<dev2>(cumVarTarget2, 0, target.cumulativeVariance.size());
					}
				}
			}
		});

		return { std::move(rb1), std::move(rb2) };
	}

	// Do some finalization, like variance computations
	template < Device dev >
	void end_iteration() {
		m_targets.for_each([=](auto& target) {
			using Type = std::decay_t<decltype(target)>;
			using TargetType = typename Type::TargetType;
			using PixelType = typename TargetType::PixelType;
			constexpr bool REDUCE_MOMENTS = output_handler_details::IsReduceMomentsEnabled<TargetType>::value;
			if(target.record) {
				auto iterTarget = (ConstRenderTarget<dev, TargetType>)target.iteration.template acquire_const<dev>();
				auto cumTarget = (RenderTargetBuffer<dev, float>)target.cumulative.template acquire<dev>();
				target.cumulative.mark_changed(dev);
				RenderTargetBuffer<dev, float> varTarget = {};
				if(target.recordVariance) {
					varTarget = (RenderTargetBuffer<dev, float>)target.cumulativeVariance.template acquire<dev>();
					target.cumulativeVariance.mark_changed(dev);
				}
				if constexpr(dev == Device::CPU) {
					for(int y = 0; y < m_height; ++y) for(int x = 0; x < m_width; ++x)
						output_handler_details::UpdateIter<PixelType, REDUCE_MOMENTS>::f(
							iterTarget, cumTarget, varTarget, x, y,
							TargetType::NUM_CHANNELS, m_width, float(m_iteration));

				} else if constexpr(dev == Device::CUDA) {
					output_handler_details::update_iter_cuda<PixelType, REDUCE_MOMENTS>(
						iterTarget, cumTarget, varTarget,
						TargetType::NUM_CHANNELS, m_width, m_height, m_iteration);
				} else {
					// Simple copy (this means that OpenGL renderers do not average their images!)
					copy(cumTarget, iterTarget, m_width * m_height * TargetType::NUM_CHANNELS * sizeof(PixelType));
				}
			}
		});

		// Check the NaN counter
		u32 counter;
		if constexpr(dev == Device::CUDA) {
			counter = output_handler_details::get_cuda_nan_counter_value();
		} else {
			counter = m_nanCounter.load();
		}

		if(counter > 0u)
			logWarning("[RenderBuffer] Detected NaN on output (", counter, " times). 0 was returned instead.");
	}

	template < Device from, Device to >
	void sync_back(const int ySplit) {
		static_assert(to == Device::CPU, "Cannot sync to other devices yet!");

		const int actualSplit = std::min(ySplit, get_height());
		
		m_targets.for_each([this, actualSplit](auto& target) {
			using Type = std::decay_t<decltype(target)>;
			using TargetType = typename Type::TargetType;
			using PixelType = typename TargetType::PixelType;

			if(target.record) {
				ArrayDevHandle_t<to, cuda::Atomic<to, PixelType>> dst;
				ConstArrayDevHandle_t<from, cuda::Atomic<from, PixelType>> src;
				dst = reinterpret_cast<ArrayDevHandle_t<to, cuda::Atomic<to, PixelType>>>(target.iteration.template acquire<to>(false));
				src = reinterpret_cast<ConstArrayDevHandle_t<from, cuda::Atomic<from, PixelType>>>(target.iteration.template acquire<from>(false));

				const std::size_t PIXEL_SIZE_FROM = TargetType::NUM_CHANNELS * sizeof(cuda::Atomic<from, PixelType>);
				const std::size_t PIXEL_SIZE_TO = TargetType::NUM_CHANNELS * sizeof(cuda::Atomic<to, PixelType>);
				std::size_t index = static_cast<std::size_t>(get_width() * actualSplit);
				src += index;
				dst += index;

				if constexpr((PIXEL_SIZE_FROM != PIXEL_SIZE_TO) && (to == Device::CPU || from == Device::CPU)) {
					// We cannot simply memcpy due to the possibly differing sizes of atomics
					for(int y = actualSplit; y < get_height(); ++y) {
						for(int x = 0; x < get_width(); ++x) {
							if constexpr(to == Device::CPU) {
								cuda::Atomic<from, PixelType> val;
								copy(&val, src, PIXEL_SIZE_FROM);
								const auto loaded = cuda::atomic_load<from, PixelType>(val);
								cuda::atomic_exchange<to, PixelType>(*dst, loaded);
							}
							++index;
						}
					}
				} else {
					// Simple memcpy is enough
					copy((ArrayDevHandle_t<to, char>)dst, (ConstArrayDevHandle_t<from, char>)src,
						 get_width() * (get_height() - actualSplit) * PIXEL_SIZE_FROM);
				}
				// TODO: mark changed?
			}
		});
	}

	// Get the formated output of one quantity for the purpose of exporting screenshots.
	// which: The quantity to export. Causes an error if the quantity is not recorded.
	// The returned buffer is either Vec3 or float, depending on the number of channels in
	// the queried quantity.
	template < class TargetType >
	std::unique_ptr<float[]> get_data(const bool variance) {
		auto& target = m_targets.template get<Target<TargetType>>();
		if(variance && !target.recordVariance) {
			logError("[OutputHandler::get_data] Render target '", TargetType::NAME, "' does not record variance!");
			return nullptr;
		}

		const int numValues = m_width * m_height * TargetType::NUM_CHANNELS;
		auto data = std::make_unique<float[]>(numValues);
		auto src = variance
			? reinterpret_cast<ConstRenderTargetBuffer<Device::CPU, float>>(target.cumulativeVariance.template acquire_const<Device::CPU>())
			: reinterpret_cast<ConstRenderTargetBuffer<Device::CPU, float>>(target.cumulative.template acquire_const<Device::CPU>());

#pragma PARALLEL_FOR
		for(int i = 0; i < numValues; ++i) {
			const float value = cuda::atomic_load<Device::CPU, float>(src[i]);
			data[i] = value;
		}

		return data;
	}

	// Returns the value of a pixel as a Vec4, regardless of the underlying format
	template < class TargetType >
	ei::Vec4 get_pixel_value(Pixel pixel, const bool variance) {
		auto& target = m_targets.template get<Target<TargetType>>();
		if(variance && !target.recordVariance) {
			logError("[OutputHandler::get_data] Render target '", TargetType::NAME, "' does not record variance!");
			return ei::Vec4{0.f};
		}

		auto src = variance
			? reinterpret_cast<ConstRenderTargetBuffer<Device::CPU, float>>(target.cumulativeVariance.template acquire_const<Device::CPU>())
			: reinterpret_cast<ConstRenderTargetBuffer<Device::CPU, float>>(target.cumulative.template acquire_const<Device::CPU>());

		const int idx = pixel.x + pixel.y * m_width * TargetType::NUM_CHANNELS;
		if(TargetType::NUM_CHANNELS > 4)
			logWarning("[OutputHandler::get_pixel_value] Render target has more than 4 channels (returning first 4)");
		ei::Vec4 res{ 0.f };
		for(u32 i = 0u; i < std::max(TargetType::NUM_CHANNELS, 4u); ++i) {
			const float channel = cuda::atomic_load<Device::CPU, float>(src[idx + i]);
			res[i] = channel;
		}
		return res;
	}

	std::unique_ptr<float[]> get_data(StringView name, const bool variance) override {
		std::unique_ptr<float[]> res;
		m_targets.for_each([&res, name, variance, this](auto& target) {
			using Type = std::decay_t<decltype(target)>;
			using TargetType = typename Type::TargetType;
			if(name.compare(TargetType::NAME) == 0)
				res = this->get_data<TargetType>(variance);
		});

		if(res == nullptr)
			logError("[OutputHandler::get_data] No target named '", name, "' exists");
		return res;
	}

	ei::Vec4 get_pixel_value(Pixel pixel, StringView name, const bool variance) override {
		ei::Vec4 res{ 0.f };
		m_targets.for_each([&res, pixel, name, variance, this](auto& target) {
			using Type = std::decay_t<decltype(target)>;
			using TargetType = typename Type::TargetType;
			if(name.compare(TargetType::NAME) == 0)
				res = this->get_pixel_value<TargetType>(pixel, variance);
		});
		return res;
	}

	std::size_t get_render_target_count() const noexcept override { return TARGET_COUNT; }
	StringView get_render_target_name(std::size_t index) const override {
		if(index >= TARGET_COUNT)
			throw std::runtime_error("Render target index out of range (" + std::to_string(index)
									 + " >= " + std::to_string(TARGET_COUNT) + ")");

		const char* name = nullptr;
		std::size_t i = 0;
		m_targets.for_each([&name, &i, index](auto& target) {
			using Type = std::decay_t<decltype(target)>;
			using TargetType = typename Type::TargetType;
			// Since tagged_tuple iterates back to front we need to invert the index here
			if(TARGET_COUNT - i - 1 == index)
				name = TargetType::NAME;
			++i;
		});
		return name;
	}

	void enable_render_target(StringView name, const bool variance) override {
		bool foundTarget = false;
		m_targets.for_each([name, variance, &foundTarget](auto& target) {
			using Type = std::decay_t<decltype(target)>;
			using TargetType = typename Type::TargetType;
			if(!foundTarget && name.compare(TargetType::NAME) == 0) {
				foundTarget = true;
				target.record = true;
				if(variance)
					target.recordVariance = true;
			}
		});
		if(!foundTarget)
			throw std::runtime_error("Cannot find render target '" + std::string(name) + "'");
	}

	void disable_render_target(StringView name, const bool variance) override {
		bool foundTarget = false;
		m_targets.for_each([name, variance, &foundTarget](auto& target) {
			using Type = std::decay_t<decltype(target)>;
			using TargetType = typename Type::TargetType;
			if(!foundTarget && name.compare(TargetType::NAME) == 0) {
				foundTarget = true;
				if(!output_handler_details::IsVarianceRequired<TargetType>::value) {
					target.recordVariance = false;
					target.cumulativeVariance.template unload<Device::CPU>();
					target.cumulativeVariance.template unload<Device::CUDA>();
					target.cumulativeVariance.template unload<Device::OPENGL>();
					if(!variance && !output_handler_details::IsRequired<TargetType>::value) {
						target.record = false;
						target.iteration.template unload<Device::CPU>();
						target.iteration.template unload<Device::CUDA>();
						target.iteration.template unload<Device::OPENGL>();
						target.cumulative.template unload<Device::CPU>();
						target.cumulative.template unload<Device::CUDA>();
						target.cumulative.template unload<Device::OPENGL>();
					}
				}
			}
		});
		if(!foundTarget)
			throw std::runtime_error("Cannot find render target '" + std::string(name) + "'");
	}

	void enable_all_render_targets(const bool includeVariance) noexcept override {
		m_targets.for_each([includeVariance](auto& target) {
			target.record = true;
			if(includeVariance)
				target.recordVariance = true;
		});
	}
	void disable_all_render_targets(const bool varianceOnly) noexcept override {
		m_targets.for_each([varianceOnly](auto& target) {
			using Type = std::decay_t<decltype(target)>;
			using TargetType = typename Type::TargetType;
			if(!output_handler_details::IsVarianceRequired<TargetType>::value) {
				target.recordVariance = false;
				target.cumulativeVariance.template unload<Device::CPU>();
				target.cumulativeVariance.template unload<Device::CUDA>();
				target.cumulativeVariance.template unload<Device::OPENGL>();
				if(!varianceOnly && !output_handler_details::IsRequired<TargetType>::value) {
					target.record = false;
					target.iteration.template unload<Device::CPU>();
					target.iteration.template unload<Device::CUDA>();
					target.iteration.template unload<Device::OPENGL>();
					target.cumulative.template unload<Device::CPU>();
					target.cumulative.template unload<Device::CUDA>();
					target.cumulative.template unload<Device::OPENGL>();
				}
			}
		});
	}

	bool has_render_target(StringView name) const override {
		bool foundTarget = false;
		m_targets.for_each([name, &foundTarget](auto& target) {
			using Type = std::decay_t<decltype(target)>;
			using TargetType = typename Type::TargetType;
			if(!foundTarget && name.compare(TargetType::NAME) == 0)
				foundTarget = true;
		});
		return foundTarget;
	}

	bool is_render_target_enabled(StringView name, const bool variance) const override {
		bool foundTarget = false;
		bool enabled = false;
		m_targets.for_each([name, variance, &foundTarget, &enabled](auto& target) {
			using Type = std::decay_t<decltype(target)>;
			using TargetType = typename Type::TargetType;
			if(!foundTarget && name.compare(TargetType::NAME) == 0) {
				foundTarget = true;
				enabled = variance ? target.recordVariance : target.record;
			}
		});
		if(!foundTarget)
			throw std::runtime_error("Cannot find render target '" + std::string(name) + "'");
		return enabled;
	}

	bool is_render_target_required(StringView name, const bool variance) const override {
		bool foundTarget = false;
		bool required = false;
		m_targets.for_each([name, variance, &foundTarget, &required](auto& target) {
			using Type = std::decay_t<decltype(target)>;
			using TargetType = typename Type::TargetType;
			if(!foundTarget && name.compare(TargetType::NAME) == 0) {
				foundTarget = true;
				required = variance
					? output_handler_details::IsVarianceRequired<TargetType>::value
					: output_handler_details::IsRequired<TargetType>::value;
			}
		});
		if(!foundTarget)
			throw std::runtime_error("Cannot find render target '" + std::string(name) + "'");
		return required;
	}

	u32 get_num_channels(StringView targetName) const override{
		bool foundTarget = false;
		u32 numChannels = 0;
		m_targets.for_each([targetName, &foundTarget, &numChannels](auto& target) {
			using Type = std::decay_t<decltype(target)>;
			using TargetType = typename Type::TargetType;
			if(!foundTarget && targetName.compare(TargetType::NAME) == 0) {
				foundTarget = true;
				numChannels = TargetType::NUM_CHANNELS;
			}
		});
		if(!foundTarget)
			throw std::runtime_error("Cannot find render target '" + std::string(targetName) + "'");
		return numChannels;
	}

	int get_current_iteration() const noexcept override { return m_iteration; }

	int get_width() const noexcept override { return m_width; }
	int get_height() const noexcept override { return m_height; }
	int get_num_pixels() const { return m_width * m_height; }
	ei::IVec2 get_resolution() const { return { m_width, m_height }; }

private:
	// In each block either none, m_iter... only, or all three are defined.
	// If variances is required all three will be used and m_iter resets every iteration.
	// Otherwise m_iter contains the cumulative (non-normalized) radiance.
	template < class T >
	struct Target {
		using TargetType = T;
		GenericResource cumulative;
		GenericResource iteration;
		GenericResource cumulativeVariance;
		bool record = false;
		bool recordVariance = false;
	};

	template < Device dev >
	__host__ ArrayDevHandle_t<dev, cuda::Atomic<dev, u32>> get_and_reset_nan_counter() {
		if constexpr(dev == Device::CUDA) {
			return output_handler_details::get_cuda_nan_counter_ptr_and_set_zero();
		} else if constexpr(dev == Device::CPU) {
			m_nanCounter.store(0u);
			return &m_nanCounter;
		} else {
			m_nanCounter.store(0u);
			// TODO: should OpenGL also get a NaN checker?
			return {};
		}
	}

	util::TaggedTuple<Target<Ts>...> m_targets;

	int m_iteration;			// Number of completed iterations / index of current one
	int m_width;
	int m_height;
	cuda::Atomic<Device::CPU, u32> m_nanCounter;	// Counter for CPU-side renderers
};

}} // namespace mufflon::renderer
