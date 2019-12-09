#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <string_view>

namespace mufflon {

#include "core_interface.h"
#include "mff_interface.h"

} // namespace mufflon

#define CHECK_ERROR(expr)											\
	if(!(expr))														\
		throw std::runtime_error("Expression failed: '" #expr "'")
	
using namespace mufflon;
	
void enable_renderer(const std::string_view name, RenderDevice device) {
	const std::uint32_t count = render_get_renderer_count();
	for(std::uint32_t i = 0; i < count; ++i) {
		const std::string_view currName = render_get_renderer_name(i);
		const std::string_view currShortName = render_get_renderer_short_name(i);
		if(name == currName || name == currShortName) {
			const std::uint32_t variations = render_get_renderer_variations(i);
			for(std::uint32_t v = 0; v < variations; ++v)
				if(device == render_get_renderer_devices(i, v)) {
					render_enable_renderer(i, v);
					return;
				}
		}
	}
	throw std::runtime_error("Could not find renderer '" + std::string(name)
							 + "' with given device");
}


int main(int argc, const char* argv[]) {
	if(argc < 5) {
		printf("Usage: %s renderer rendertarget scenefile iterations\n", argv[0]);
		return EXIT_FAILURE;
	}
	
	try {
		const std::string renderer = argv[1];
		const std::string target = argv[2];
		const std::filesystem::path filepath = argv[3];
		const auto filename = filepath.stem();
		const long iterations = std::strtol(argv[4], NULL, 10);

		CHECK_ERROR(mufflon_initialize());
		
		if(loader_load_json(filepath.string().c_str()) != LoaderStatus::LOADER_SUCCESS) {
			fprintf(stderr, "Failed to load scene file '%s'\n", filepath.string().c_str());
			return EXIT_FAILURE;
		}
		
		enable_renderer(renderer, RenderDevice::DEVICE_CPU);
		CHECK_ERROR(render_enable_render_target(target.c_str(), false));
		
		for(long i = 0; i < iterations; ++i)
			CHECK_ERROR(render_iterate(NULL));
		
		const auto outputName = renderer + "-" + target + "-" + filename.string() + ".pfm";
		CHECK_ERROR(render_save_screenshot(outputName.c_str(), target.c_str(), false));
		mufflon_destroy();
	} catch(const std::exception& e) {
		fprintf(stderr, "Exception: '%s'\n", e.what());
		return EXIT_FAILURE;
	}
	return EXIT_SUCCESS;
}
