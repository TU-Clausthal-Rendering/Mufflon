#include "core_interface.h"
#include "mff_interface.h"
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK_ERROR(expr)										\
	if(!(expr)) {												\
		fprintf(stderr, "Expression failed: '%s'\n", #expr);	\
		return EXIT_FAILURE;									\
	}
	
bool enable_renderer(const char* name, RenderDevice device) {
	const uint32_t count = render_get_renderer_count();
	for(uint32_t i = 0; i < count; ++i) {
		const char* currName = render_get_renderer_name(i);
		const char* currShortName = render_get_renderer_short_name(i);
		if(strcmp(name, currName) == 0 || strcmp(name, currShortName) == 0) {
			const uint32_t variations = render_get_renderer_variations(i);
			for(uint32_t v = 0; v < variations; ++v)
				if(device == render_get_renderer_devices(i, v))
					return render_enable_renderer(i, v);
		}
	}
	return false;
}


int main(int argc, const char* argv[]) {
	if(argc < 5) {
		printf("Usage: %s renderer rendertarget scenefile iterations\n", argv[0]);
		return EXIT_FAILURE;
	}
		
	const char* renderer = argv[1];
	const char* target = argv[2];
	const char* filepath = argv[3];
	const char* filename = strrchr(filepath, '/');
	if(filename == NULL)
		filename = filepath;
	else
		filename = filename + 1;
	const long iterations = strtol(argv[4], NULL, 10);

	CHECK_ERROR(mufflon_initialize());
	
	if(loader_load_json(filepath) != LOADER_SUCCESS) {
		fprintf(stderr, "Failed to load scene file '%s'\n", filepath);
		return EXIT_FAILURE;
	}
	
	if(!enable_renderer(renderer, DEVICE_CPU)) {
		fprintf(stderr, "Could not find renderer with the name '%s'\n", renderer);
		return EXIT_FAILURE;
	}
	CHECK_ERROR(render_enable_render_target(target, false));
	
	for(long i = 0; i < iterations; ++i)
		CHECK_ERROR(render_iterate(NULL));
	
	const size_t bufferLen = strlen(renderer) + strlen(target) + strlen(filename) + 7u;
	char* buffer = (char*)malloc(bufferLen);
	sprintf(buffer, "%s-%s-%s.pfm", renderer, target, filename);
	CHECK_ERROR(render_save_screenshot(buffer, target, false));
	free(buffer);
	mufflon_destroy();
	return EXIT_SUCCESS;
}
