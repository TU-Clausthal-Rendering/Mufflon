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
	
bool enable_renderer(MufflonInstanceHdl mffInst, const char* name, RenderDevice device) {
	const uint32_t count = render_get_renderer_count(mffInst);
	for(uint32_t i = 0; i < count; ++i) {
		const char* currName = render_get_renderer_name(mffInst, i);
		const char* currShortName = render_get_renderer_short_name(mffInst, i);
		if(strcmp(name, currName) == 0 || strcmp(name, currShortName) == 0) {
			const uint32_t variations = render_get_renderer_variations(mffInst, i);
			for(uint32_t v = 0; v < variations; ++v)
				if(device == render_get_renderer_devices(mffInst, i, v))
					return render_enable_renderer(mffInst, i, v);
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
	
	MufflonInstanceHdl mffInst;
	CHECK_ERROR(mffInst = mufflon_initialize());
	
	MufflonLoaderInstanceHdl mffLoaderInst;
	CHECK_ERROR(mffLoaderInst = loader_initialize(mffInst));
	if(loader_load_json(mffLoaderInst, filepath) != LOADER_SUCCESS) {
		fprintf(stderr, "Failed to load scene file '%s'\n", filepath);
		return EXIT_FAILURE;
	}
	
	if(!enable_renderer(mffInst, renderer, DEVICE_CPU)) {
		fprintf(stderr, "Could not find renderer with the name '%s'\n", renderer);
		return EXIT_FAILURE;
	}
	CHECK_ERROR(render_enable_render_target(mffInst, target, false));
	
	for(long i = 0; i < iterations; ++i)
		CHECK_ERROR(render_iterate(mffInst, NULL));
	
	const size_t bufferLen = strlen(renderer) + strlen(target) + strlen(filename) + 7u;
	char* buffer = (char*)malloc(bufferLen);
	sprintf(buffer, "%s-%s-%s.pfm", renderer, target, filename);
	CHECK_ERROR(render_save_screenshot(mffInst, buffer, target, false));
	free(buffer);
	loader_destroy(mffLoaderInst);
	mufflon_destroy(mffInst);
	return EXIT_SUCCESS;
}
