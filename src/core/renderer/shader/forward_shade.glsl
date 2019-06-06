#define FORWARD_SHADE
layout(location = 0) out vec4 out_fragColor;
layout(location = 1) out vec3 out_position;
layout(location = 2) out vec3 out_albedo;
layout(location = 3) out vec3 out_normal;
layout(location = 4) out vec3 out_lightness;

// #define MAX_MATERIAL_DESCRIPTOR_SIZE
layout(binding = 0) buffer materialsBuffer {
	uint u_materialData[];
};

uint readShort(uint byteOffset) {
	uint index = byteOffset / 4;
	uint remainder = byteOffset % 4;
	// assume remainder is either 0 or 2
	if(remainder == 0) return u_materialData[index] >> 16;
	return u_materialData[index] & 0xFFFF;
}

void shade(vec3 pos, vec3 normal, vec2 texcoord, int materialIndex) {
	out_normal = normal;
	out_position = pos;

	// read correct material
	uint matOffset = materialIndex * MAX_MATERIAL_DESCRIPTOR_SIZE;
	uint matType = readShort(matOffset);

	//out_fragColor = vec4(normal * 0.5 + vec3(0.5), 1.0);
	out_fragColor = vec4(float(matType) / 10.0f + 0.1f);
	//out_fragColor = vec4(fract(texcoord), 1.0, 1.0);

	// TODO
	out_albedo = vec3(1.0f);
	out_lightness = vec3(0.0f);
}