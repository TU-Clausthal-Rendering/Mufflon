#define FORWARD_SHADE

layout(location = 0) out vec4 out_fragColor;
layout(location = 1) out vec3 out_position;
layout(location = 2) out vec3 out_albedo;
layout(location = 3) out vec3 out_normal;
layout(location = 4) out vec3 out_lightness;

#define EMISSIVE 0
#define LAMBERT 1
#define ORENNAYAR 2				
#define TORRANCE 3				
#define WALTER 4				
#define LAMBERT_EMISSIVE 5
#define TORRANCE_LAMBERT 6			
#define FRESNEL_TORRANCE_LAMBERT 7
#define WALTER_TORRANCE 8
#define FRESNEL_TORRANCE_WALTER 9
#define MICROFACET 10	

layout(binding = 0) buffer materialsBuffer {
	uint u_materialData[];
};

uint readShort(uint byteOffset) {
	uint index = byteOffset / 4;
	uint remainder = byteOffset % 4;
	// assume remainder is either 0 or 2
	if(remainder != 0) return u_materialData[index] >> 16;
	return u_materialData[index] & 0xFFFF;
}

uvec2 readTexHdl(uint byteOffset) {
	uint index = byteOffset / 4;
	return uvec2(u_materialData[index], u_materialData[index + 1]);
}

void shade(vec3 pos, vec3 normal, vec2 texcoord, int materialIndex) {
	out_normal = normal;
	out_position = pos;
	const vec3 uv = vec3(texcoord, 0.0); // all textures are sampler2DArrays with layer 0

	// read correct material
	uint matOffset = u_materialData[materialIndex];
	uint matType = readShort(matOffset);
	uint matFlags = readShort(matOffset + 2);

	vec3 color = vec3(0.0);
	// next is medium handle => until (matOffset + 8)
	switch(matType) {
	case EMISSIVE: {
		sampler2DArray emissionTex = sampler2DArray(readTexHdl(matOffset + 8));
		color = texture(emissionTex, uv).rgb;
	} break;
	case ORENNAYAR:
	case LAMBERT: {
		sampler2DArray albedoTex = sampler2DArray(readTexHdl(matOffset + 8));
		color = texture(albedoTex, uv).rgb;
		out_albedo = color;
	} break;
	case TORRANCE: {
		sampler2DArray albedoTex = sampler2DArray(readTexHdl(matOffset + 8));
		sampler2DArray roughnessTex = sampler2DArray(readTexHdl(matOffset + 16));
		out_albedo = texture(albedoTex, uv).rgb;
		color = texture(roughnessTex, uv).rgb;
	} break;
	case WALTER: {
		sampler2DArray roughnessTex = sampler2DArray(readTexHdl(matOffset + 8));
		color = texture(roughnessTex, uv).rgb;
	} break;
	}



	//out_fragColor = vec4(normal * 0.5 + vec3(0.5), 1.0);
	//out_fragColor = vec4(fract(texcoord), 1.0, 1.0);

	out_fragColor = vec4(color, 1.0);

	// TODO
	out_lightness = vec3(0.0f);
}