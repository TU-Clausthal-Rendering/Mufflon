#define FORWARD_SHADE

layout(location = 0) out vec4 out_fragColor;
layout(location = 1) out vec3 out_position;
layout(location = 2) out vec3 out_albedo;
layout(location = 3) out vec3 out_normal;
layout(location = 4) out vec3 out_lightness;

struct ColorInfo
{
	vec3 color;
	vec3 albedo;
	vec3 light;
};

struct LightInfo
{
	vec3 color; // color multiplied with attenuation
	vec3 direction; // outgoing from material, normalized
};

// light travel direction
LightInfo calcDirLight(vec3 direction, vec3 radiance) {
	LightInfo i;
	i.color = radiance;
	i.direction = normalize(-direction);
	return i;
}

LightInfo calcPointLight(vec3 pos, vec3 lightPos, vec3 radiance) {
	LightInfo i;
	i.direction = lightPos - pos;
	float dist = length(i.direction);
	i.direction /= dist;
	i.color = radiance / (dist * dist);
	//i.color = vec3(1.0);
	return i;
}



void calcRadiance(inout ColorInfo c, vec3 pos, vec3 normal, vec3 albedo, vec3 emission) {
	c.color += emission;
	c.light += emission;
	c.albedo += albedo;

	LightInfo light;

	// iterate thought small lights
	for(uint i = 0; i < numSmallLights; ++i) {
		switch(smallLights[i].type) {
		case LIGHT_TYPE_POINT:
			light = calcPointLight(pos, smallLights[i].position, smallLights[i].intensity);
			break;
		case LIGHT_TYPE_SPOT:
			continue;
			break;
		case LIGHT_TYPE_SPHERE:
			continue;
			break;
		case LIGHT_TYPE_DIRECTIONAL:
			light = calcDirLight(smallLights[i].direction, smallLights[i].intensity);
			break;
		default: continue;
		}

		// add to material color
		// TODO use microfacet for roughness?
		float cosTheta = max(dot(light.direction, normal), 0.0);
		c.color += albedo * light.color * cosTheta;
		c.light += light.color * cosTheta;
	}

	for(uint i = 0; i < numBigLights; ++i) {

	}

	// TODO iterate thought big lights
	//c.color = vec3(float(numSmallLights), float(numSmallLights) * 0.1f, float(numSmallLights) * 0.01f);
	//c.color = c.light;
}

#define EMISSIVE 0u
#define LAMBERT 1u
#define ORENNAYAR 2u				
#define TORRANCE 3u				
#define WALTER 4u			
#define LAMBERT_EMISSIVE 5u
#define TORRANCE_LAMBERT 6u			
#define FRESNEL_TORRANCE_LAMBERT 7u
#define WALTER_TORRANCE 8u
#define FRESNEL_TORRANCE_WALTER 9u
#define MICROFACET 10u	

layout(binding = 0) readonly buffer materialsBuffer {
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

void shadeEmissive(inout ColorInfo c, vec3 pos, vec3 normal, vec3 uv, inout uint offset) {
	sampler2DArray emissionTex = sampler2DArray(readTexHdl(offset));
	offset += 8;
	calcRadiance(c, pos, normal, vec3(0.0), texture(emissionTex, uv).rgb);
}

void shadeLambert(inout ColorInfo c, vec3 pos, vec3 normal, vec3 uv, inout uint offset) {
	sampler2DArray albedoTex = sampler2DArray(readTexHdl(offset));
	offset += 8;
	calcRadiance(c, pos, normal, texture(albedoTex, uv).rgb, vec3(0.0));
}

void shadeOrennayar(inout ColorInfo c, vec3 pos, vec3 normal, vec3 uv, inout uint offset) {
	sampler2DArray albedoTex = sampler2DArray(readTexHdl(offset));
	offset += 8;
	calcRadiance(c, pos, normal, texture(albedoTex, uv).rgb, vec3(0.0));
}

void shadeTorrance(inout ColorInfo c, vec3 pos, vec3 normal, vec3 uv, inout uint offset) {
	sampler2DArray albedoTex = sampler2DArray(readTexHdl(offset));
	//sampler2DArray roughnessTex = sampler2DArray(readTexHdl(offset + 8));
	offset += 16;
	calcRadiance(c, pos, normal, texture(albedoTex, uv).rgb, vec3(0.0));
}

void shadeWalter(inout ColorInfo c, vec3 pos, vec3 normal, vec3 uv, inout uint offset) {
	//sampler2DArray roughnessTex = sampler2DArray(readTexHdl(offset));
	offset += 8;
	calcRadiance(c, pos, normal, vec3(0.0), vec3(0.0));
}

void shadeMicrofacet(inout ColorInfo c, vec3 pos, vec3 normal, vec3 uv, inout uint offset) {
	//sampler2DArray roughnessTex = sampler2DArray(readTexHdl(offset));
	offset += 8;
	calcRadiance(c, pos, normal, vec3(0.0), vec3(0.0));
}

void shade(vec3 pos, vec3 normal, vec2 texcoord, uint materialIndex) {
	out_normal = normal;
	out_position = pos;
	const vec3 uv = vec3(texcoord, 0.0); // all textures are sampler2DArrays with layer 0

	// read correct material
	uint matOffset = u_materialData[materialIndex];
	uint matType = readShort(matOffset);
	uint matFlags = readShort(matOffset + 2);

	ColorInfo c;
	c.color = vec3(0.0);
	c.albedo = vec3(0.0);
	c.light = vec3(0.0);
	// next is medium handle => until (matOffset + 8)
	matOffset += 8;

	switch(matType) {
	case EMISSIVE:
		shadeEmissive(c, pos, normal, uv, matOffset);
		break;
	case ORENNAYAR:
		shadeOrennayar(c, pos, normal, uv, matOffset);
		break;
	case LAMBERT:
		shadeLambert(c, pos, normal, uv, matOffset);
		break;
	case TORRANCE:
		shadeTorrance(c, pos, normal, uv, matOffset);
		break;
	case WALTER: 
		shadeWalter(c, pos, normal, uv, matOffset);
		break;
	case MICROFACET:
		shadeMicrofacet(c, pos, normal, uv, matOffset);
		break;
	case LAMBERT_EMISSIVE: // TODO blend correctly
		shadeLambert(c, pos, normal, uv, matOffset);
		shadeEmissive(c, pos, normal, uv, matOffset);
		break;
	case TORRANCE_LAMBERT:
		shadeTorrance(c, pos, normal, uv, matOffset);
		shadeLambert(c, pos, normal, uv, matOffset);
		break;
	case FRESNEL_TORRANCE_LAMBERT:
		shadeTorrance(c, pos, normal, uv, matOffset);
		shadeLambert(c, pos, normal, uv, matOffset);
		break;
	case WALTER_TORRANCE:
		shadeWalter(c, pos, normal, uv, matOffset);
		shadeTorrance(c, pos, normal, uv, matOffset);
		break;
	case FRESNEL_TORRANCE_WALTER:
		shadeTorrance(c, pos, normal, uv, matOffset);
		shadeWalter(c, pos, normal, uv, matOffset);
		break;
	}

	out_fragColor = vec4(c.color, 1.0);
	out_albedo = c.albedo;
	out_lightness = c.light;
}