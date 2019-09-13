#define FORWARD_SHADE

layout(location = 2) out vec3 out_fragColor;
layout(location = 1) out vec3 out_position;
layout(location = 0) out vec3 out_normal;

layout(bindless_sampler, location = 24) uniform sampler2DArray ltc_mat_tex;
layout(bindless_sampler, location = 26) uniform sampler2DArray ltc_fres_tex;

struct LightInfo
{
	vec3 color; // color multiplied with attenuation
	vec3 direction; // outgoing from material, normalized
};

struct MaterialInfo
{
	vec3 diffuse;
	vec3 emission;
	vec3 specular;
	float roughness;
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

vec3 getMaterialEmission(uint materialId);

vec3 calcRadiance(vec3 pos, vec3 normal, MaterialInfo mat) {
	const vec3 view = normalize(u_cam.position - pos);

	vec3 color = vec3(0.0);
	color += mat.emission;

	LightInfo light;

	// iterate thought small lights
	for(uint i = 0; i < numSmallLights; ++i) {
		switch(smallLights[i].type) {
		case LIGHT_TYPE_POINT:
			light = calcPointLight(pos, smallLights[i].position, smallLights[i].intensity);
			break;
		case LIGHT_TYPE_SPOT:
			// TODO add spot light properties
			light = calcPointLight(pos, smallLights[i].position, smallLights[i].intensity);
			break;
		case LIGHT_TYPE_SPHERE: {
			// project closest point
			vec3 lightPos = smallLights[i].position + normalize(pos - smallLights[i].position) * smallLights[i].radiusOrFalloff;
			light = calcPointLight(pos, lightPos, getMaterialEmission(floatBitsToUint(smallLights[i].materialOrTheta)));
		} break;
		case LIGHT_TYPE_DIRECTIONAL:
			light = calcDirLight(smallLights[i].direction, smallLights[i].intensity);
			break;
		default: continue;
		}

		// add to material color
		// TODO use microfacet for roughness?
		float cosTheta = max(dot(light.direction, normal), 0.0);
		color += mat.diffuse * light.color * cosTheta;


	}

	// angle between normal and view
	float NdotV = max(dot(normal, view), 0.0);

	for(uint i = 0; i < numBigLights; ++i) {
		vec3 points[4];
		points[0] = bigLights[i].pos;
		points[1] = bigLights[i].pos + bigLights[i].v1;
		points[2] = bigLights[i].pos + bigLights[i].v2;
		points[3] = bigLights[i].pos + bigLights[i].v3;

		vec3 lightColor = getMaterialEmission(bigLights[i].material);

		// diffuse part
		{
			vec3 luminance = LTC_Evaluate(normal, view, pos, mat3(1.0), points, int(bigLights[i].numPoints), ltc_fres_tex);
			vec3 lightness = luminance * lightColor; //* 0.159154943; // normalize with 1 / (2 * pi)
			color += mat.diffuse * lightness;
		}
		
		// specular part
		{
			if (mat.specular == vec3(0.0)) continue;

			vec2 texCoords = LTC_Coords(NdotV, mat.roughness);
			mat3 Minv = LTC_Matrix(ltc_mat_tex, texCoords);
			vec3 luminance = LTC_Evaluate(normal, view, pos, Minv, points, int(bigLights[i].numPoints), ltc_fres_tex);
			vec3 lightness = luminance * lightColor;// *0.159154943; // normalize with 1 / (2 * pi)
			// BRDF shadowing and Fresnel
			vec2 fresnel = textureLod(ltc_fres_tex, vec3(texCoords, 0.0), 0.0).rg;
			color += lightness * (fresnel.x * mat.specular + (1.0 - mat.specular) * fresnel.y);
		}
	}

	return color;
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

float readFloat(uint byteOffset)
{
	uint index = byteOffset / 4;
	return uintBitsToFloat(u_materialData[index]);
}

MaterialInfo getEmissive(vec3 pos, vec3 normal, vec3 uv, inout uint offset) {
	sampler2DArray emissionTex = sampler2DArray(readTexHdl(offset));
	offset += 8;
	return MaterialInfo(vec3(0.0), texture(emissionTex, uv).rgb, vec3(0.0), 0.0);
}

void getEmissiveParams(inout uint offset, out vec3 scale) {
	scale = vec3(readFloat(offset), readFloat(offset + 4), readFloat(offset + 8));
	offset += 12;
}

MaterialInfo getLambert(vec3 pos, vec3 normal, vec3 uv, inout uint offset) {
	sampler2DArray albedoTex = sampler2DArray(readTexHdl(offset));
	offset += 8;
	return MaterialInfo(texture(albedoTex, uv).rgb, vec3(0.0), vec3(0.0), 0.0);
}

MaterialInfo getOrennayar(vec3 pos, vec3 normal, vec3 uv, inout uint offset) {
	sampler2DArray albedoTex = sampler2DArray(readTexHdl(offset));
	offset += 8;
	return MaterialInfo(texture(albedoTex, uv).rgb, vec3(0.0), vec3(0.0), 0.0);
}

void getOrennayarParams(inout uint offset, out float a, out float b)
{
	a = readFloat(offset);
	offset += 4;
	b = readFloat(offset);
	offset += 4;
}

MaterialInfo getTorrance(vec3 pos, vec3 normal, vec3 uv, inout uint offset) {
	sampler2DArray albedoTex = sampler2DArray(readTexHdl(offset)); // specular
	sampler2DArray roughnessTex = sampler2DArray(readTexHdl(offset + 8));
	offset += 16;
	return MaterialInfo(vec3(0.0), vec3(0.0), texture(albedoTex, uv).rgb, texture(roughnessTex, uv).r);
}

void getTorranceParams(inout uint offset, out uint shadowing, out uint ndf) {
	shadowing = readShort(offset);
	offset += 2;
	ndf = readShort(offset);
	offset += 2;
}

MaterialInfo getWalter(vec3 pos, vec3 normal, vec3 uv, inout uint offset) {
	sampler2DArray roughnessTex = sampler2DArray(readTexHdl(offset));
	offset += 8;
	return MaterialInfo(vec3(0.0), vec3(0.0), vec3(1.0), texture(roughnessTex, uv).r);
}

void getWalterParams(inout uint offset, out vec3 absorption, out uint shadowing, out uint ndf) {
	absorption = vec3(readFloat(offset), readFloat(offset + 4), readFloat(offset + 8));
	offset += 12;
	shadowing = readShort(offset);
	offset += 2;
	ndf = readShort(offset);
	offset += 2;
}

MaterialInfo getMicrofacet(vec3 pos, vec3 normal, vec3 uv, inout uint offset) {
	sampler2DArray roughnessTex = sampler2DArray(readTexHdl(offset));
	offset += 8;
	return MaterialInfo(vec3(0.0), vec3(0.0), vec3(1.0), texture(roughnessTex, uv).r);
}

void getMicrofacetParams(inout uint offset, out vec3 absorption, out uint shadowing, out uint ndf) {
	absorption = vec3(readFloat(offset), readFloat(offset + 4), readFloat(offset + 8));
	offset += 12;
	shadowing = readShort(offset);
	offset += 2;
	ndf = readShort(offset);
	offset += 2;
}

void getBlendParams(inout uint offset, out float factor1, out float factor2) {
	factor1 = readFloat(offset);
	factor2 = readFloat(offset + 4);
	offset += 8;
}

MaterialInfo blendMaterial(MaterialInfo mat1, MaterialInfo mate2, float factor1, float factor2)
{
	return MaterialInfo(
		mat1.diffuse * factor1 + mate2.diffuse * factor2,
		mat1.emission * factor1 + mate2.emission * factor2,
		mat1.specular * factor1 + mate2.specular * factor2,
		mat1.roughness * factor1 + mate2.roughness * factor2
	);
}

void shade(vec3 pos, vec3 normal, vec2 texcoord, uint materialIndex) {
	out_normal = normal;
	out_position = pos;
	const vec3 uv = vec3(texcoord, 0.0); // all textures are sampler2DArrays with layer 0

	// read correct material
	uint matOffset = u_materialData[materialIndex];
	uint matType = readShort(matOffset);
	uint matFlags = readShort(matOffset + 2);
	// next is medium handle => until (matOffset + 8)
	matOffset += 8;

	MaterialInfo mat;

	out_fragColor = vec3(0.0, 0.0, 0.0);


	switch (matType) {
	case EMISSIVE: {
		mat = getEmissive(pos, normal, uv, matOffset);
		vec3 scale;
		getEmissiveParams(matOffset, scale);
		mat.emission *= scale;
	} break;
	case ORENNAYAR: {
		mat = getOrennayar(pos, normal, uv, matOffset);
		float a, b;
		getOrennayarParams(matOffset, a, b);
	} break;
	case LAMBERT:
		mat = getLambert(pos, normal, uv, matOffset);
		break;
	case TORRANCE: {
		mat = getTorrance(pos, normal, uv, matOffset);
		uint shadowing, ndf;
		getTorranceParams(matOffset, shadowing, ndf);
	} break;
	case WALTER: {
		mat = getWalter(pos, normal, uv, matOffset);
		vec3 absorption; uint shadowing, ndf;
		getWalterParams(matOffset, absorption, shadowing, ndf);
	} break;
	case MICROFACET: {
		mat = getMicrofacet(pos, normal, uv, matOffset);
		vec3 absorption; uint shadowing, ndf;
		getMicrofacetParams(matOffset, absorption, shadowing, ndf);
	} break;
	case LAMBERT_EMISSIVE: {
		// LayerA
		MaterialInfo m1 = getLambert(pos, normal, uv, matOffset);
		// LayerB
		MaterialInfo m2 = getEmissive(pos, normal, uv, matOffset);
		// LayerB params
		vec3 scale;
		getEmissiveParams(matOffset, scale);
		m2.emission *= scale;
		// Blend
		float factor1, factor2;
		getBlendParams(matOffset, factor1, factor2);
		mat = blendMaterial(m1, m2, factor1, factor2);
	} break;
	case TORRANCE_LAMBERT: {
		// LayerA
		MaterialInfo m1 = getTorrance(pos, normal, uv, matOffset);
		// LayerB
		MaterialInfo m2 = getLambert(pos, normal, uv, matOffset);
		// LayerA params
		uint shadowing, ndf;
		getTorranceParams(matOffset, shadowing, ndf);
		// Blend
		float factor1, factor2;
		getBlendParams(matOffset, factor1, factor2);
		mat = blendMaterial(m1, m2, factor1, factor2);
	} break;
	case FRESNEL_TORRANCE_LAMBERT: {
		// LayerA
		MaterialInfo m1 = getTorrance(pos, normal, uv, matOffset);
		// LayerB
		MaterialInfo m2 = getLambert(pos, normal, uv, matOffset);
		// LayerA params
		uint shadowing, ndf;
		getTorranceParams(matOffset, shadowing, ndf);

		mat = blendMaterial(m1, m2, 1.0, 1.0);
	} break;
	case WALTER_TORRANCE: {
		// LayerA
		MaterialInfo m1 = getWalter(pos, normal, uv, matOffset);
		// LayerB
		MaterialInfo m2 = getTorrance(pos, normal, uv, matOffset);
		// LayerA params
		vec3 absorption; uint shadowing, ndf;
		getWalterParams(matOffset, absorption, shadowing, ndf);
		// LayerB params
		//uint shadowing, ndf;
		getTorranceParams(matOffset, shadowing, ndf);
		// Blend
		float factor1, factor2;
		getBlendParams(matOffset, factor1, factor2);
		mat = blendMaterial(m1, m2, factor1, factor2);
	} break;
	case FRESNEL_TORRANCE_WALTER: {
		// LayerA
		MaterialInfo m1 = getTorrance(pos, normal, uv, matOffset);
		// LayerB
		MaterialInfo m2 = getWalter(pos, normal, uv, matOffset);
		// LayerA params
		uint shadowing, ndf;
		getTorranceParams(matOffset, shadowing, ndf);
		// LayerB params
		vec3 absorption; //uint shadowing, ndf;
		getWalterParams(matOffset, absorption, shadowing, ndf);

		mat = blendMaterial(m1, m2, 1.0, 1.0);
	} break;
	}

	out_fragColor = calcRadiance(pos, normal, mat);
}

vec3 getMaterialEmission(uint materialIndex) {
	// read correct material
	uint matOffset = u_materialData[materialIndex];
	uint matType = readShort(matOffset);
	uint matFlags = readShort(matOffset + 2);

	// next is medium handle => until (matOffset + 8)
	matOffset += 8;
	switch(matType) {
	case EMISSIVE: {
		sampler2DArray emissionTex = sampler2DArray(readTexHdl(matOffset));
		return texture(emissionTex, vec3(0.0)).rgb;
	} break;
	case LAMBERT_EMISSIVE: {
		// TODO correct blending
		sampler2DArray emissionTex = sampler2DArray(readTexHdl(matOffset + 8));
		return texture(emissionTex, vec3(0.0)).rgb;
	} break;
	}
	return vec3(0.0);
}