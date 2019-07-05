struct SmallLightTransforms
{
	vec3 intesity;
	uint type;
	
	vec3 position;
	float radiusOrFalloff; // radius or cosFalloffStart

	vec3 direction;
	float materialOrTheta; // int material or float cosThetaMax
};

struct BigLightTransforms
{
	vec3 pos;
	uint material;

	vec3 v1;
	uint numPoints;

	vec3 v2;
	float dummy0;

	vec3 v3;
	float dummy1;
};

layout(binding = 2) readonly buffer u_smallLightsBuffer {
	SmallLightTransforms smallLights[];
};

layout(binding = 3) readonly buffer u_bigLightsBuffer {
	BigLightTransforms bigLights[];
};

layout(location = 10) uniform uint numSmallLights;
layout(location = 11) uniform uint numBigLights;
