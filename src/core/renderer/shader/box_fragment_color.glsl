layout(early_fragment_tests) in;
layout(location = 0) out vec4 out_fragColor;
layout(location = 0) in vec2 texcoords;
layout(location = 1) in flat int in_level;

layout(location = 10) uniform int maxLevel;
layout(location = 11) uniform int highlightLevel;

layout(binding = 6, std430) coherent buffer ssbo_fragmentCount
{
	uint b_fragmentCount[];
};

layout(binding = 7, std430) readonly buffer ssbo_fragmentBase
{
	uint b_fragmentBase[];
};

struct Fragment
{
	float depth;
	uint color;
};

layout(binding = 8, std430) writeonly buffer ssbo_fragmentStore
{
	Fragment b_fragmentDest[];
};

layout(location = 2) uniform vec3 color;

void main() {
	// opacient gradient based on distance to edges (edges have texoords of 1 or 0)
	// transform to [0, 1.0] where 1.0 = border
	float opacity = max(abs(texcoords.x - 0.5), abs(texcoords.y - 0.5)) * 2.0;
	opacity = pow(opacity, 10.0);

	vec4 color = vec4(color, opacity * 0.5);

#ifdef SHADE_LEVEL
	// level gradient
	//float levelFactor = float(in_level) / float(maxLevel);
	//levelFactor = 1.0 - levelFactor;
	
	//color.rgb *= levelFactor;
	if (in_level == highlightLevel)
	{
		color.rgb = vec3(1.0, 0.5, 0.0);
	}
#endif

	// store color etc.
	uint index = uint(gl_FragCoord.y) * uint(u_cam.screen.x) + uint(gl_FragCoord.x);
	uint offset = atomicAdd(b_fragmentCount[index], uint(-1)) - 1;
	uint base = 0;
	if (index > 0)
		base = b_fragmentBase[index - 1];

	// store
	uint storeIdx = base + offset;

	b_fragmentDest[storeIdx].depth = gl_FragCoord.z;
	b_fragmentDest[storeIdx].color = packUnorm4x8(color);

	out_fragColor = vec4(1.0);
}