layout(early_fragment_tests) in;
layout(location = 0) out vec4 out_fragColor;

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

void main() {
	vec4 color = vec4(1.0, 0.0, 0.0, 0.1);

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