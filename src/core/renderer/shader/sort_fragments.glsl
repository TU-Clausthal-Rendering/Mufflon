layout(location = 0) out vec4 out_fragColor;

layout(binding = 7, std430) readonly buffer ssbo_fragmentBase
{
	uint b_fragmentBase[];
};

struct Fragment
{
	float depth;
	uint color;
};

layout(binding = 8, std430) restrict buffer ssbo_fragmentStore
{
	Fragment b_fragmentData[];
};

layout(location = 0) uniform uint u_screenWidth;

void main()
{
	uint index = uint(gl_FragCoord.y) * u_screenWidth + uint(gl_FragCoord.x);
	uint start = 0;
	if (index > 0)
		start = b_fragmentBase[index - 1];
	uint end = b_fragmentBase[index];

	// something to sort?
	if (start == end)
	{
		out_fragColor = vec4(0.0, 0.0, 0.0, 1.0);
		return;
	}

	// sort b_fragmentData between start and end
	for (uint i = start + 1; i < end; ++i)
	{
		Fragment curFrag = b_fragmentData[i];
		uint j = i;
		for (; j > start && b_fragmentData[j - 1].depth < curFrag.depth; --j)
		{
			b_fragmentData[j] = b_fragmentData[j - 1];
		}
		b_fragmentData[j] = curFrag;
	}

	// the fragment with the highest depth will be the first fragment
	// blend together:
	vec4 color = unpackUnorm4x8(b_fragmentData[start].color);
	color.rgb *= color.a;
	// background occlusion
	color.a = (1.0 - color.a);

	for (uint i = start + 1; i < end; ++i)
	{
		vec4 next = unpackUnorm4x8(b_fragmentData[i].color);
		// fragment color
		color.rgb = next.a * next.rgb + (1.0 - next.a) * color.rgb;
		// background occlusion
		color.a *= (1.0 - next.a);
	}

	// blending GL_ONE, GL_SRC_ALPHA
	out_fragColor = color;
	//out_fragColor = vec4(0.0, 0.0, 0.0, 1.0);
}