
layout(binding = 5) readonly buffer instanceTransformBuffer
{
	float u_instTrans[];
};

mat4x3 getModelMatrix(uint id) {
	uint o = id * 12;
	return mat4x3(
		vec3(u_instTrans[o + 0], u_instTrans[o + 4], u_instTrans[o + 8]),
		vec3(u_instTrans[o + 1], u_instTrans[o + 5], u_instTrans[o + 9]),
		vec3(u_instTrans[o + 2], u_instTrans[o + 6], u_instTrans[o + 10]),
		vec3(u_instTrans[o + 3], u_instTrans[o + 7], u_instTrans[o + 11])
	);
}