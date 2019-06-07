layout(quads, equal_spacing) in;

layout(location = 0) in vec3 in_position[];
layout(location = 1) in vec3 in_normal[];
layout(location = 2) in vec2 in_texcoord[];

layout(location = 0) out vec3 out_position;
layout(location = 1) out vec3 out_normal;
layout(location = 2) out vec2 out_texcoord;
layout(location = 3) flat out uint out_materialIndex;

// offset for material indices
layout(location = 2) uniform uint numTriangles;

void main()
{
	gl_Position = mix(
		mix(gl_in[0].gl_Position, gl_in[1].gl_Position, gl_TessCoord.x),
		mix(gl_in[3].gl_Position, gl_in[2].gl_Position, gl_TessCoord.x), 
			gl_TessCoord.y);

	out_position = mix(
		mix(in_position[0], in_position[1], gl_TessCoord.x),
		mix(in_position[3], in_position[2], gl_TessCoord.x),
		gl_TessCoord.y);

	out_normal = mix(
		mix(in_normal[0], in_normal[1], gl_TessCoord.x),
		mix(in_normal[3], in_normal[2], gl_TessCoord.x),
		gl_TessCoord.y);

	out_texcoord = mix(
		mix(in_texcoord[0], in_texcoord[1], gl_TessCoord.x),
		mix(in_texcoord[3], in_texcoord[2], gl_TessCoord.x),
		gl_TessCoord.y);

	out_materialIndex = readMaterialShort(numTriangles + gl_PrimitiveID);
}