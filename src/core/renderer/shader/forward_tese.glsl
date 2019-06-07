layout(triangles, equal_spacing) in;

layout(location = 0) in vec3 in_position[];
layout(location = 1) in vec3 in_normal[];
layout(location = 2) in vec2 in_texcoord[];

layout(location = 0) out vec3 out_position;
layout(location = 1) out vec3 out_normal;
layout(location = 2) out vec2 out_texcoord;
layout(location = 3) flat out uint out_materialIndex;

void main()
{
	gl_Position = 
		gl_in[0].gl_Position * gl_TessCoord.x +
		gl_in[1].gl_Position * gl_TessCoord.y +
		gl_in[2].gl_Position * gl_TessCoord.z;

	out_position = 
		in_position[0] * gl_TessCoord.x +
		in_position[1] * gl_TessCoord.y +
		in_position[2] * gl_TessCoord.z;

	out_normal =
		in_normal[0] * gl_TessCoord.x +
		in_normal[1] * gl_TessCoord.y +
		in_normal[2] * gl_TessCoord.z;

	out_texcoord = 
		in_texcoord[0] * gl_TessCoord.x +
		in_texcoord[1] * gl_TessCoord.y +
		in_texcoord[2] * gl_TessCoord.z;

	out_materialIndex = readMaterialShort(gl_PrimitiveID);
}