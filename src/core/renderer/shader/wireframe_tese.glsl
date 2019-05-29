#version 460
layout(quads, equal_spacing) in;

void main()
{
	vec4 position = mix(
		mix(gl_in[0].gl_Position, gl_in[1].gl_Position, gl_TessCoord.x),
		mix(gl_in[3].gl_Position, gl_in[2].gl_Position, gl_TessCoord.x), 
			gl_TessCoord.y);

	gl_Position = position;
}