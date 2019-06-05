layout(quads, equal_spacing) in;

layout(location = 0) out int out_index;

void main()
{
	vec4 position = mix(
		mix(gl_in[0].gl_Position, gl_in[1].gl_Position, gl_TessCoord.x),
		mix(gl_in[3].gl_Position, gl_in[2].gl_Position, gl_TessCoord.x), 
			gl_TessCoord.y);

	if(gl_TessCoord.x > 0.0f) {
		if(gl_TessCoord.y > 0.0f)
			out_index = 2;
		else
			out_index = 1;
	} else {
		if(gl_TessCoord.y > 0.0f)
			out_index = 3;
		else
			out_index = 0;
	}


	gl_Position = position;
}