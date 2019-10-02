layout(location = 0) out vec2 texCoords;

void main(void)
{
	vec4 vertex = vec4(0.0, 0.0, 0.0, 1.0);
	if (gl_VertexID == 0u) vertex = vec4(1.0, -1.0, 0.0, 1.0);
	if (gl_VertexID == 1u) vertex = vec4(-1.0, -1.0, 0.0, 1.0);
	if (gl_VertexID == 2u) vertex = vec4(1.0, 1.0, 0.0, 1.0);
	if (gl_VertexID == 3u) vertex = vec4(-1.0, 1.0, 0.0, 1.0);
	gl_Position = vertex;

	texCoords = vertex.xy * vec2(0.5) + vec2(0.5);
}