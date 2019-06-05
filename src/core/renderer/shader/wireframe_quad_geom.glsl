layout(triangles) in;
layout(line_strip, max_vertices = 4) out;

layout(location = 0) in int in_index[];

// if the quad is labeled:
// 0 --- 1
// |     |
// 2 --- 3
// => no line should be drawn between 0 and 2 or 3 and 1
// => difference between two vertices should not be two
bool isAllowed(int idx1, int idx2) {
	return abs(in_index[idx1] - in_index[idx2]) != 2;
}

void main() {

	// recreate triangle (but only draw two lines instead of three)
	int prev = 2;
	for(int cur = 0; cur < 3; ++cur) {
		if(isAllowed(prev, cur)) {
			gl_Position = gl_in[prev].gl_Position;
			EmitVertex();

			gl_Position = gl_in[cur].gl_Position;
			EmitVertex();

			EndPrimitive();
		}
		prev = cur;
	}
}