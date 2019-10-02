layout(location = 0) in vec3 in_position;
layout(location = 1) in vec3 in_normal;
layout(location = 2) in vec2 in_texcoord;
layout(location = 3) flat in uint in_materialIndex;

void main() {
	shade(in_position, normalize(in_normal));
}