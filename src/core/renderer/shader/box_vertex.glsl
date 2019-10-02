layout(location = 0) in vec3 in_position;
layout(location = 1) in int in_level;

layout(location = 0) out flat int out_level;

//layout(location = 1) uniform mat4x3 u_instanceTrans;

void main() {
	// bbox coordinates (min, max) in world space
	gl_Position = vec4(in_position, 1.0);//u_instanceTrans * vec4(in_position, 1.0);
	out_level = in_level;
}