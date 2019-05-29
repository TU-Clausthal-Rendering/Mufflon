#version 460
layout(location = 0) in vec3 in_position;

layout(location = 0) uniform mat4 u_viewProj;
layout(location = 1) uniform mat4x3 u_instanceTrans;

void main() {
	gl_Position = u_viewProj * vec4(u_instanceTrans * vec4(in_position, 1.0), 1.0);
}