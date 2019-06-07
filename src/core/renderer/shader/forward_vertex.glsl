layout(location = 0) in vec3 in_position;
layout(location = 1) in vec3 in_normal;
layout(location = 2) in vec2 in_texcoord;

layout(location = 0) out vec3 out_position;
layout(location = 1) out vec3 out_normal;
layout(location = 2) out vec2 out_texcoord;

layout(location = 1) uniform mat4x3 u_instanceTrans;

void main() {
	out_position = u_instanceTrans * vec4(in_position, 1.0);
	out_normal = mat3(u_instanceTrans) * in_normal;
	out_texcoord = in_texcoord;
	gl_Position = u_cam.viewProj * vec4(out_position, 1.0);
}