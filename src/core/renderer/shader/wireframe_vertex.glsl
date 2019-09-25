layout(location = 0) in vec3 in_position;

layout(location = 1) uniform uint u_instanceId;

void main() {
	gl_Position = u_cam.viewProj * vec4(getModelMatrix(u_instanceId) * vec4(in_position, 1.0), 1.0);
}