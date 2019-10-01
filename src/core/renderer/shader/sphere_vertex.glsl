layout(location = 0) in vec3 in_position;
layout(location = 1) in float in_radius;
layout(location = 2) in uint in_materialIndex;

layout(location = 1) uniform uint u_instanceId;

// out in camera space
layout(location = 0) out vec3 out_position;
layout(location = 1) out float out_radius;
layout(location = 2) flat out uint out_materialIndex;

void main() {
	mat4x3 model = getModelMatrix(u_instanceId);
	out_position = (u_cam.view * vec4(model * vec4(in_position, 1.0), 1.0)).xyz;
	out_radius = in_radius * model[0][0];
	out_materialIndex = in_materialIndex;
}