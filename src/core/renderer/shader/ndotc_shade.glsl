#define NDOTC_SHADE
layout(location = 0) out vec3 out_color;

void shade(vec3 position, vec3 normal) {
	out_color = vec3(dot(normal, normalize(u_cam.position - position)));
}