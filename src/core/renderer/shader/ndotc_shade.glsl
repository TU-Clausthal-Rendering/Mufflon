#define NDOTC_SHADE
layout(location = 0) out vec3 out_color;

#if COLOR_INSTANCE
layout(location = 1) uniform uint u_instanceId;
#endif

void shade(vec3 position, vec3 normal) {
	out_color = vec3(abs(dot(normal, normalize(u_cam.position - position))));
	
#if COLOR_INSTANCE
	out_color.z = float(u_instanceId);
#endif
}