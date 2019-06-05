layout(location = 0) in vec2 in_location;
layout(location = 1) flat in float in_radius;
layout(location = 2) in vec3 in_position;

layout(binding = 0) uniform u_camTrans
{
	CameraTransforms u_cam;
};

// convert view space to world space
vec3 toWorld(vec4 viewVec) {
	vec4 res = u_cam.invView * viewVec;
	//res.xyz /= res.w;
	return res.xyz;
}

void main() {
	float curRadius = length(in_location);
	if(curRadius > 1.0f)
		discard;

	// view space normal
	vec3 normal;
	normal.xy = in_location;
	normal.z = -sqrt(1.0 - curRadius);

	// view space position
	vec3 position = in_position;
	position.z += (normal * in_radius).z;

	// clip space position
	vec4 clipPos = u_cam.projection * vec4(position, 1.0);
	gl_FragDepth = clipPos.z / clipPos.w;

	shade(toWorld(vec4(position, 1.0)), toWorld(vec4(normal, 0.0)), in_location);
}