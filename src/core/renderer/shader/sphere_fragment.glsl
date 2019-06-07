layout(location = 0) in vec2 in_location;
layout(location = 1) flat in float in_radius;
layout(location = 2) in vec3 in_position;
layout(location = 3) flat in uint in_materialIndex;

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
	// map clipPos from [-1, 1] to (probably) [0, 1]. With default values: clip.z/clip.w * 0.5 + 0.5
	gl_FragDepth = (clipPos.z / clipPos.w * gl_DepthRange.diff + (gl_DepthRange.near + gl_DepthRange.far)) * 0.5;

#ifdef FORWARD_SHADE
	// reconstruct polar coordinates from normal (radius is one)
	const vec3 worldNormal = toWorld(vec4(normal, 0.0));
	float theta = acos(worldNormal.y);
	float phi = atan(worldNormal.z, worldNormal.x);
	
	const float invPi = 1.0 / 3.14159265359;
	shade(
		toWorld(vec4(position, 1.0)), 
		worldNormal, 
		vec2(phi * invPi * 0.5 + 0.5, theta * invPi), 
		in_materialIndex
	);
#endif
}