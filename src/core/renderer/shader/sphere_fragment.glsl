layout(location = 0) flat in vec3 in_center;
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
	float curRadius = distance(in_center, in_position);
	if (curRadius > in_radius)
		discard;

	// calculate ray-sphere intersection position
	vec3 toPosition = normalize(in_position);
	float tca = dot(in_center, toPosition);
	vec3 closestToCenter = toPosition * tca; // points on the toPosition vector that is closest to the sphere center
	
	// distance from closest to center to the sphere center
	float distToCenterSq = dot(closestToCenter - in_center, closestToCenter - in_center);
	// distance form closest to center to the sphere hull in -toPosition direction
	float thc = sqrt(in_radius * in_radius - distToCenterSq);

	// view space position
	vec3 position = toPosition * (tca - thc);
	vec3 normal = normalize(position - in_center);

	// clip space position
	vec4 clipPos = u_cam.projection * vec4(position, 1.0);
	// map clipPos from [-1, 1] to (probably) [0, 1]. With default values: clip.z/clip.w * 0.5 + 0.5
	gl_FragDepth = (clipPos.z / clipPos.w * gl_DepthRange.diff + (gl_DepthRange.near + gl_DepthRange.far)) * 0.5;

	// reconstruct polar coordinates from normal (radius is one)
	const vec3 worldNormal = toWorld(vec4(normal, 0.0));
	float theta = acos(worldNormal.y);
	float phi = atan(worldNormal.z, worldNormal.x);
	
	const float invPi = 1.0 / 3.14159265359;

#ifdef FORWARD_SHADE
	shade(
		toWorld(vec4(position, 1.0)), 
		worldNormal,  
		vec2(phi * invPi * 0.5 + 0.5, 1.0f - theta * invPi), 
		in_materialIndex
	);
#endif
#ifdef NDOTC_SHADE
	shade(toWorld(vec4(position, 1.0)), worldNormal);
#endif
}