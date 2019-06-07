struct CameraTransforms
{
	mat4 viewProj;
	mat4 view;
	mat4 projection;
	mat4 invView;
	// additional useful information
	vec3 position;
	float near;
	vec3 direction;
	float far;
	uvec2 screen;
};