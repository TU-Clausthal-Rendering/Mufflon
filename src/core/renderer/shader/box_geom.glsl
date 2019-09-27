layout(lines) in;
layout(triangle_strip, max_vertices = 4 * 6) out;

// instance id to determine model matrix
// => if u_instanceId < 0 use gl_PrimitiveIDIn
layout(location = 1) uniform int u_instanceId;

void draw_face(vec3 topLeft, vec3 topRight, vec3 botLeft, vec3 botRight) {
	mat4x3 model = getModelMatrix(u_instanceId < 0 ? gl_PrimitiveIDIn : u_instanceId);
	
	gl_Position = u_cam.viewProj * vec4(model * vec4(topLeft, 1.0), 1.0);
	EmitVertex();

	gl_Position = u_cam.viewProj * vec4(model * vec4(topRight, 1.0), 1.0);
	EmitVertex();

	gl_Position = u_cam.viewProj * vec4(model * vec4(botLeft, 1.0), 1.0);
	EmitVertex();

	gl_Position = u_cam.viewProj * vec4(model * vec4(botRight, 1.0), 1.0);
	EmitVertex();

	EndPrimitive();
}

#define BACK_TOP_LEFT 0
#define BACK_TOP_RIGHT 1
#define BACK_BOT_LEFT 2
#define BACK_BOT_RIGHT 3
#define FRONT_TOP_LEFT 4
#define FRONT_TOP_RIGHT 5
#define FRONT_BOT_LEFT 6
#define FRONT_BOT_RIGHT 7

void main() {
	// draw cube
	vec3 corners[8];

	const vec3 pmin = gl_in[0].gl_Position.xyz;
	const vec3 pmax = gl_in[1].gl_Position.xyz;

	// fill all points with FRONT_BOT_LEFT (bbox min)
	for (int i = 0; i < 8; ++i)
		corners[i] = pmin;
	
	corners[BACK_TOP_RIGHT] = pmax;

	// backside
	corners[BACK_TOP_LEFT].yz = pmax.yz;
	corners[BACK_BOT_RIGHT].xz = pmax.xz;
	corners[BACK_BOT_LEFT].z = pmax.z;

	// frontside
	corners[FRONT_TOP_RIGHT].xy = pmax.xy;
	corners[FRONT_TOP_LEFT].y = pmax.y;
	corners[FRONT_BOT_RIGHT].x = pmax.x;

	// draw all sides

	// front, back
	draw_face(corners[FRONT_TOP_LEFT], corners[FRONT_TOP_RIGHT], corners[FRONT_BOT_LEFT], corners[FRONT_BOT_RIGHT]);
	draw_face(corners[BACK_TOP_LEFT], corners[BACK_TOP_RIGHT], corners[BACK_BOT_LEFT], corners[BACK_BOT_RIGHT]);
	// left, right
	draw_face(corners[BACK_TOP_LEFT], corners[FRONT_TOP_LEFT], corners[BACK_BOT_LEFT], corners[FRONT_BOT_LEFT]);
	draw_face(corners[BACK_TOP_RIGHT], corners[FRONT_TOP_RIGHT], corners[BACK_BOT_RIGHT], corners[FRONT_BOT_RIGHT]);
	// top, bottom
	draw_face(corners[BACK_TOP_LEFT], corners[BACK_TOP_RIGHT], corners[FRONT_TOP_LEFT], corners[FRONT_TOP_RIGHT]);
	draw_face(corners[BACK_BOT_LEFT], corners[BACK_BOT_RIGHT], corners[FRONT_BOT_LEFT], corners[FRONT_BOT_RIGHT]);
}