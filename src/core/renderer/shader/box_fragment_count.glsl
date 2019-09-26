layout(early_fragment_tests) in;
layout(location = 0) out vec4 out_fragColor;

layout(binding = 6) buffer fragmentCountBuffer {
	uint u_fragmentCount[];
};

void main() {
	// count fragment
	uint index = uint(gl_FragCoord.y) * uint(u_cam.screen.x) + uint(gl_FragCoord.x);
	atomicAdd(u_fragmentCount[index], 1);

	out_fragColor = vec4(1.0);
}