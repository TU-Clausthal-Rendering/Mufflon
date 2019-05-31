#version 460
layout(local_size_x = 16, local_size_y  = 16) in;
layout(binding = 0) uniform sampler2D src_image;
layout(binding = 0) restrict buffer dst_buffer {
    float data[];
};
layout(location = 0) uniform uvec2 size;

void main(){
    vec3 color = texelFetch(src_image, ivec2(gl_GlobalInvocationID), 0).rgb;
    if(gl_GlobalInvocationID.x >= size.x) return;
    if(gl_GlobalInvocationID.y >= size.y) return;

	// TODO why has x to be flipped?
    uint index = (gl_GlobalInvocationID.y + 1) * size.x - gl_GlobalInvocationID.x - 1;
    //uint index = gl_GlobalInvocationID.y * size.x + gl_GlobalInvocationID.x;
    data[3 * index] += color.r;
    data[3 * index + 1] += color.g;
    data[3 * index + 2] += color.b;
}