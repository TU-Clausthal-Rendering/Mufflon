layout(local_size_x = 64, local_size_y = 1) in;

#define TYPE uint

layout(binding = 0) restrict readonly buffer aux_buffer
{
	TYPE aux_data[];
};

layout(binding = 1) restrict buffer out_buffer
{
	TYPE data[];
};

layout(location = 0) uniform uint u_stride;
layout(location = 1) uniform uint u_lastWrite;

layout(binding = 2) restrict buffer out_buffer_size
{
	TYPE maxValue;
};

void main()
{
	data[gl_GlobalInvocationID.x + u_stride] += aux_data[int(gl_GlobalInvocationID.x / u_stride)];
	if (u_lastWrite != 0u && gl_GlobalInvocationID.x + u_stride == u_lastWrite)
	{
		maxValue = data[gl_GlobalInvocationID.x + u_stride];
	}
}