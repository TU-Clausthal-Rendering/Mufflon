#define WK_SIZE 1024

layout(local_size_x = WK_SIZE, local_size_y = 1) in;

#define TYPEV4 uvec4
#define TYPE uint

layout(binding = 2) readonly buffer in_buffer
{
	TYPEV4 in_data[];
};

layout(binding = 0) writeonly restrict buffer out_buffer
{
	TYPEV4 data[];
};

layout(binding = 1) writeonly restrict buffer out_auxiliaryBuffer
{
	TYPE aux[];
};

shared TYPE s_temp[WK_SIZE * 2];
shared TYPE s_blockSums[WK_SIZE / 32];

// Inclusive scan which produces the prefix sums for 64 elements in parallel
void intraWarpScan(int threadID)
{
	int id = threadID * 2;
	s_temp[id + 1] += s_temp[id];

	id = (threadID | 1) << 1;
	s_temp[id + (threadID & 1)] += s_temp[id - 1];

	id = ((threadID >> 1) | 1) << 2;
	s_temp[id + (threadID & 3)] += s_temp[id - 1];

	id = ((threadID >> 2) | 1) << 3;
	s_temp[id + (threadID & 7)] += s_temp[id - 1];

	id = ((threadID >> 3) | 1) << 4;
	s_temp[id + (threadID & 15)] += s_temp[id - 1];

	id = ((threadID >> 4) | 1) << 5;
	s_temp[id + (threadID & 31)] += s_temp[id - 1];
}

void intraBlockScan(int threadID)
{
	int id = threadID * 2;
	s_blockSums[id + 1] += s_blockSums[id];

	id = (threadID | 1) << 1;
	s_blockSums[id + (threadID & 1)] += s_blockSums[id - 1];

	id = ((threadID >> 1) | 1) << 2;
	s_blockSums[id + (threadID & 3)] += s_blockSums[id - 1];

	id = ((threadID >> 2) | 1) << 3;
	s_blockSums[id + (threadID & 7)] += s_blockSums[id - 1];

	id = ((threadID >> 3) | 1) << 4;
	s_blockSums[id + (threadID & 15)] += s_blockSums[id - 1];

	//id = ((threadID >> 4) | 1) << 5;
	//s_blockSums[id + (threadID & 31)] += s_blockSums[id - 1];

	/*int lane = threadID & 31;
	if(lane >= 1) s_temp[threadID] += s_temp[threadID - 1];
	if(lane >= 2) s_temp[threadID] += s_temp[threadID - 2];
	if(lane >= 4) s_temp[threadID] += s_temp[threadID - 4];
	if(lane >= 8) s_temp[threadID] += s_temp[threadID - 8];
	if(lane >= 16) s_temp[threadID] += s_temp[threadID - 16];*/
}

layout(location = 0) uniform uint u_bufferSize;

void main()
{
	//int idx = int(gl_GlobalInvocationID.x);
	int threadID = int(gl_LocalInvocationID.x);
	int idx = int(gl_WorkGroupID.x * WK_SIZE * 2) + threadID;
	TYPEV4 inputValuesA = in_data[idx];
	TYPEV4 inputValuesB = in_data[idx + WK_SIZE];
	inputValuesA.y += inputValuesA.x;
	inputValuesA.z += inputValuesA.y;
	inputValuesA.w += inputValuesA.z;
	s_temp[threadID] = inputValuesA.w;
	inputValuesB.y += inputValuesB.x;
	inputValuesB.z += inputValuesB.y;
	inputValuesB.w += inputValuesB.z;
	s_temp[threadID + WK_SIZE] = inputValuesB.w;
	barrier();
	memoryBarrierShared();

	// 1. Intra-warp scan in each warp
	intraWarpScan(threadID);
	//intraWarpScan(threadID + WK_SIZE);
	barrier();
	memoryBarrierShared();

	// 2. Collect per-warp sums
	if (threadID < (WK_SIZE / 32))
		s_blockSums[threadID] = s_temp[threadID * 64 + 63];

	// 3. Use 1st warp to scan per-warp results
	if (threadID < (WK_SIZE / 64))
		intraBlockScan(threadID);
	barrier();
	memoryBarrierShared();

	// 4. Add new warp offsets from step 3 to the results
	//idx = int(gl_WorkGroupID.x * WK_SIZE * 2);
	TYPE blockOffset = threadID < 64 ? 0 : s_blockSums[threadID / 64 - 1];
	TYPE val = s_temp[threadID] + blockOffset;
	if (idx < u_bufferSize / 4)
		data[idx] = TYPEV4(val - inputValuesA.w + inputValuesA.xyz, val);


	blockOffset = s_blockSums[(threadID + WK_SIZE) / 64 - 1];
	val = s_temp[threadID + WK_SIZE] + blockOffset;

	if (idx + WK_SIZE < u_bufferSize / 4)
		data[idx + WK_SIZE] = TYPEV4(val - inputValuesB.w + inputValuesB.xyz, val);

	// 5. The last thread in each block must return into the (thickly packed) auxiliary array
	if (threadID == WK_SIZE - 1)
		aux[gl_WorkGroupID.x] = val;
}