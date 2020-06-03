#include "util/types.hpp"
#include "core/cuda/error.hpp"
#include "core/memory/unique_device_ptr.hpp"
#include <ei/3dtypes.hpp>
#include <ei/vector.hpp>
#include <cooperative_groups.h>
#include <device_launch_parameters.h>
#include <tuple>

namespace mufflon { namespace scene { namespace geometry {



class GpuStream;

class GpuEvent {
public:
	GpuEvent() : m_event{}
	{
		cuda::check_error(::cudaEventCreate(&m_event));
	}
	GpuEvent(const bool timing) :
		m_event{} {
		cuda::check_error(::cudaEventCreateWithFlags(&m_event, timing ? cudaEventDisableTiming : cudaEventDefault));
	}
	GpuEvent(const GpuEvent&) = delete;
	GpuEvent(GpuEvent&& other) :
		m_event{ other.m_event }
	{
		other.m_event = cudaEvent_t{};
	}
	GpuEvent& operator=(const GpuEvent&) = delete;
	GpuEvent& operator=(GpuEvent&& other) = delete;
	~GpuEvent() {
		if(m_event) cuda::check_error(::cudaEventDestroy(m_event));
	}

	static GpuEvent create_recorded() {
		GpuEvent evt{};
		evt.record();
		return evt;
	}
	static GpuEvent create_recorded(const GpuStream& stream) {
		GpuEvent evt{};
		evt.record(stream);
		return evt;
	}

	GpuEvent& record() {
		cuda::check_error(::cudaEventRecord(m_event, 0)); return *this;
	}
	GpuEvent& record(const GpuStream& stream);
	GpuEvent& synchronize() {
		cuda::check_error(::cudaEventSynchronize(m_event)); return *this;
	}
	const GpuEvent& synchronize() const {
		cuda::check_error(::cudaEventSynchronize(m_event)); return *this;
	}
	float elapsed_time(const GpuEvent& start) const {
		float time; cuda::check_error(::cudaEventElapsedTime(&time, start.m_event, m_event)); return time;
	}

	cudaEvent_t native() const noexcept { return m_event; }

private:
	cudaEvent_t m_event;
};

class GpuStream {
public:
	GpuStream(const bool nonBlocking = false) :
		m_stream{}
	{ 
		cuda::check_error(::cudaStreamCreateWithFlags(&m_stream, nonBlocking ? cudaStreamNonBlocking : cudaStreamDefault));
	}
	GpuStream(const GpuStream&) = delete;
	GpuStream(GpuStream&& other) noexcept :
		m_stream{ other.m_stream }
	{
		other.m_stream = cudaStream_t{};
	}
	GpuStream& operator=(const GpuStream&) = delete;
	GpuStream& operator=(GpuStream&&) = delete;
	~GpuStream() {
		if(m_stream) cuda::check_error(::cudaStreamDestroy(m_stream));
	}

	static GpuStream default_stream() noexcept {
		return GpuStream{ cudaStream_t{ 0 } };
	}

	void synchronize() {
		cuda::check_error(::cudaStreamSynchronize(m_stream));
	}
	void wait_for(const GpuEvent& evt) {
		cuda::check_error(::cudaStreamWaitEvent(m_stream, evt.native(), 0));
	}
	bool completed() const {
		const auto res = ::cudaStreamQuery(m_stream);
		if(res == cudaError::cudaSuccess)
			return true;
		else if(res != cudaError::cudaErrorNotReady)
			cuda::check_error(res);
		return false;
	}

	void begin_capture() {
		cuda::check_error(::cudaStreamBeginCapture(m_stream, cudaStreamCaptureModeGlobal));
	}
	cudaGraph_t end_capture() {
		cudaGraph_t graph;
		cuda::check_error(::cudaStreamEndCapture(m_stream, &graph));
		return graph;
	}

	cudaStream_t native() const noexcept { return m_stream; }

private:
	GpuStream(cudaStream_t stream) : m_stream{ stream } {}

	cudaStream_t m_stream;
};

GpuEvent& GpuEvent::record(const GpuStream& stream) {
	cuda::check_error(::cudaEventRecord(m_event, stream.native())); return *this;
}

template < class T >
class HostPinnedMemory {
public:
	HostPinnedMemory() {
		cuda::check_error(::cudaMallocHost(&m_ptr, sizeof(*m_ptr)));
	}
	HostPinnedMemory(const HostPinnedMemory&) = delete;
	HostPinnedMemory(HostPinnedMemory&& other) noexcept : m_ptr{ other.m_ptr }
	{
		other.m_ptr = nullptr;
	}
	HostPinnedMemory& operator=(const HostPinnedMemory&) = delete;
	HostPinnedMemory& operator=(HostPinnedMemory&&) = delete;
	~HostPinnedMemory() {
		cuda::check_error(::cudaFreeHost(m_ptr));
	}

	T& operator*() noexcept { return *m_ptr; }
	T* operator->() noexcept { return m_ptr; }

private:
	T* m_ptr;
};

std::pair<dim3, dim3> get_max_block_dim_sizes() {
	int values[6];
	cuda::check_error(::cudaDeviceGetAttribute(values + 0, ::cudaDevAttrMaxBlockDimX, 0));
	cuda::check_error(::cudaDeviceGetAttribute(values + 1, ::cudaDevAttrMaxBlockDimY, 0));
	cuda::check_error(::cudaDeviceGetAttribute(values + 2, ::cudaDevAttrMaxBlockDimZ, 0));
	cuda::check_error(::cudaDeviceGetAttribute(values + 3, ::cudaDevAttrMaxGridDimX, 0));
	cuda::check_error(::cudaDeviceGetAttribute(values + 4, ::cudaDevAttrMaxGridDimY, 0));
	cuda::check_error(::cudaDeviceGetAttribute(values + 5, ::cudaDevAttrMaxGridDimZ, 0));
	return std::make_pair(dim3{ static_cast<unsigned>(values[0]), static_cast<unsigned>(values[1]), static_cast<unsigned>(values[2]) },
						  dim3{ static_cast<unsigned>(values[3]), static_cast<unsigned>(values[4]), static_cast<unsigned>(values[5]) });
}

template < class Func, class... Args >
void launch_with_best_grid_block_size(const GpuStream& stream, const Func& func, const std::pair<dim3, dim3>& maxDims,
									  const unsigned requiredThreadCount,
									  Args&&... args) {
	launch_with_best_grid_block_size_dyn_smem(stream, func, maxDims, requiredThreadCount, 0u, std::forward<Args>(args)...);
}

template < class Func, class... Args >
void launch_with_best_grid_block_size_dyn_smem(const GpuStream& stream, const Func& func, const std::pair<dim3, dim3>& maxDims,
											   const unsigned requiredThreadCount, const unsigned smemPerThread,
											   Args&&... args) {
	if(requiredThreadCount == 0u)
		return;
	struct SMemPerBlockFunctor {
		unsigned perThread;

		__host__ __device__ unsigned operator()(const unsigned blockSize) const noexcept {
			return blockSize * perThread;
		}
	};

	int blockSize1D, minGridSize1D;
	cuda::check_error(::cudaOccupancyMaxPotentialBlockSizeVariableSMem(&minGridSize1D, &blockSize1D, func, SMemPerBlockFunctor{ smemPerThread }));

	// Distribute block size
	dim3 blockSize{ static_cast<unsigned>(blockSize1D), 1u, 1u };
	if(blockSize.x > maxDims.first.x) {
		blockSize.x = maxDims.first.x;
		blockSize.y = 1u + (static_cast<unsigned>(blockSize1D) - 1u) / maxDims.first.x;
		// TODO: use root for less overhead
		// TODO: if y too large, distribute to z
		if(blockSize.y > maxDims.first.y)
			throw std::runtime_error("Too many threads required in block");
	}

	const auto totalBlockSize = blockSize.x * blockSize.y * blockSize.z;
	const auto gridSize1D = 1u + (requiredThreadCount - 1u) / totalBlockSize;
	dim3 gridSize{ gridSize1D, 1u, 1u };
	if(gridSize.x > maxDims.second.x) {
		gridSize.x = maxDims.second.x;
		gridSize.y = 1u + (requiredThreadCount - 1u) / maxDims.second.x;
		// TODO: use root for less overhead
		// TODO: if y too large, distribute to z
		if(gridSize.y > maxDims.first.y)
			throw std::runtime_error("Too many threads required in grid");
	}

	func<<<gridSize, blockSize, smemPerThread* blockSize.x* blockSize.y, stream.native()>>>(std::forward<Args>(args)...);
	//cuda::check_error(::cudaDeviceSynchronize());
}

struct EdgeDeviation {
	ei::Vec3 baseVertPos;
	float minCos;
};
struct VertexGrading {
	u32 vertexIndex;
	union GradeOrGridIndex {
		float grade;
		u32 gridIndex;
	} data;
};
union DeviationOrGrading {
	EdgeDeviation devation;
	VertexGrading grading;

	__device__ float get_grade() const noexcept { return grading.data.grade; }
	__device__ u32 get_grid_index() const noexcept { return grading.data.gridIndex; }
	__device__ void set_grid_index(const u32 gridIndex) noexcept { grading.data.gridIndex = gridIndex; }
};

struct Cluster {
	ei::Vec3 sum;
	u32 id;
	struct GradePair {
		u32 reprVertIdx;
		union CountOrMaxGrade {
			u32 count;
			float maxGrade;
		} countOrGrade;

		__device__ bool operator!=(const GradePair& rhs) const noexcept {
			return reprVertIdx != rhs.reprVertIdx && countOrGrade.maxGrade != rhs.countOrGrade.maxGrade;
		}
	} gradePair;

	__device__ u32 repr_vert_idx() const noexcept { return gradePair.reprVertIdx; }
	__device__ u32 vertex_count() const noexcept { return gradePair.countOrGrade.count; }
	__device__ bool is_valid() const noexcept { return id != std::numeric_limits<u32>::max(); }
};
struct ChosenCluster {
	u32 index;
	float distance;
};

struct VertexDataSharedMem {
	ei::Vec3 point;
	float grade;
};

struct VertFaceCounters {
	unsigned remVertices;
	unsigned remTris;
	unsigned remQuads;
};

struct KernelExecParams {
	Cluster* grids;
	DeviationOrGrading* vertexGrade;
	u32* __restrict__ indices;
	u32* __restrict__ remainingIndices;
	const char* __restrict__ vertexAttributes;
	char* __restrict__ remVertAttributes;
	const char* __restrict__ faceAttributes;
	char* __restrict__ remFaceAttributes;
	const unsigned* __restrict__ vertAttribSizes;
	const unsigned* __restrict__ faceAttribSizes;
	const unsigned vertAttribCount;
	const unsigned faceAttribCount;
	const ei::UVec3 gridRes;
	const ei::Box aabb;
	const unsigned vertexAttribSize;
	const unsigned faceAttribSize;
	const unsigned triangleCount;
	const unsigned quadCount;
	const unsigned vertexCount;
	const unsigned clusterCount;
	VertFaceCounters counters;

	__forceinline__ __device__ const ei::Vec3& point(const unsigned index) const noexcept {
		return *(reinterpret_cast<const ei::Vec3*>(vertexAttributes) + index);
	}
};

struct KernelParams {
	Cluster* grids;
	DeviationOrGrading* vertexGrade;
	u32* indices;
	u32* remainingIndices;
	char* vertexAttributes;
	char* remVertAttributes;
	char* faceAttributes;
	char* remFaceAttributes;
	unsigned* vertAttribSizes;
	unsigned* faceAttribSizes;
	unsigned vertAttribCount;
	unsigned faceAttribCount;
	ei::UVec3 gridRes;
	ei::Box aabb;
	unsigned vertexAttribSize;
	unsigned faceAttribSize;
	unsigned triangleCount;
	unsigned quadCount;
	unsigned vertexCount;
	unsigned clusterCount;
};

constexpr std::size_t align_up(const std::size_t offset, const std::size_t alignment) noexcept {
	return (1u + (offset - 1u) / alignment) * alignment;
}

std::pair<KernelParams, unique_device_ptr<Device::CUDA, char[]>> allocate_kernel_parameters(const unsigned vertexCount,
																							const unsigned triangleCount,
																							const unsigned quadCount,
																							const unsigned vertexAttribSize,
																							const unsigned faceAttribSize,
																							const std::size_t vertAttribCount,
																							const std::size_t faceAttribCount,
																							const ei::UVec3& gridRes,
																							const ei::Box& aabb) {
	// Perform one big allocation for all the necessary data
	// First compute the sizes necessary
	const std::size_t gridsSize = sizeof(Cluster) * ei::prod(gridRes);
	const std::size_t vertexGradeSize = sizeof(DeviationOrGrading) * vertexCount;
	const std::size_t indicesSize = sizeof(u32) * (3u * triangleCount + 4u * quadCount);
	const std::size_t vertexAttribsSize = vertexAttribSize * vertexCount;
	const std::size_t faceAttribsSize = faceAttribSize * (triangleCount + quadCount);
	const std::size_t vertexSizesSize = sizeof(unsigned) * vertAttribCount;
	const std::size_t faceSizesSize = sizeof(unsigned) * faceAttribCount;

	// Now compute the offsets of each array, incorporating alignment requirements
	const std::size_t gridsOffset = align_up(sizeof(KernelExecParams), alignof(Cluster));
	const std::size_t vertexGradeOffset = align_up(gridsOffset + gridsSize, alignof(DeviationOrGrading));
	const std::size_t indicesOffset = align_up(vertexGradeOffset + vertexGradeSize, alignof(u32));
	const std::size_t remIndicesOffset = align_up(indicesOffset + indicesSize, alignof(u32));
	const std::size_t vertexAttribOffset = align_up(remIndicesOffset + indicesSize, alignof(float));
	const std::size_t remVertAttribOffset = align_up(vertexAttribOffset + vertexAttribsSize, alignof(float));
	const std::size_t faceAttribOffset = align_up(remVertAttribOffset + vertexAttribsSize, alignof(float));
	const std::size_t remFaceAttribOffset = align_up(faceAttribOffset + faceAttribsSize, alignof(float));
	const std::size_t vertAttribSizesOffset = align_up(remFaceAttribOffset + faceAttribsSize, alignof(unsigned));
	const std::size_t faceAttribSizesOffset = align_up(vertAttribSizesOffset + vertexSizesSize, alignof(unsigned));
	const std::size_t totalSize = faceAttribSizesOffset + faceSizesSize;

	auto buffer = make_udevptr_array<Device::CUDA, char, false>(totalSize);
	KernelParams hostParams{
		reinterpret_cast<Cluster*>(buffer.get() + gridsOffset),
		reinterpret_cast<DeviationOrGrading*>(buffer.get() + vertexGradeOffset),
		reinterpret_cast<u32*>(buffer.get() + indicesOffset),
		reinterpret_cast<u32*>(buffer.get() + remIndicesOffset),
		reinterpret_cast<char*>(buffer.get() + vertexAttribOffset),
		reinterpret_cast<char*>(buffer.get() + remVertAttribOffset),
		reinterpret_cast<char*>(buffer.get() + faceAttribOffset),
		reinterpret_cast<char*>(buffer.get() + remFaceAttribOffset),
		reinterpret_cast<unsigned*>(buffer.get() + vertAttribSizesOffset),
		reinterpret_cast<unsigned*>(buffer.get() + faceAttribSizesOffset),
		static_cast<unsigned>(vertAttribCount), static_cast<unsigned>(faceAttribCount),
		gridRes, aabb, static_cast<unsigned>(vertexAttribSize), static_cast<unsigned>(faceAttribSize),
		triangleCount, quadCount, vertexCount, ei::prod(gridRes)
	};

	auto* devParams = reinterpret_cast<KernelExecParams*>(buffer.get());
	cuda::check_error(::cudaMemcpy(devParams, &hostParams, sizeof(hostParams), cudaMemcpyHostToDevice));
	cuda::check_error(::cudaMemset(&devParams->counters, 0, sizeof(VertFaceCounters)));
	return std::make_pair(hostParams, std::move(buffer));
}

__global__ void get_base_edge_triangles(const KernelExecParams* params) {
	const auto xCoord = threadIdx.x + blockIdx.x * blockDim.x;
	const auto yCoord = threadIdx.y + blockIdx.y * blockDim.y;
	const auto index = xCoord + yCoord * blockDim.x * gridDim.x;
	if(index < params->triangleCount) {
		const ei::UVec3 tri{ params->indices[3u * index + 0u], params->indices[3u * index + 1u], params->indices[3u * index + 2u] };
		params->vertexGrade[tri.x].devation = { params->point(tri.y), 1.f };
		params->vertexGrade[tri.y].devation = { params->point(tri.z), 1.f };
		params->vertexGrade[tri.z].devation = { params->point(tri.x), 1.f };
	}
}

__global__ void get_base_edge_quads(const KernelExecParams* params) {
	const auto xCoord = threadIdx.x + blockIdx.x * blockDim.x;
	const auto yCoord = threadIdx.y + blockIdx.y * blockDim.y;
	const auto index = xCoord + yCoord * blockDim.x * gridDim.x;
	if(index < params->quadCount) {
		const ei::UVec4 quad{
			params->indices[3u * params->triangleCount + 4u * index + 0u],
			params->indices[3u * params->triangleCount + 4u * index + 1u],
			params->indices[3u * params->triangleCount + 4u * index + 2u],
			params->indices[3u * params->triangleCount + 4u * index + 3u]
		};
		params->vertexGrade[quad.x].devation = { params->point(quad.y), 1.f };
		params->vertexGrade[quad.y].devation = { params->point(quad.z), 1.f };
		params->vertexGrade[quad.z].devation = { params->point(quad.w), 1.f };
		params->vertexGrade[quad.w].devation = { params->point(quad.x), 1.f };
	}
}

__global__ void compute_max_edge_deviation_triangles(const KernelExecParams* params) {
	const auto xCoord = threadIdx.x + blockIdx.x * blockDim.x;
	const auto yCoord = threadIdx.y + blockIdx.y * blockDim.y;
	const auto index = xCoord + yCoord * blockDim.x * gridDim.x;
	if(index < params->triangleCount) {
		const ei::UVec3 tri{ params->indices[3u * index + 0u], params->indices[3u * index + 1u], params->indices[3u * index + 2u] };
		const ei::Vec3 points[] = { params->point(tri.x), params->point(tri.y), params->point(tri.z) };

		for(int i = 0; i < 3; ++i) {
			const int indices[] = { (i + 1) % 3, (i + 2) % 3 };
			for(unsigned j = 0u; j < 2u; ++j) {
				const auto otherIndex = indices[j];
				if(points[otherIndex] != params->vertexGrade[tri[i]].devation.baseVertPos) {
					const auto base = ei::normalize(params->vertexGrade[tri[i]].devation.baseVertPos - points[i]);
					const auto edge = ei::normalize(points[otherIndex] - points[i]);
					params->vertexGrade[tri[i]].devation.minCos = std::min(params->vertexGrade[tri[i]].devation.minCos, ei::dot(base, edge));
				}
			}
		}
	}
}
__global__ void compute_max_edge_deviation_quads(const KernelExecParams* params) {
	const auto xCoord = threadIdx.x + blockIdx.x * blockDim.x;
	const auto yCoord = threadIdx.y + blockIdx.y * blockDim.y;
	const auto index = xCoord + yCoord * blockDim.x * gridDim.x;
	if(index < params->quadCount) {
		const ei::UVec4 quad{
			params->indices[3u * params->triangleCount + 4u * index + 0u],
			params->indices[3u * params->triangleCount + 4u * index + 1u],
			params->indices[3u * params->triangleCount + 4u * index + 2u],
			params->indices[3u * params->triangleCount + 4u * index + 3u]
		};
		const ei::Vec3 points[] = { params->point(quad.x), params->point(quad.y), params->point(quad.z), params->point(quad.w) };

		for(int i = 0; i < 4; ++i) {
			// Reach around quad for other index (I don't trust negative modulo^^)
			const int indices[] = { (i + 1) % 4, (i + 3) % 4 };
			for(unsigned j = 0u; j < 2u; ++j) {
				const auto otherIndex = indices[j];
				if(points[otherIndex] != params->vertexGrade[quad[i]].devation.baseVertPos) {
					const auto base = ei::normalize(params->vertexGrade[quad[i]].devation.baseVertPos - points[i]);
					const auto edge = ei::normalize(points[otherIndex] - points[i]);
					params->vertexGrade[quad[i]].devation.minCos = std::min(params->vertexGrade[quad[i]].devation.minCos, ei::dot(base, edge));
				}
			}
		}
	}
}

__global__ void grade_vertices(const KernelExecParams* params) {
	const auto xCoord = threadIdx.x + blockIdx.x * blockDim.x;
	const auto yCoord = threadIdx.y + blockIdx.y * blockDim.y;
	const auto index = xCoord + yCoord * blockDim.x * gridDim.x;
	if(index < params->vertexCount)
		params->vertexGrade[index].grading = { index, std::cos(std::acos(params->vertexGrade[index].devation.minCos) / 2.f) };
}

__device__ void check_neighboring_grid_cells(const KernelExecParams* params, const u32 centreGridIdx,
											 const ei::UVec3& doubleResGridPos, const ei::Vec3& currVertPos,
											 ChosenCluster& chosenGrid) {
	// The other grid indices we get by adding/subtracting one if the double grid position is odd/even
	const auto compareGrid = [params, &chosenGrid, centreGridIdx, currVertPos](const i32 indexOffset) {
		const auto currGridIndex = centreGridIdx + indexOffset;
		// Check if the grid cell even has a cluster associated with it
		if(params->grids[currGridIndex].is_valid()) {
			// We compare the distance between vertex and cluster instead of using cluster weight
			const auto distance = ei::len(currVertPos - params->point(params->grids[currGridIndex].repr_vert_idx()));
			if(distance < chosenGrid.distance)
				chosenGrid = { currGridIndex, distance };
		}
	};

	const auto gridRes = params->gridRes;
	// Iterate all neighbor cells
	const ei::UVec3 neighborOffset{ 1u, gridRes.x, gridRes.x * gridRes.y };
	for(unsigned i = 1u; i < 8u; ++i) {
		// Loop over the three dimensions
		for(unsigned j = 0u; j < 3u; ++j) {
			// Interpret 'i' as the bitmask determining the dimension, so 1 == 001 == x only, 5 == 101 == x and z etc.
			if((i & (1u << j)) != 0u) {
				// Depending on which side of the cell we are we either check the "left" or "right" neighbor
				// Also make sure that we don't run out of the grid (if we do, there is no neighbor for this dimension)
				if((doubleResGridPos[j] & 1u) == 0u) {
					if(doubleResGridPos[j] > 0u)
						compareGrid(-static_cast<i32>(neighborOffset[j]));
				} else if(doubleResGridPos[j] < gridRes[j] * 2u - 1u) {
					compareGrid(static_cast<i32>(neighborOffset[j]));
				}
			}
		}
	}
}

__global__ void determine_cluster_representatives(const KernelExecParams* params) {
	const auto xCoord = threadIdx.x + blockIdx.x * blockDim.x;
	const auto yCoord = threadIdx.y + blockIdx.y * blockDim.y;
	const auto index = xCoord + yCoord * blockDim.x * gridDim.x;
	if(index < params->vertexCount) {
		// Determine which grid cell the vertex lies in
		const auto aabbDiag = params->aabb.max - params->aabb.min;
		const auto normPos = (params->point(index) - params->aabb.min) / aabbDiag;
		const auto gridPos = ei::min(ei::UVec3{ normPos * ei::Vec3{ params->gridRes } }, params->gridRes - 1u);
		const auto gridIndex = gridPos.x + gridPos.y * params->gridRes.x + gridPos.z * params->gridRes.y * params->gridRes.x;

		// Check if the grade in the grid cell is lower
		const auto grade = params->vertexGrade[index].get_grade();

		// Try to overwrite the index and grade if we're better than the previous
		auto* gradeAddress = &params->grids[gridIndex].gradePair;
		static_assert(sizeof(*gradeAddress) == sizeof(unsigned long long),
					  "For the CAS to work this has to have a size of 8 bytes");
		auto old = *gradeAddress;
		Cluster::GradePair assumed;
		Cluster::GradePair newPair;
		newPair.reprVertIdx = index;
		newPair.countOrGrade.maxGrade = grade;
		do {
			assumed = old;
			if(grade <= old.countOrGrade.maxGrade && old.reprVertIdx == std::numeric_limits<u32>::max())
				break;
			unsigned long long assumedInt, newInt;
			memcpy(&assumedInt, &assumed, sizeof(assumed));
			memcpy(&newInt, &newPair, sizeof(newPair));
			const auto oldInt = ::atomicCAS(reinterpret_cast<unsigned long long*>(gradeAddress), assumedInt, newInt);
			memcpy(&old, &oldInt, sizeof(old));
		} while(assumed != old);
	}
}

__global__ void compute_cluster_ids_and_clear_count(KernelExecParams* params) {
	const auto xCoord = threadIdx.x + blockIdx.x * blockDim.x;
	const auto yCoord = threadIdx.y + blockIdx.y * blockDim.y;
	const auto index = xCoord + yCoord * blockDim.x * gridDim.x;
	if(index < params->clusterCount) {
		if(params->grids[index].repr_vert_idx() != std::numeric_limits<u32>::max()) {
			params->grids[index].sum = ei::Vec3{ 0.f };
			params->grids[index].gradePair.countOrGrade.count = 0u;
			// Tell the vertex which cluster it is a part of
			// TODO: this probably isn't needed because the next kernel does it
			params->vertexGrade[params->grids[index].repr_vert_idx()].set_grid_index(index);
			auto g = cooperative_groups::coalesced_threads();
			unsigned warpRes;
			if(g.thread_rank() == 0u)
				warpRes = ::atomicAdd(&params->counters.remVertices, g.size());
			params->grids[index].id = g.shfl(warpRes, 0u) + g.thread_rank();
		}
	}
}

__global__ void assign_vertices_to_clusters(const KernelExecParams* params) {
	const auto xCoord = threadIdx.x + blockIdx.x * blockDim.x;
	const auto yCoord = threadIdx.y + blockIdx.y * blockDim.y;
	const auto index = xCoord + yCoord * blockDim.x * gridDim.x;
	if(index < params->vertexCount) {
		// We now have to determine what cluster to assign the remaining vertices to
		const auto aabbDiag = params->aabb.max - params->aabb.min;
		const auto get_grid_index = [params](const ei::UVec3& gridPos) {
			return gridPos.x + gridPos.y * params->gridRes.x + gridPos.z * params->gridRes.y * params->gridRes.x;
		};

		auto& vertex = params->vertexGrade[index];
		const auto pos = params->point(vertex.grading.vertexIndex);

		// Get the normalized position [0, 1]^3
		const auto normPos = (pos - params->aabb.min) / aabbDiag;
		// Get the discretized grid position with twice the grid resolution
		const auto doubleResGridPos = ei::min(ei::UVec3{ normPos * ei::Vec3{ 2u * params->gridRes } }, 2u * params->gridRes - 1u);
		const auto centreGridIdx = get_grid_index(doubleResGridPos / 2u);
		// The grid has to be valid since we determined the grid centres earlier
		ChosenCluster chosenGrid{ centreGridIdx, ei::len(pos - params->point(params->grids[centreGridIdx].repr_vert_idx())) };

		// Check the closest surrounding neighbor cells whether their centre is closer to our vertex
		check_neighboring_grid_cells(params, centreGridIdx, doubleResGridPos, pos, chosenGrid);

		const auto gridIdx = chosenGrid.index;
		::atomicAdd(&params->grids[gridIdx].gradePair.countOrGrade.count, 1u);
		::atomicAdd(&params->grids[gridIdx].sum.x, pos.x);
		::atomicAdd(&params->grids[gridIdx].sum.y, pos.y);
		::atomicAdd(&params->grids[gridIdx].sum.z, pos.z);

		// We set the grid index for the vertex - but after this loop, the array will be indexable by vertex index
		// again since we don't need the sorting by grade anymore!
		params->vertexGrade[vertex.grading.vertexIndex].set_grid_index(gridIdx);
	}
}

__global__ void coalesce_non_degenerate_triangles(KernelExecParams* params) {
	const auto xCoord = threadIdx.x + blockIdx.x * blockDim.x;
	const auto yCoord = threadIdx.y + blockIdx.y * blockDim.y;
	const auto index = xCoord + yCoord * blockDim.x * gridDim.x;
	if(index < params->triangleCount) {
		auto* tri = params->indices + 3u * index;
		// Check if triangle is degenerate
		if(params->vertexGrade[tri[0u]].get_grid_index() != params->vertexGrade[tri[1u]].get_grid_index()
		   && params->vertexGrade[tri[0u]].get_grid_index() != params->vertexGrade[tri[2u]].get_grid_index()
		   && params->vertexGrade[tri[1u]].get_grid_index() != params->vertexGrade[tri[2u]].get_grid_index()) {
			auto g = cooperative_groups::coalesced_threads();
			unsigned warpRes;
			if(g.thread_rank() == 0)
				warpRes = ::atomicAdd(&params->counters.remTris, g.size());
			const auto newIndex = g.shfl(warpRes, 0u) + g.thread_rank();

			// Compute sector normal for vertices
			const auto* points = reinterpret_cast<const ei::Vec3*>(params->vertexAttributes);
			const auto e01 = ei::normalize(points[tri[1u]] - points[tri[0u]]);
			const auto e02 = ei::normalize(points[tri[2u]] - points[tri[0u]]);
			const auto e12 = ei::normalize(points[tri[2u]] - points[tri[1u]]);
			const auto faceNormal = ei::cross(e01, e02);

			const float angles[] = { std::acos(ei::dot(e01, e02)), std::acos(-ei::dot(e01, e12)), std::acos(ei::dot(e02, e12)) };
			auto* normals = reinterpret_cast<ei::Vec3*>(params->remVertAttributes + params->vertAttribSizes[0u] * params->counters.remVertices);

			for(unsigned i = 0u; i < 3u; ++i) {
				const auto newVertexIndex = params->grids[params->vertexGrade[tri[i]].get_grid_index()].id;
				params->remainingIndices[3u * newIndex + i] = newVertexIndex;
				const auto sectorNormal = angles[i] * faceNormal;
				::atomicAdd(&normals[newVertexIndex].x, sectorNormal.x);
				::atomicAdd(&normals[newVertexIndex].y, sectorNormal.y);
				::atomicAdd(&normals[newVertexIndex].z, sectorNormal.z);
			}

			// Copy over the attributes (they won't be packed tightly on the GPU anymore)
			const auto faceCount = params->triangleCount + params->quadCount;
			const char* oldAttribStart = params->faceAttributes;
			char* newAttribStart = params->remFaceAttributes;
			for(unsigned i = 0u; i < params->faceAttribCount; ++i) {
				memcpy(newAttribStart + newIndex * params->faceAttribSizes[i],
					   oldAttribStart + index * params->faceAttribSizes[i],
					   params->faceAttribSizes[i]);
				oldAttribStart += params->faceAttribSizes[i] * faceCount;
				newAttribStart += params->faceAttribSizes[i] * faceCount;
			}
		}
	}
}

__global__ void count_and_copy_quads_turned_triangle(KernelExecParams* params) {
	const auto xCoord = threadIdx.x + blockIdx.x * blockDim.x;
	const auto yCoord = threadIdx.y + blockIdx.y * blockDim.y;
	const auto index = xCoord + yCoord * blockDim.x * gridDim.x;
	if(index < params->quadCount) {
		auto* quad = params->indices + 3u * params->triangleCount + 4u * index;
		ei::UVec4 gridIndices{
			params->vertexGrade[quad[0u]].get_grid_index(),
			params->vertexGrade[quad[1u]].get_grid_index(),
			params->vertexGrade[quad[2u]].get_grid_index(),
			params->vertexGrade[quad[3u]].get_grid_index()
		};
		u8 equalCount = 0u;
		for(u8 i = 0u; i < 4u; ++i) {
			for(u8 j = i + 1u; j < 4u; ++j) {
				if(gridIndices[i] == gridIndices[j]) {
					equalCount += 1u;
					gridIndices[i] = std::numeric_limits<u32>::max();
				}
			}
		}

		if(equalCount == 1u) {
			auto g = cooperative_groups::coalesced_threads();
			unsigned warpRes;
			if(g.thread_rank() == 0)
				warpRes = ::atomicAdd(&params->counters.remTris, g.size());
			const auto triIndex = g.shfl(warpRes, 0u) + g.thread_rank();
			unsigned newIndex = 3u * triIndex;

			// Get the indices of the triangle and copy over adjusted indices
			u32 tri[3u];
			unsigned currIndex = 0u;
			for(unsigned i = 0u; i < 4u; ++i) {
				if(gridIndices[i] != std::numeric_limits<u32>::max()) {
					params->remainingIndices[newIndex + currIndex] = params->grids[gridIndices[i]].id;
					tri[currIndex] = quad[i];
					currIndex += 1u;
				}
			}

			// Compute sector normal for vertices
			const auto* points = reinterpret_cast<const ei::Vec3*>(params->vertexAttributes);
			const auto e01 = ei::normalize(points[tri[1u]] - points[tri[0u]]);
			const auto e02 = ei::normalize(points[tri[2u]] - points[tri[0u]]);
			const auto e12 = ei::normalize(points[tri[2u]] - points[tri[1u]]);
			const auto faceNormal = ei::cross(e01, e02);

			const float angles[] = { std::acos(ei::dot(e01, e02)), std::acos(-ei::dot(e01, e12)), std::acos(ei::dot(e02, e12)) };
			auto* normals = reinterpret_cast<ei::Vec3*>(params->remVertAttributes + params->vertAttribSizes[0u] * params->counters.remVertices);

			for(unsigned i = 0u; i < 3u; ++i) {
				const auto newVertexIndex = params->grids[params->vertexGrade[tri[i]].get_grid_index()].id;
				params->remainingIndices[3u * newIndex + i] = newVertexIndex;
				const auto sectorNormal = angles[i] * faceNormal;
				::atomicAdd(&normals[newVertexIndex].x, sectorNormal.x);
				::atomicAdd(&normals[newVertexIndex].y, sectorNormal.y);
				::atomicAdd(&normals[newVertexIndex].z, sectorNormal.z);
			}

			// Copy over the attributes (they won't be packed tightly on the GPU anymore)
			const auto faceCount = params->triangleCount + params->quadCount;
			const char* oldAttribStart = params->faceAttributes;
			char* newAttribStart = params->remFaceAttributes;
			for(unsigned i = 0u; i < params->faceAttribCount; ++i) {
				memcpy(newAttribStart + triIndex * params->faceAttribSizes[i],
					   oldAttribStart + index * params->faceAttribSizes[i],
					   params->faceAttribSizes[i]);
				oldAttribStart += params->faceAttribSizes[i] * faceCount;
				newAttribStart += params->faceAttribSizes[i] * faceCount;
			}
		}
		
		if(equalCount != 0u) {
			quad[0] = std::numeric_limits<u32>::max();
			quad[1] = std::numeric_limits<u32>::max();
		}
	}
}

__global__ void coalesce_non_degenerate_quads(KernelExecParams* params) {
	const auto xCoord = threadIdx.x + blockIdx.x * blockDim.x;
	const auto yCoord = threadIdx.y + blockIdx.y * blockDim.y;
	const auto index = xCoord + yCoord * blockDim.x * gridDim.x;
	if(index < params->quadCount) {
		auto* quad = params->indices + 3u * params->triangleCount + 4u * index;
		if(quad[0u] != std::numeric_limits<u32>::max() && quad[1u] != std::numeric_limits<u32>::max()) {
			ei::UVec4 gridIndices{
				params->vertexGrade[quad[0u]].get_grid_index(),
				params->vertexGrade[quad[1u]].get_grid_index(),
				params->vertexGrade[quad[2u]].get_grid_index(),
				params->vertexGrade[quad[3u]].get_grid_index()
			};

			auto g = cooperative_groups::coalesced_threads();
			unsigned warpRes;
			if(g.thread_rank() == 0)
				warpRes = ::atomicAdd(&params->counters.remQuads, g.size());
			const auto quadIndex = g.shfl(warpRes, 0u) + g.thread_rank();
			const auto faceIndex = params->counters.remTris + quadIndex;
			const auto newIndex = 3u * params->counters.remTris + 4u * quadIndex;

			// Compute sector normal for vertices
			const auto* points = reinterpret_cast<const ei::Vec3*>(params->vertexAttributes);
			ei::Vec3 edges[] = {
				ei::normalize(points[quad[1u]] - points[quad[0u]]),
				ei::normalize(points[quad[2u]] - points[quad[1u]]),
				ei::normalize(points[quad[3u]] - points[quad[2u]]),
				ei::normalize(points[quad[0u]] - points[quad[3u]]),
			};

			const float angles[] = { std::acos(-ei::dot(edges[3], edges[0])), std::acos(-ei::dot(edges[0], edges[1])),
				std::acos(-ei::dot(edges[1], edges[2])), std::acos(-ei::dot(edges[2], edges[3])) };
			auto* normals = reinterpret_cast<ei::Vec3*>(params->remVertAttributes + params->vertAttribSizes[0u] * params->counters.remVertices);

			for(unsigned i = 0u; i < 4u; ++i) {
				const auto newVertexIndex = params->grids[gridIndices[i]].id;
				params->remainingIndices[newIndex + i] = newVertexIndex;
				const auto vertexNormal = ei::cross(edges[(i + 3u) % 4u], edges[i]);
				const auto sectorNormal = angles[i] * vertexNormal;
				::atomicAdd(&normals[newVertexIndex].x, sectorNormal.x);
				::atomicAdd(&normals[newVertexIndex].y, sectorNormal.y);
				::atomicAdd(&normals[newVertexIndex].z, sectorNormal.z);
			}

			// Copy over the attributes (they won't be packed tightly on the GPU anymore)
			const auto faceCount = params->triangleCount + params->quadCount;
			const char* oldAttribStart = params->faceAttributes;
			char* newAttribStart = params->remFaceAttributes;
			for(unsigned i = 0u; i < params->faceAttribCount; ++i) {
				memcpy(newAttribStart + faceIndex * params->faceAttribSizes[i],
					   oldAttribStart + (params->triangleCount + index) * params->faceAttribSizes[i],
					   params->faceAttribSizes[i]);
				oldAttribStart += params->faceAttribSizes[i] * faceCount;
				newAttribStart += params->faceAttribSizes[i] * faceCount;
			}
		}
	}
}

__global__ void swap_vertices(const KernelExecParams* params) {
	const auto xCoord = threadIdx.x + blockIdx.x * blockDim.x;
	const auto yCoord = threadIdx.y + blockIdx.y * blockDim.y;
	const auto index = xCoord + yCoord * blockDim.x * gridDim.x;
	if(index < params->clusterCount) {
		auto& cluster = params->grids[index];
		if(cluster.is_valid()) {
			const auto newIndex = cluster.id;
			const auto oldIndex = cluster.repr_vert_idx();

			auto* points = reinterpret_cast<ei::Vec3*>(params->remVertAttributes);
			auto* normals = reinterpret_cast<ei::Vec3*>(params->remVertAttributes + params->vertAttribSizes[0u] * params->counters.remVertices);
			// Set position as cluster average (more involved schemes possible, but cost more memory e.g. error quadrics)
			points[newIndex] = cluster.sum / static_cast<float>(cluster.vertex_count());
			// Normal has already been computed when compacting the triangles/quads, but needs to be normalized
			normals[newIndex] = ei::normalize(normals[newIndex]);

			// Unlike the face attributes we know how many remaining vertices there are before the kernel runs
			// and thus can directly place the attributes at the right position
			const char* oldAttribStart = params->vertexAttributes + (params->vertAttribSizes[0u] + params->vertAttribSizes[1u]) * params->vertexCount;
			char* newAttribStart = params->remVertAttributes + (params->vertAttribSizes[0u] + params->vertAttribSizes[1u]) * params->counters.remVertices;
			for(unsigned i = 2u; i < params->vertAttribCount; ++i) {
				memcpy(newAttribStart + newIndex * params->vertAttribSizes[i],
					   oldAttribStart + oldIndex * params->vertAttribSizes[i],
					   params->vertAttribSizes[i]);
				oldAttribStart += params->vertAttribSizes[i] * params->vertexCount;
				newAttribStart += params->vertAttribSizes[i] * params->counters.remVertices;
			}
		}
	}
}

std::tuple<unique_device_ptr<Device::CPU, char[]>, unique_device_ptr<Device::CPU, char[]>,
							 unique_device_ptr<Device::CPU, u32[]>,
							 unsigned, unsigned, unsigned>
cluster_uniformly_cuda(unique_device_ptr<Device::CPU, char[]> vertexAttributes,
					   unique_device_ptr<Device::CPU, char[]> faceAttributes,
					   unique_device_ptr<Device::CPU, u32[]> indices, const unsigned vertexAttribSize,
					   const unsigned faceAttribSize, const unsigned vertexCount,
					   const unsigned triangleCount, const unsigned quadCount,
					   const ei::Box& aabb, const ei::UVec3& gridRes,
					   const unsigned* vertAttribElemSizes,
					   const unsigned* faceAttribElemSizes,
					   const std::size_t vertAttribCount,
					   const std::size_t faceAttribCount) {
	const auto gridCount = ei::prod(gridRes);
	const auto indexCount = 3u * triangleCount + 4u * quadCount;

	// Set up device stuff
	cuda::check_error(::cudaSetDevice(0));
	//GpuStream mainStream{};
	GpuStream mainStream{ true };
	GpuEvent afterCopies{ false };
	GpuEvent afterVertCountBackcopy{ false };
	GpuEvent afterFaceCountBackcopy{ false };


	const auto maxDims = get_max_block_dim_sizes();
	// Allocate the counters, which we can host-pin (TODO: RAII)
	HostPinnedMemory<VertFaceCounters> vertFaceCounters{};
	// Perform one big allocation for all the necessary data
	auto allocedParams = allocate_kernel_parameters(vertexCount, triangleCount, quadCount,
													vertexAttribSize, faceAttribSize,
													vertAttribCount, faceAttribCount,
													gridRes, aabb);
	auto kernelParams = allocedParams.first;
	auto buffer = std::move(allocedParams.second);
	auto* devParams = reinterpret_cast<KernelExecParams*>(buffer.get());

	// Copy over data (part of this could be done in parallel with some kernels, but
	// the price of having to use page-pinned memory is too large)
	copy(kernelParams.indices, indices.get(), sizeof(u32) * indexCount);
	copy(kernelParams.vertexAttributes, vertexAttributes.get(), vertexAttribSize * vertexCount);
	copy(kernelParams.faceAttributes, faceAttributes.get(), faceAttribSize * (triangleCount + quadCount));
	copy(kernelParams.vertAttribSizes, vertAttribElemSizes, sizeof(*vertAttribElemSizes) * vertAttribCount);
	copy(kernelParams.faceAttribSizes, faceAttribElemSizes, sizeof(*faceAttribElemSizes) * faceAttribCount);
	// The clusters get initialized to 'invalid' with this
	mem_set<Device::CUDA>(kernelParams.grids, 0xFFFFFFFF, sizeof(*kernelParams.grids) * gridCount);
	// Clear the normals (they will be recomputed)
	mem_set<Device::CUDA>(kernelParams.remVertAttributes, 0, (vertAttribElemSizes[0u] + vertAttribElemSizes[1u]) * vertexCount);
	copy(kernelParams.vertAttribSizes, vertAttribElemSizes, sizeof(*vertAttribElemSizes) * vertAttribCount);
	copy(kernelParams.faceAttribSizes, faceAttribElemSizes, sizeof(*faceAttribElemSizes) * faceAttribCount);
	afterCopies.record();

	// Main stream activity
	mainStream.wait_for(afterCopies);
	launch_with_best_grid_block_size(mainStream, get_base_edge_triangles, maxDims, triangleCount, devParams);
	launch_with_best_grid_block_size(mainStream, get_base_edge_quads, maxDims, quadCount, devParams);
	launch_with_best_grid_block_size(mainStream, compute_max_edge_deviation_triangles, maxDims, triangleCount, devParams);
	launch_with_best_grid_block_size(mainStream, compute_max_edge_deviation_quads, maxDims, quadCount, devParams);
	launch_with_best_grid_block_size(mainStream, grade_vertices, maxDims, vertexCount, devParams);
	launch_with_best_grid_block_size(mainStream, determine_cluster_representatives, maxDims, vertexCount, devParams);
	launch_with_best_grid_block_size(mainStream, compute_cluster_ids_and_clear_count, maxDims, gridCount, devParams);
	cuda::check_error(::cudaMemcpyAsync(&vertFaceCounters->remVertices, &devParams->counters.remVertices,
										sizeof(unsigned), cudaMemcpyDeviceToHost, mainStream.native()));
	afterVertCountBackcopy.record(mainStream);
	launch_with_best_grid_block_size(mainStream, assign_vertices_to_clusters, maxDims, vertexCount, devParams);
	launch_with_best_grid_block_size(mainStream, coalesce_non_degenerate_triangles, maxDims, triangleCount, devParams);
	launch_with_best_grid_block_size(mainStream, count_and_copy_quads_turned_triangle, maxDims, quadCount, devParams);
	launch_with_best_grid_block_size(mainStream, coalesce_non_degenerate_quads, maxDims, quadCount, devParams);
	cuda::check_error(::cudaMemcpyAsync(&vertFaceCounters->remTris, &devParams->counters.remTris,
										2u * sizeof(unsigned), cudaMemcpyDeviceToHost, mainStream.native()));
	afterFaceCountBackcopy.record(mainStream);
	launch_with_best_grid_block_size(mainStream, swap_vertices, maxDims, gridCount, devParams);
	// Create the new CPU-side buffers (can happen async to some GPU work)
	afterVertCountBackcopy.synchronize();
	vertexAttributes = make_udevptr_array<Device::CPU, char, false>(vertFaceCounters->remVertices * vertexAttribSize);
	afterFaceCountBackcopy.synchronize();
	const auto remFaceCount = vertFaceCounters->remTris + vertFaceCounters->remQuads;
	const auto remIndexCount = 3u * vertFaceCounters->remTris + 4u * vertFaceCounters->remQuads;
	faceAttributes = make_udevptr_array<Device::CPU, char, false>(remFaceCount * faceAttribSize);
	indices = make_udevptr_array<Device::CPU, u32, false>(remIndexCount);
	
	// Copy back results
	mainStream.synchronize();
	copy(indices.get(), kernelParams.remainingIndices, remIndexCount * sizeof(u32));
	copy(vertexAttributes.get(), kernelParams.remVertAttributes, vertexAttribSize * vertFaceCounters->remVertices);
	// Face attributes have to be copied piece by piece because we didn't know the proper offsets during kernel
	std::size_t oldOffset = 0u;
	std::size_t newOffset = 0u;
	for(std::size_t i = 0u; i < faceAttribCount; ++i) {
		copy(faceAttributes.get() + newOffset, kernelParams.remFaceAttributes + oldOffset, faceAttribElemSizes[i] * remFaceCount);
		newOffset += faceAttribElemSizes[i] * remFaceCount;
		oldOffset += faceAttribElemSizes[i] * (triangleCount + quadCount);
	}

	return std::make_tuple(std::move(vertexAttributes), std::move(faceAttributes), std::move(indices),
						   vertFaceCounters->remVertices, vertFaceCounters->remTris, vertFaceCounters->remQuads);
}

}}} // namespace mufflon::scene::geometry