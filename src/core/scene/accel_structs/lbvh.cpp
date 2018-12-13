#include "lbvh.hpp"
#include "core/scene/object.hpp"
#include "build_lbvh.hpp"

namespace mufflon {	namespace scene {namespace accel_struct {

/*LBVH::LBVH():
	m_device(Device::NONE)
{
}

LBVH::~LBVH()
{
	unload_resident(Device::CUDA);
	unload_resident(Device::CPU);
}

bool LBVH::is_resident(Device res) const
{
	return static_cast<bool>(m_device & res);
}

void LBVH::make_resident(Device res)
{
	if ((res & Device::CPU) != Device::NONE) {
		if (!is_resident(Device::CPU)) {
			mAssert(is_resident(Device::CUDA));

			// Copy input arrays from cuda to cpu.
			// TODO
			if (m_inputCPU.triIndices = nullptr) {

			}

			// Copy output arrays from cuda to cpu.
			/*cudaMemcpy(m_sizes.bvh, m_inputCUDA.bvh, m_inputCUDA.bvhSize * sizeof(ei::Vec4), cudaMemcpyDefault);
			m_sizes.primIds = (i32*)malloc(m_sizes.numPrimives * sizeof(i32));
			cudaMemcpy(m_sizes.primIds, m_inputCUDA.primIds, m_sizes.numPrimives * sizeof(i32), cudaMemcpyDefault);*/
	/*
		}
	}

	m_device = m_device | res;
}

void LBVH::unload_resident(Device res)
{
	if ((res & Device::CPU) != Device::NONE) {
		if (is_resident(Device::CPU)) {
			// Here m_inputCPU is not set to {},
			// since if we want to make_resident on cpu,
			// we need its info.
			mAssert(m_inputCPU.triIndices != nullptr);
			free(m_inputCPU.triIndices);
			m_inputCPU.triIndices = nullptr;
			m_inputCPU.quadIndices = nullptr;

			mAssert(m_outputCPU.bvh != nullptr);
			mAssert(m_outputCPU.primIds != nullptr);
			free(m_outputCPU.bvh);
			free(m_outputCPU.primIds);
			m_outputCPU = {};
		}

		if (!is_resident(Device::CUDA))
			m_sizes = {};
	}

	if ((res & Device::CUDA) != Device::NONE) {
		if (is_resident(Device::CUDA)) {
			mAssert(m_inputCUDA.triIndices != nullptr);
			cudaFree(m_inputCUDA.triIndices);
			m_inputCUDA = {};

			mAssert(m_outputCUDA.bvh != nullptr);
			mAssert(m_outputCUDA.primIds != nullptr);
			cudaFree(m_outputCUDA.bvh);
			cudaFree(m_outputCUDA.primIds);
			m_outputCUDA = {};
		}

		if (!is_resident(Device::CPU))
			m_sizes = {};
	}

	m_device = m_device & (~res);
}

void LBVH::build(const std::vector<InstanceHandle>& scene)
{
	for (auto iter: scene)
	{
		//TODO, how to get ObjectData?
		iter->get_bounding_box();
	}
}

void LBVH::build(ObjectData data)
{
	auto faces = data.faces;
	auto& spheres = data.spheres;
	auto& faceVertices = data.faceVertices;

	const i32 numTriangles = static_cast<i32>(data.triangleCount);
	const i32 numQuads = static_cast<i32>(data.quadCount);
	const i32 numSpheres = static_cast<i32>(spheres.get_elem_size());

	// Get index values.
	i32 numPolygons = 0;
	i32 triCount = 0;
	i32 quadCount = 0;
	i32* triIndices = (i32*)malloc((numTriangles * 3 + numQuads * 4) * sizeof(i32));
	i32* quadIndices = triIndices + numTriangles * 3;
	for (auto face = faces.cbegin(); face != faces.cend(); ++face)
	{
		if (3 == face.get_vertex_count()) {
			int id = 3 * triCount;
			for (auto v : *face) {
				triIndices[id++] = v.idx();
			}
			++triCount;
		}
		else if (4 == face.get_vertex_count()) {
			int id = 4 * quadCount;
			for (auto v : *face) {
				triIndices[id++] = v.idx();
			}
			++quadCount;
		}
		else {
			throw std::runtime_error("Face is neither triangle nor quad!\n");
		}
		++numPolygons;
	}
	mAssert(numTriangles == triCount);
	mAssert(numQuads == quadCount);
	mAssert(numPolygons == triCount + quadCount);

	// Set m_sizes.
	const i32 pointsByteCount = static_cast<const i32>(faceVertices.get_byte_count());
	const i32 numVertices = static_cast<const i32>(faceVertices.get_size());
	m_sizes.offsetQuads = numTriangles;
	m_sizes.offsetSpheres = numTriangles + numQuads;
	m_sizes.numPrimives = numPolygons + numSpheres;
	m_sizes.numVertices = numVertices;

	// Set m_inputCPU.
	m_inputCPU.meshVertices = (ei::Vec3*)(faceVertices.aquireConst<Device::CPU>());
	// m_infoCPU.meshUVs // TODO adjusted it.
	m_inputCPU.meshUVs = (ei::Vec2*)malloc(m_sizes.numVertices * sizeof(ei::Vec2));
	m_inputCPU.triIndices = triIndices;
	m_inputCPU.quadIndices = quadIndices;
	m_inputCPU.spheres = (ei::Vec4*)(spheres.aquireConst<Device::CPU>());

	// Copy arrays into m_inputCUDA.
	// Allocate memory.
	i32 totalCudaMemInBytes = (numTriangles * 3 + numQuads * 4) * sizeof(i32) // For indices.
		+ numSpheres * sizeof(ei::Vec4) // For spheres.
		+ numVertices * (sizeof(ei::Vec3) + sizeof(ei::Vec2));// For vertices and uvs.
	cudaMalloc((void**)&m_inputCUDA.triIndices, totalCudaMemInBytes);
	m_inputCUDA.quadIndices = triIndices + numTriangles * 3;
	m_inputCUDA.spheres = (ei::Vec4*)(m_inputCUDA.quadIndices + numQuads * 4);
	m_inputCUDA.meshVertices = (ei::Vec3*)(m_inputCUDA.spheres + numSpheres);
	m_inputCUDA.meshUVs = (ei::Vec2*)(m_inputCUDA.meshVertices + numVertices);
	// Copy to cuda array.
	cudaMemcpy(m_inputCUDA.triIndices, m_inputCPU.triIndices,
		(numTriangles * 3 + numQuads * 4) * sizeof(i32), cudaMemcpyDefault);
	cudaMemcpy(m_inputCUDA.spheres, m_inputCPU.spheres, numSpheres * sizeof(ei::Vec4), cudaMemcpyDefault);
	cudaMemcpy(m_inputCUDA.meshVertices, m_inputCPU.meshVertices,
		numVertices * sizeof(ei::Vec3), cudaMemcpyDefault);
	cudaMemcpy(m_inputCUDA.meshUVs, m_inputCPU.meshUVs,
		numVertices * sizeof(ei::Vec2), cudaMemcpyDefault);

	// Build lbvh.
	ei::Vec4 traverseCost = { 0.8f, 1.2f, 2.4f, 1.f };
	build_lbvh64_info(m_sizes, m_inputCUDA, m_outputCUDA, data.aabb, traverseCost);
	
	// Mark the device tag.
	m_device = Device::CUDA;
}

bool LBVH::is_dirty(Device res) const
{
	return false;
}
*/
}}}
