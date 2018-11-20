#include "lbvh.hpp"
#include "core/scene/object.hpp"
#include "build_lbvh.hpp"

namespace mufflon {	namespace scene {namespace accel_struct {


LBVH::LBVH():
	m_device(Device::NONE),
	m_collapsedBVH_CPU(nullptr),
	m_collapsedBVH_CUDA(nullptr),
	m_sizeCollapsedBVH(0u)
{
}

LBVH::~LBVH()
{
	if (m_collapsedBVH_CPU)
		free(m_collapsedBVH_CPU);

	if (m_collapsedBVH_CUDA)
		cudaFree(m_collapsedBVH_CUDA);
}

bool LBVH::is_resident(Device res) const
{
	return static_cast<bool>(m_device & res);
}

void LBVH::make_resident(Device res)
{
}

void LBVH::unload_resident(Device res)
{
	if ((res & Device::CPU) != Device::NONE) {
		if (is_resident(Device::CPU)) {
			free(m_collapsedBVH_CPU);
			m_sizeCollapsedBVH = 0u;
		}
	}

	if ((res & Device::CUDA) != Device::NONE) {
		if (is_resident(Device::CUDA)) {
			cudaFree(m_collapsedBVH_CUDA);
			m_sizeCollapsedBVH = 0u;
		}
	}
}

void LBVH::build(const std::vector<InstanceHandle>&)
{
}

void LBVH::build(ObjectData data)
{
	auto faces = data.faces;
	auto& spheres = data.spheres;
	auto& faceVertices = data.faceVertices;

	i32 numVertices = static_cast<i32>(faces.begin().get_vertex_count());

	i32 numTriangles = static_cast<i32>(data.triangleCount);
	i32 numQuads = static_cast<i32>(data.quadCount);
	i32 numSpheres = static_cast<i32>(spheres.get_elem_size());

	// TODO determine where to store the data of vertices and indices.

	i32 numPolygons = 0u;
	for (auto var : faces)
	{
		// TODO: How to get mesh data? 
		// At present only ConstFaceVertexRange and ConstFaceIter can be got by the
		// FaceIterator.
		//geometry::Polygons::FaceIterator fit = faces.begin();
		//fit->handle(); // Function handle is undefined.
		if (numPolygons < numTriangles) {
			// Add triangles.
			
		}
		else {
			// Add quads.


		}
	}

	//build_lbvh64();
}

bool LBVH::is_dirty(Device res) const
{
	return false;
}

}}}
