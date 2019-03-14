#pragma once

#include "core/scene/geometry/polygon_mesh.hpp"
#include <OpenMesh/Tools/Utils/HeapT.hh>
#include <OpenMesh/Tools/Decimater/BaseDecimaterT.hh>
#include <memory>

namespace mufflon::scene::decimation {

class MaxOneDecimater : virtual public OpenMesh::Decimater::BaseDecimaterT<geometry::PolygonMeshType> {
public:
	MaxOneDecimater(Mesh& mesh);
	~MaxOneDecimater();

	std::size_t decimate(const std::size_t nCollapses);

private:
	// Interface for the vertex heap
	class HeapInterface {
	public:
		HeapInterface(Mesh& mesh, OpenMesh::VPropHandleT<float> priority,
					  OpenMesh::VPropHandleT<int> heapPosition) :
			m_mesh(mesh),
			m_priority(priority),
			m_heapPosition(heapPosition) {}

		bool less(Mesh::VertexHandle _vh0, Mesh::VertexHandle _vh1) {
			return m_mesh.property(m_priority, _vh0) < m_mesh.property(m_priority, _vh1);
		}

		inline bool
			greater(Mesh::VertexHandle _vh0, Mesh::VertexHandle _vh1) {
			return m_mesh.property(m_priority, _vh0) > m_mesh.property(m_priority, _vh1);
		}

		inline int
			get_heap_position(Mesh::VertexHandle _vh) {
			return m_mesh.property(m_heapPosition, _vh);
		}

		inline void
			set_heap_position(Mesh::VertexHandle _vh, int _pos) {
			m_mesh.property(m_heapPosition, _vh) = _pos;
		}


	private:
		Mesh& m_mesh;
		OpenMesh::VPropHandleT<float> m_priority;
		OpenMesh::VPropHandleT<int> m_heapPosition;
	};

	using DeciHeap = OpenMesh::Utils::HeapT<OpenMesh::VertexHandle, HeapInterface>;

	// Add a vertex to the heap
	void heap_vertex(const Mesh::VertexHandle vh);

	Mesh& m_mesh;
	std::unique_ptr<DeciHeap> m_heap = nullptr;
	OpenMesh::VPropHandleT<Mesh::HalfedgeHandle> m_collapseTarget;
	OpenMesh::VPropHandleT<float> m_priority;
	OpenMesh::VPropHandleT<int> m_heapPosition;
};

} // namespace mufflon::scene::decimation