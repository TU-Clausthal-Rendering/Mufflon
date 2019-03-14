#include "max_one_decimater.hpp"
#include "util/assert.hpp"
#include <climits>

namespace mufflon::scene::decimation {

MaxOneDecimater::MaxOneDecimater(Mesh& mesh) :
	OpenMesh::Decimater::BaseDecimaterT<geometry::PolygonMeshType>(mesh),
	m_mesh(mesh)
{
	m_mesh.add_property(m_collapseTarget);
	m_mesh.add_property(m_priority);
	m_mesh.add_property(m_heapPosition);
}

MaxOneDecimater::~MaxOneDecimater() {
	m_mesh.remove_property(m_collapseTarget);
	m_mesh.remove_property(m_priority);
	m_mesh.remove_property(m_heapPosition);
}

// Taken from OpenMesh::Decimater::DecimaterT
void MaxOneDecimater::heap_vertex(const MaxOneDecimater::Mesh::VertexHandle vh) {
	float bestPriority = std::numeric_limits<float>::max();
	typename Mesh::HalfedgeHandle collapseTarget;

	// find best target in one ring
	for(auto vohIter = m_mesh.voh_begin(vh); vohIter.is_valid(); ++vohIter) {
		const auto heh = *vohIter;
		CollapseInfo ci(m_mesh, heh);

		if(this->is_collapse_legal(ci)) {
			const float priority = this->collapse_priority(ci);
			if(priority >= 0.f && priority < bestPriority) {
				bestPriority = priority;
				collapseTarget = heh;
			}
		}
	}

	if(collapseTarget.is_valid()) {
		// target found -> put vertex on heap
		m_mesh.property(m_collapseTarget, vh) = collapseTarget;
		m_mesh.property(m_priority, vh) = bestPriority;

		if(m_heap->is_stored(vh))
			m_heap->update(vh);
		else
			m_heap->insert(vh);
	} else {
		// not valid -> remove from heap
		if(m_heap->is_stored(vh))
			m_heap->remove(vh);

		m_mesh.property(m_collapseTarget, vh) = collapseTarget;
		m_mesh.property(m_priority, vh) = Module::ILLEGAL_COLLAPSE;
	}
}

// Mostly taken from OpenMesh::Decimater::DecimaterT
std::size_t MaxOneDecimater::decimate(const std::size_t nCollapses) {
	if(!this->is_initialized())
		return 0u;

	std::vector<typename Mesh::VertexHandle> support(15u);
	std::size_t currCollapses = 0u;
	const std::size_t desiredCollapses = (nCollapses == 0u) ? m_mesh.n_vertices() : nCollapses;

	// initialize heap
	m_heap = std::make_unique<DeciHeap>(HeapInterface(m_mesh, m_priority, m_heapPosition));


	m_heap->reserve(m_mesh.n_vertices());

	for(auto vertex : m_mesh.vertices()) {
		m_heap->reset_heap_position(vertex);
		if(!m_mesh.status(vertex).deleted())
			this->heap_vertex(vertex);
	}

	const bool updateNormals = m_mesh.has_face_normals();

	// process heap
	while((!m_heap->empty()) && (currCollapses < desiredCollapses)) {
		// get 1st heap entry
		const auto vp = m_heap->front();
		const auto v0v1 = m_mesh.property(m_collapseTarget, vp);
		m_heap->pop_front();

		// Check if the collapse has been invalidated
		if(!v0v1.is_valid())
			continue;

		// setup collapse info
		CollapseInfo ci(m_mesh, v0v1);

		// check topological correctness AGAIN !
		if(!this->is_collapse_legal(ci))
			continue;

		// store support (= one ring of *vp)
		support.clear();
		for(auto vvIter = m_mesh.vv_iter(ci.v0); vvIter.is_valid(); ++vvIter)
			support.push_back(*vvIter);

		// pre-processing
		this->preprocess_collapse(ci);

		// perform collapse
		m_mesh.collapse(v0v1);
		++currCollapses;

		if(updateNormals) {
			// update triangle normals
			for(auto vfIter = m_mesh.vf_iter(ci.v1); vfIter.is_valid(); ++vfIter) {
				if(!m_mesh.status(*vfIter).deleted())
					m_mesh.set_normal(*vfIter, m_mesh.calc_face_normal(*vfIter));
			}
		}

		// post-process collapse
		this->postprocess_collapse(ci);

		// update heap (former one ring of decimated vertex)
		for(auto supportVertex : support) {
			mAssert(!m_mesh.status(supportVertex).deleted());
			this->heap_vertex(supportVertex);
		}

		// Effectively remove the vertex we collapsed to (since we only want vertices to move by one collapse per decimation call)
		m_mesh.property(m_collapseTarget, ci.v1).invalidate();

		// notify observer and stop if the observer requests it
		if(!this->notify_observer(currCollapses))
			return currCollapses;
	}

	// delete heap
	m_heap.reset();

	// DON'T do garbage collection here! It's up to the application.
	return currCollapses;
}

} // namespace mufflon::scene::decimation