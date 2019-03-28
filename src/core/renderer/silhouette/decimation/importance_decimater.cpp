#pragma once

#include "importance_decimater.hpp"
#include "util/punning.hpp"
#include "modules/collapse_tracker.hpp"
#include "modules/convex_edge.hpp"
#include "modules/importance.hpp"
#include "modules/normal_deviation.hpp"
#include "modules/silhouette.hpp"
#include <OpenMesh/Tools/Decimater/DecimaterT.hh>
#include <OpenMesh/Tools/Decimater/ModQuadricT.hh>

namespace mufflon::renderer::silhouette::decimation {

using namespace scene;
using namespace scene::geometry;
using namespace modules;

namespace {

template < class T >
inline void atomic_add(std::atomic<T>& atom, const T& val) {
	float expected = atom.load();
	float desired;
	do {
		desired = expected + val;
	} while(!atom.compare_exchange_weak(expected, desired));
}

inline float compute_area(const PolygonMeshType& mesh, const OpenMesh::VertexHandle vertex) {
	// Important: only works for triangles!
	float area = 0.f;
	const auto center = mesh.point(vertex);
	auto circIter = mesh.cvv_ccwbegin(vertex);
	OpenMesh::Vec3f a = mesh.point(*circIter);
	OpenMesh::Vec3f b;
	++circIter;
	for(; circIter.is_valid(); ++circIter) {
		b = a;
		a = mesh.point(*circIter);
		area += ((a - center) % (b - center)).length();
	}
	return 0.5f * area;
}

} // namespace

ImportanceDecimater::ImportanceDecimater(Lod& original, Lod& decimated,
										 const Degrees maxNormalDeviation,
										 const CollapseMode mode,
										 const std::size_t initialCollapses) :
	m_original(original),
	m_decimated(decimated),
	m_originalPoly(m_original.template get_geometry<Polygons>()),
	m_decimatedPoly(&m_decimated.template get_geometry<Polygons>()),
	m_originalMesh(m_originalPoly.get_mesh()),
	m_decimatedMesh(&m_decimatedPoly->get_mesh()),
	m_importance(nullptr),
	m_maxNormalDeviation(maxNormalDeviation),
	m_collapseMode(mode)
{
	// Add necessary properties
	m_decimatedMesh->add_property(m_originalVertex);
	m_decimatedMesh->add_property(m_importanceDensity);
	m_originalMesh.add_property(m_collapsedTo);
	m_originalMesh.add_property(m_collapsed);
	m_originalMesh.add_property(m_silhouette);
	m_decimatedMesh->request_vertex_status();
	m_decimatedMesh->request_edge_status();
	m_decimatedMesh->request_halfedge_status();
	m_decimatedMesh->request_face_status();

	// Set the original vertices for the (to be) decimated mesh
	for(auto vertex : m_decimatedMesh->vertices())
		m_decimatedMesh->property(m_originalVertex, vertex) = vertex;
	for(auto vertex : m_originalMesh.vertices()) {
		m_originalMesh.property(m_collapsed, vertex) = false;
		// Valid, since no decimation has been performed yet
		m_originalMesh.property(m_collapsedTo, vertex) = vertex;
	}

	// Perform initial decimation
	if(initialCollapses > 0u) {
		auto decimater = m_decimatedPoly->create_decimater();
		OpenMesh::Decimater::ModQuadricT<scene::geometry::PolygonMeshType>::Handle modQuadricHandle;
		CollapseTrackerModule<>::Handle trackerHandle;
		NormalDeviationModule<>::Handle normalHandle;
		ConvexDecimationModule<>::Handle convexHandle;
		decimater.add(modQuadricHandle);
		decimater.add(trackerHandle);
		decimater.add(normalHandle);
		decimater.module(trackerHandle).set_properties(m_originalMesh, m_collapsed, m_collapsedTo);
		decimater.module(normalHandle).set_max_deviation(m_maxNormalDeviation);
		if(m_collapseMode == CollapseMode::NO_CONCAVE)
			decimater.add(convexHandle);
		// Possibly repeat until we reached the desired count
		const std::size_t targetCollapses = std::min(initialCollapses, m_originalPoly.get_vertex_count());
		const std::size_t targetVertexCount = m_originalPoly.get_vertex_count() - targetCollapses;
		std::size_t performedCollapses = 0u;
		do {
			performedCollapses += m_decimatedPoly->decimate(decimater, targetVertexCount, false);
		} while(performedCollapses < targetCollapses);
		m_decimatedPoly->garbage_collect([this](Mesh::VertexHandle deletedVertex, Mesh::VertexHandle changedVertex) {
			// Adjust the reference from original to decimated mesh
			const auto originalVertex = this->get_original_vertex_handle(changedVertex);
			if(!m_originalMesh.property(m_collapsed, originalVertex))
				m_originalMesh.property(m_collapsedTo, originalVertex) = deletedVertex;
		});

		logPedantic("Initial mesh decimation (", m_decimatedPoly->get_vertex_count(), "/", m_originalPoly.get_vertex_count(), ")");
	}

	// Initialize importance map
	m_importance = std::make_unique<std::atomic<float>[]>(m_decimatedPoly->get_vertex_count());
	for(std::size_t i = 0u; i < m_decimatedPoly->get_vertex_count(); ++i)
		m_importance[i].store(0.f);
}

ImportanceDecimater::ImportanceDecimater(ImportanceDecimater&& dec) :
	m_original(dec.m_original),
	m_decimated(dec.m_decimated),
	m_originalPoly(dec.m_originalPoly),
	m_decimatedPoly(dec.m_decimatedPoly),
	m_originalMesh(dec.m_originalMesh),
	m_decimatedMesh(dec.m_decimatedMesh),
	m_importance(std::move(dec.m_importance)),
	m_originalVertex(dec.m_originalVertex),
	m_importanceDensity(dec.m_importanceDensity),
	m_collapsedTo(dec.m_collapsedTo),
	m_collapsed(dec.m_collapsed),
	m_silhouette(dec.m_silhouette),
	m_maxNormalDeviation(dec.m_maxNormalDeviation),
	m_collapseMode(dec.m_collapseMode)
{
	// Request the status again since it will get removed once in the destructor
	m_decimatedMesh->request_vertex_status();
	m_decimatedMesh->request_edge_status();
	m_decimatedMesh->request_halfedge_status();
	m_decimatedMesh->request_face_status();
	// Invalidate the handles here so we know not to remove them in the destructor
	dec.m_originalVertex.invalidate();
	dec.m_importanceDensity.invalidate();
	dec.m_collapsedTo.invalidate();
	dec.m_collapsed.invalidate();
	dec.m_silhouette.invalidate();
}

ImportanceDecimater::~ImportanceDecimater() {
	if(m_originalVertex.is_valid())
		m_decimatedMesh->remove_property(m_originalVertex);
	if(m_importanceDensity.is_valid())
		m_decimatedMesh->remove_property(m_importanceDensity);
	if(m_collapsedTo.is_valid())
		m_originalMesh.remove_property(m_collapsedTo);
	if(m_collapsed.is_valid())
		m_originalMesh.remove_property(m_collapsed);
	if(m_silhouette.is_valid())
		m_originalMesh.remove_property(m_silhouette);

	m_decimatedMesh->release_vertex_status();
	m_decimatedMesh->release_edge_status();
	m_decimatedMesh->release_halfedge_status();
	m_decimatedMesh->release_face_status();
}

void ImportanceDecimater::udpate_importance_density() {
	// Update our statistics: the importance density of each vertex
	double importanceSum = 0.0;
	for(const auto vertex : m_decimatedMesh->vertices()) {
		const float importance = m_importance[vertex.idx()].load();
		mAssert(!isnan(importance));
		importanceSum += importance;

		// Important: only works for triangles!
		const float area = compute_area(*m_decimatedMesh, vertex);

		mAssertMsg(area != 0.f, "Degenerated vertex");
		m_decimatedMesh->property(m_importanceDensity, vertex) = importance / area;
	}
}

void ImportanceDecimater::iterate(const std::size_t minVertexCount, const float threshold, const float reduction) {
	// Add the temporary density storage to the original mesh
	OpenMesh::VPropHandleT<float> tempImportanceDensity;
	m_originalMesh.add_property(tempImportanceDensity);

	// Map the importance back to the original mesh
	for(auto vertex : m_originalMesh.vertices()) {
		// Traverse collapse chain and snatch importance
		auto v = vertex;
		while(m_originalMesh.property(m_collapsed, v))
			v = m_originalMesh.property(m_collapsedTo, v);

		// Put importance into temporary storage
		m_originalMesh.property(tempImportanceDensity, vertex) = m_decimatedMesh->property(m_importanceDensity, m_originalMesh.property(m_collapsedTo, v));
	}

	// Reset the collapse property
	for(auto vertex : m_originalMesh.vertices()) {
		m_originalMesh.property(m_collapsed, vertex) = false;
		m_originalMesh.property(m_collapsedTo, vertex) = vertex;
	}

	// Recreate the LoD for decimation
	m_decimated.~Lod();
	new(&m_decimated) Lod(m_original);
	// Refetch the affected pointers
	m_decimatedPoly = &m_decimated.template get_geometry<geometry::Polygons>();
	m_decimatedMesh = &m_decimatedPoly->get_mesh();
	// Re-request properties
	m_decimatedMesh->add_property(m_originalVertex);
	m_decimatedMesh->add_property(m_importanceDensity);
	m_decimatedMesh->request_vertex_status();
	m_decimatedMesh->request_edge_status();
	m_decimatedMesh->request_halfedge_status();
	m_decimatedMesh->request_face_status();

	// Set the original vertices for the (to be) decimated mesh
	for(auto vertex : m_decimatedMesh->vertices())
		m_decimatedMesh->property(m_originalVertex, vertex) = vertex;

	// Perform decimation
	auto decimater = m_decimatedPoly->create_decimater();
	CollapseTrackerModule<>::Handle trackerHandle;
	NormalDeviationModule<>::Handle normalHandle;
	ImportanceDecimationModule<>::Handle impHandle;
	ConvexDecimationModule<>::Handle convexHandle;
	SilhouetteDecimationModule<>::Handle silhouetteHandle;
	decimater.add(trackerHandle);
	decimater.add(normalHandle);
	decimater.add(impHandle);
	decimater.module(trackerHandle).set_properties(m_originalMesh, m_collapsed, m_collapsedTo);
	decimater.module(normalHandle).set_max_deviation(m_maxNormalDeviation);
	decimater.module(impHandle).set_properties(m_originalMesh, tempImportanceDensity, threshold);
	if(m_collapseMode == CollapseMode::NO_CONCAVE) {
		decimater.add(convexHandle);
	} else if(m_collapseMode == CollapseMode::NO_SILHOUETTE) {
		decimater.add(silhouetteHandle);
		decimater.module(silhouetteHandle).set_properties(m_originalMesh, m_silhouette);
	}
	
	const std::size_t targetCount = (reduction == 0.f) ? 0u : static_cast<std::size_t>((1.f - reduction) * m_originalPoly.get_vertex_count());
	const auto collapses = m_decimatedPoly->decimate(decimater, targetCount, false);
	m_decimatedPoly->garbage_collect([this](Mesh::VertexHandle deletedVertex, Mesh::VertexHandle changedVertex) {
		// Adjust the reference from original to decimated mesh
		const auto originalVertex = this->get_original_vertex_handle(changedVertex);
		if(!m_originalMesh.property(m_collapsed, originalVertex))
			m_originalMesh.property(m_collapsedTo, originalVertex) = deletedVertex;
	});

	// Initialize importance map
	m_importance = std::make_unique<std::atomic<float>[]>(m_decimatedPoly->get_vertex_count());
	for(std::size_t i = 0u; i < m_decimatedPoly->get_vertex_count(); ++i) {
		m_importance[i] = m_originalMesh.property(tempImportanceDensity, m_decimatedMesh->property(m_originalVertex, m_decimatedMesh->vertex_handle((u32)i)));
	}
		//m_importance[i].store(0.f);
	// TODO: take over the old importance!

	// Cleanup
	m_originalMesh.remove_property(tempImportanceDensity);

	logPedantic("Performed ", collapses, " collapses, remaining vertices: ", m_decimatedMesh->n_vertices());

	// Update the damping factor
	// TODO: use parameter
	m_damping *= 0.925f;

	/*if(m_decimatedMesh->n_vertices() > minVertexCount) {
		collapses = this->collapse(threshold);
	}
	// ....then we undecimate (although the order shouldn't play a role)
	const std::size_t uncollapses = this->uncollapse(threshold);
	logPedantic("Performed ", collapses, " collapses, ", uncollapses, " uncollapses");
	logPedantic("Remaining vertices: ", m_decimatedMesh->n_vertices());*/
}

#if 0
std::size_t ImportanceDecimater::collapse(const float threshold) {

	// Setup - we don't need these to persist
	m_decimatedMesh->add_property(m_collapseTarget);
	m_decimatedMesh->add_property(m_priority);
	m_decimatedMesh->add_property(m_heapPosition);

	std::vector<typename Mesh::VertexHandle> support(15u);
	std::size_t currCollapses = 0u;

	// initialize heap
	m_heap = std::make_unique<OpenMesh::Utils::HeapT<Mesh::VertexHandle, HeapInterface>>(HeapInterface(m_decimatedMesh, m_priority, m_heapPosition));

	for(auto vertex : m_decimatedMesh->vertices()) {
		m_heap->reset_heap_position(vertex);
		if(!m_decimatedMesh->status(vertex).deleted())
			this->add_vertex_collapse(vertex, threshold);
	}

	// process heap
	while((!m_heap->empty())) {
		// get 1st heap entry
		const auto vp = m_heap->front();
		const auto v0v1 = m_decimatedMesh->property(m_collapseTarget, vp);
		m_heap->pop_front();

		// Check if the collapse has been invalidated
		if(!v0v1.is_valid())
			continue;

		// setup collapse info
		CollapseInfo ci(m_decimatedMesh, v0v1);

		// check topological correctness AGAIN !
		if(!this->is_collapse_legal(ci))
			continue;

		// store support (= one ring of *vp)
		support.clear();
		for(auto vvIter = m_decimatedMesh->vv_iter(ci.v0); vvIter.is_valid(); ++vvIter)
			support.push_back(*vvIter);

		// perform collapse
		m_decimatedMesh->collapse(v0v1);
		++currCollapses;

		// Update the collapse properties
		const auto originalV0 = get_original_vertex_handle(ci.v0);
		const auto originalV1 = get_original_vertex_handle(ci.v1);
		m_originalMesh.property(m_collapsed, originalV0) = true;
		m_originalMesh.property(m_collapsedTo, originalV0) = CollapseHistory{
			originalV1, get_original_vertex_handle(ci.vl), get_original_vertex_handle(ci.vr)
		};

		// update heap (former one ring of decimated vertex)
		for(auto supportVertex : support) {
			mAssert(!m_decimatedMesh->status(supportVertex).deleted());
			this->add_vertex_collapse(supportVertex, threshold);
		}

		// Effectively remove the vertex we collapsed to (since we only want vertices to move by one collapse per decimation call)
		m_decimatedMesh->property(m_collapseTarget, ci.v1).invalidate();
	}

	// delete heap
	m_heap.reset();

	// Rebuild the index buffer and perform garbage collection
	m_decimatedMesh->remove_property(m_collapseTarget);
	m_decimatedMesh->remove_property(m_priority);
	m_decimatedMesh->remove_property(m_heapPosition);

	m_decimatedPoly->garbage_collect([this](Mesh::VertexHandle deletedVertex, Mesh::VertexHandle changedVertex) {
		// Adjust the reference from original to decimated mesh
		const auto originalVertex = this->get_original_vertex_handle(changedVertex);
		if(!m_originalMesh.property(m_collapsed, originalVertex))
			m_originalMesh.property(m_collapsedTo, originalVertex).v1 = deletedVertex;
	});
	m_decimated.clear_accel_structure();

	return currCollapses;
}

std::size_t ImportanceDecimater::uncollapse(const float threshold) {
	// Cache for new importance + densities for the participating vertices
	// Order: vl -> ... -> vr -> v1
	static thread_local std::vector<std::pair<float, float>> newDensities;

	std::size_t uncollapses = 0u;

	// Go through all vertices and check those that got collapsed somewhere
	for(const auto vertex : m_originalMesh.vertices()) {
		// Weed out vertices that are not collapsed
		if(!m_originalMesh.property(m_collapsed, vertex))
			continue;

		// Check if the collapsed-to vertex is also collapsed
		const auto history = m_originalMesh.property(m_collapsedTo, vertex);
		if(m_originalMesh.property(m_collapsed, history.v1))
			continue;

		// Check if a connection exists in the original mesh
		// TODO!

		// Query the vertex handle of the collapse target in the decimated mesh
		const auto decimatedV1 = m_originalMesh.property(m_collapsedTo, history.v1).v1;
		mAssert(decimatedV1.is_valid());

		// Determine the left and right vertices from the collapse (only works for triangles!)

		// Only works with triangles!
		auto vl = history.vl;
		auto vr = history.vr;
		mAssert(m_originalMesh.find_halfedge(vertex, vl).is_valid());
		mAssert(m_originalMesh.find_halfedge(vertex, vr).is_valid());

		while(m_originalMesh.property(m_collapsed, vl))
			vl = m_originalMesh.property(m_collapsedTo, vl).v1;
		while(m_originalMesh.property(m_collapsed, vr))
			vr = m_originalMesh.property(m_collapsedTo, vr).v1;
		const auto decimatedVl = m_originalMesh.property(m_collapsedTo, vl).v1;
		const auto decimatedVr = m_originalMesh.property(m_collapsedTo, vr).v1;
		mAssert(decimatedVl.is_valid());
		mAssert(decimatedVr.is_valid());

		const auto v1vl = m_decimatedMesh->find_halfedge(decimatedV1, decimatedVl);
		const auto v1vr = m_decimatedMesh->find_halfedge(decimatedV1, decimatedVr);
		if(!v1vl.is_valid() || !v1vr.is_valid()) {
			//std::cout << "No edge" << std::endl;
			// If we cannot see our local collapse vertices, even after respecting their collapses,
			// then the mesh has locally changed too much and we cannot restore the vertex right now
			continue;
		}

		// Check if one of the three cornerstones for the collapse are fused together
		if(decimatedVl == decimatedVr || decimatedVl == decimatedV1 || decimatedVr == decimatedV1) {
			//std::cout << "Collapsed to one point" << std::endl;
			continue;
		}

		if(float v0Imp = compute_new_importance_densities(newDensities, vertex, decimatedV1, decimatedVl, decimatedVr, threshold); v0Imp >= 0.f) {
			// Adjust the importance of the vertices
			/*std::size_t i = 0u;
			for(auto heh = m_decimatedMesh->find_halfedge(decimatedV1, decimatedVl); m_decimatedMesh->to_vertex_handle(heh) != decimatedVr;
				heh = m_decimatedMesh->opposite_halfedge_handle(m_decimatedMesh->prev_halfedge_handle(heh))) {
				const auto vb = m_decimatedMesh->to_vertex_handle(heh);
				m_importance[get_original_vertex_handle(vb).idx()].store(newDensities[i].first);
				m_decimatedMesh->property(m_importanceDensity, vb) = newDensities[i].second;
				++i;
			}
			// VR and V1 need special treatment...
			m_importance[vr.idx()].store(newDensities[i].first);
			m_decimatedMesh->property(m_importanceDensity, decimatedVr) = newDensities[i].second;
			m_importance[history.v1.idx()].store(newDensities.back().first);
			m_decimatedMesh->property(m_importanceDensity, decimatedV1) = newDensities.back().second;*/

			// Insert V0 into the decimated LoD
			const auto point = m_originalPoly.template acquire_const<Device::CPU, ei::Vec3>(m_originalPoly.get_points_hdl())[vertex.idx()];
			const auto normal = m_originalPoly.template acquire_const<Device::CPU, ei::Vec3>(m_originalPoly.get_normals_hdl())[vertex.idx()];
			const auto uv = m_originalPoly.template acquire_const<Device::CPU, ei::Vec2>(m_originalPoly.get_uvs_hdl())[vertex.idx()];
			const auto v0Hdl = m_decimatedPoly->add(point, normal, uv);
			// Split the vertex
			auto faces = m_decimatedPoly->vertex_split(v0Hdl, decimatedV1, decimatedVl, decimatedVr);
			// Set the material indices of the two recreated faces
			auto* matIndices = m_decimatedPoly->template acquire<Device::CPU, MaterialIndex>(m_decimatedPoly->get_material_indices_hdl());
			const auto matIdx = matIndices[m_decimatedMesh->face_handle(m_decimatedMesh->halfedge_handle(decimatedV1)).idx()];
			if(faces.first.is_valid())
				matIndices[faces.first.idx()] = matIdx;
			if(faces.second.is_valid())
				matIndices[faces.first.idx()] = matIdx;
			m_decimatedPoly->mark_changed(Device::CPU, m_decimatedPoly->get_material_indices_hdl());

			// Adjust collapsed sequence
			m_originalMesh.property(m_collapsed, vertex) = false;
			m_originalMesh.property(m_collapsedTo, vertex).v1 = v0Hdl;
			m_decimatedMesh->property(m_originalVertex, v0Hdl) = vertex;

			// Count up the uncollapse, if necessary
			if(m_collapseMode == CollapseMode::NO_CONCAVE_AFTER_UNCOLLAPSE) {
				const auto v0v1 = m_decimatedMesh->find_halfedge(v0Hdl, decimatedV1);
				m_decimatedMesh->property(m_uncollapsed, v0v1) = true;
			}


			++uncollapses;
		} else {
			//std::cout << "Failed density" << std::endl;
		}
	}

	return uncollapses;
}

bool ImportanceDecimater::is_collapse_legal(const OpenMesh::Decimater::CollapseInfoT<Mesh>& ci) const {
	// locked ?
	mAssert(static_cast<std::size_t>(ci.v0.idx()) < m_decimatedMesh->n_vertices());
	mAssert(static_cast<std::size_t>(ci.v1.idx()) < m_decimatedMesh->n_vertices());

	if(m_decimatedMesh->status(ci.v0).locked())
		return false;

	// this test checks:
	// is v0v1 deleted?
	// is v0 deleted?
	// is v1 deleted?
	// are both vlv0 and v1vl boundary edges?
	// are both v0vr and vrv1 boundary edges?
	// are vl and vr equal or both invalid?
	// one ring intersection test
	// edge between two boundary vertices should be a boundary edge
	if(!m_decimatedMesh->is_collapse_ok(ci.v0v1))
		return false;

	if(ci.vl.is_valid() && ci.vr.is_valid()
	   && m_decimatedMesh->find_halfedge(ci.vl, ci.vr).is_valid()
	   && m_decimatedMesh->valence(ci.vl) == 3u && m_decimatedMesh->valence(ci.vr) == 3u) {
		return false;
	}
	//--- feature test ---

	if(m_decimatedMesh->status(ci.v0).feature()
	   && !m_decimatedMesh->status(m_decimatedMesh->edge_handle(ci.v0v1)).feature())
		return false;

	//--- test boundary cases ---
	if(m_decimatedMesh->is_boundary(ci.v0)) {

		// don't collapse a boundary vertex to an inner one
		if(!m_decimatedMesh->is_boundary(ci.v1))
			return false;

		// only one one ring intersection
		if(ci.vl.is_valid() && ci.vr.is_valid())
			return false;
	}

	// there have to be at least 2 incident faces at v0
	if(m_decimatedMesh->cw_rotated_halfedge_handle(
		m_decimatedMesh->cw_rotated_halfedge_handle(ci.v0v1)) == ci.v0v1)
		return false;

	// collapse passed all tests -> ok
	return true;
}

float ImportanceDecimater::collapse_priority(const OpenMesh::Decimater::CollapseInfoT<Mesh>& ci) {
	// Check normal, concavity
	if(!check_normal_deviation(ci))
		return -1.f;


	if(m_collapseMode == CollapseMode::NO_CONCAVE && !is_convex_collapse(ci))
		return -1.f;

	if(m_collapseMode == CollapseMode::NO_CONCAVE_AFTER_UNCOLLAPSE) {
		// Find the original halfedge and check if it has been collapsed before
		if(m_decimatedMesh->property(m_uncollapsed, ci.v0v1))
			return -1.f;
	}

	float importance = m_decimatedMesh->property(m_importanceDensity, ci.v0);
	u32 count = 0u;
	for(auto ringVertexHandle = m_decimatedMesh->vv_iter(ci.v0); ringVertexHandle.is_valid(); ++ringVertexHandle) {
		importance += m_decimatedMesh->property(m_importanceDensity, *ringVertexHandle);
		++count;
	}
	importance /= static_cast<float>(count);

	if(m_collapseMode == CollapseMode::DAMPENED_CONCAVE) {
		// TODO: modify importance?
	}

	return importance;
}

void ImportanceDecimater::add_vertex_collapse(const Mesh::VertexHandle vh, const float threshold) {
	float bestPriority = std::numeric_limits<float>::max();
	typename Mesh::HalfedgeHandle collapseTarget;

	// find best target in one ring
	for(auto vohIter = m_decimatedMesh->voh_begin(vh); vohIter.is_valid(); ++vohIter) {
		const auto heh = *vohIter;
		CollapseInfo ci(m_decimatedMesh, heh);

		if(this->is_collapse_legal(ci)) {
			const float priority = this->collapse_priority(ci);
			if(priority >= 0.f && priority < bestPriority) {
				bestPriority = priority;
				collapseTarget = heh;
			}
		}
	}


	if(collapseTarget.is_valid() && bestPriority < threshold) {
		// target found -> put vertex on heap
		m_decimatedMesh->property(m_collapseTarget, vh) = collapseTarget;
		m_decimatedMesh->property(m_priority, vh) = bestPriority;

		if(m_heap->is_stored(vh))
			m_heap->update(vh);
		else
			m_heap->insert(vh);
	} else {
		// not valid -> remove from heap
		if(m_heap->is_stored(vh))
			m_heap->remove(vh);

		m_decimatedMesh->property(m_collapseTarget, vh) = collapseTarget;
		m_decimatedMesh->property(m_priority, vh) = -1.f;
	}
}

bool ImportanceDecimater::is_convex_collapse(const CollapseInfo& ci) const {
	const auto p0 = util::pun<ei::Vec3>(ci.p0);
	const auto p1 = util::pun<ei::Vec3>(ci.p1);
	const auto p0p1 = p1 - p0;
	const auto pl = util::pun<ei::Vec3>(m_decimatedMesh->point(ci.vl));
	const auto pr = util::pun<ei::Vec3>(m_decimatedMesh->point(ci.vr));
	const auto flNormal = ei::cross(p0p1, pl - p0);
	const auto frNormal = ei::cross(pr - p0, p0p1);
	const auto p0p1Normal = 0.5f * (flNormal + frNormal); // Not normalized because not needed
	{
		// First for v0: vx -> v0
		for(auto circIter = m_decimatedMesh->cvv_ccwbegin(ci.v0); circIter.is_valid(); ++circIter) {
			if(*circIter == ci.v1)
				continue;
			const auto pxp0 = p0 - util::pun<ei::Vec3>(m_decimatedMesh->point(*circIter));
			const auto dot = ei::dot(p0p1Normal, pxp0);
			if(dot < 0.f)
				return false;
		}
		// Then for v1: vx -> v1
		for(auto circIter = m_decimatedMesh->cvv_ccwbegin(ci.v1); circIter.is_valid(); ++circIter) {
			if(*circIter == ci.v0)
				continue;
			const auto pxp1 = p1 - util::pun<ei::Vec3>(m_decimatedMesh->point(*circIter));
			const auto dot = ei::dot(p0p1Normal, pxp1);
			if(dot < 0.f)
				return false;
		}
	}

	return true;
}

bool ImportanceDecimater::check_normal_deviation(const CollapseInfo& ci) {
	static thread_local std::vector<typename Mesh::Normal> normalStorage;

	// Compute the face normals before the collapse
	normalStorage.clear();
	for(auto iter = m_decimatedMesh->cvf_ccwbegin(ci.v0); iter.is_valid(); ++iter) {
		typename Mesh::FaceHandle fh = *iter;
		if(fh != ci.fl && fh != ci.fr) {
			normalStorage.push_back(m_decimatedMesh->calc_face_normal(fh));
		}
	}

	// simulate collapse
	m_decimatedMesh->set_point(ci.v0, ci.p1);

	// check for flipping normals
	typename Mesh::Scalar c(1.0);
	u32 index = 0u;
	for(auto iter = m_decimatedMesh->cvf_ccwbegin(ci.v0); iter.is_valid(); ++iter) {
		typename Mesh::FaceHandle fh = *iter;
		if(fh != ci.fl && fh != ci.fr) {
			typename const Mesh::Normal& n1 = normalStorage[index];
			typename Mesh::Normal n2 = m_decimatedMesh->calc_face_normal(fh);

			c = dot(n1, n2);

			if(c < m_minNormalCos)
				break;

			++index;
		}
	}

	// undo simulation changes
	m_decimatedMesh->set_point(ci.v0, ci.p0);

	return c >= m_minNormalCos;
}

float ImportanceDecimater::compute_new_importance_densities(std::vector<std::pair<float, float>>& newDensities,
														   const Mesh::VertexHandle v0, const Mesh::VertexHandle v1,
														   const Mesh::VertexHandle vl, const Mesh::VertexHandle vr,
														   const float threshold) const {
	return 1.f;
	newDensities.clear();
	float distSqrSum = 0.f;
	const auto v0Pos = util::pun<ei::Vec3>(m_originalMesh.point(v0));
	const auto v1Pos = util::pun<ei::Vec3>(m_decimatedMesh->point(v1));
	const auto vlPos = util::pun<ei::Vec3>(m_decimatedMesh->point(vl));
	const auto vrPos = util::pun<ei::Vec3>(m_decimatedMesh->point(vr));
	const auto v1vl = m_decimatedMesh->find_halfedge(v1, vl);

	float v0Importance = 0.f;

	if(v1 != vl && v1 != vr && vl != vr) {
		for(auto heh = v1vl; m_decimatedMesh->to_vertex_handle(heh) != vr;
			heh = m_decimatedMesh->opposite_halfedge_handle(m_decimatedMesh->prev_halfedge_handle(heh))) {
			const auto vb = m_decimatedMesh->to_vertex_handle(heh);
			distSqrSum += ei::lensq(util::pun<ei::Vec3>(m_decimatedMesh->point(vb)) - v0Pos);
		}
	}
	distSqrSum += ei::lensq(util::pun<ei::Vec3>(m_decimatedMesh->point(vr)) - v0Pos);
	distSqrSum += ei::lensq(util::pun<ei::Vec3>(m_decimatedMesh->point(v1)) - v0Pos);

	
	// Track the current area for v1
	float v1InnerArea = 0.f;
	const auto v0vlv1Area = ei::len(ei::cross(vlPos - v0Pos, v1Pos - v0Pos));
	const auto v0vrv1Area = ei::len(ei::cross(vrPos - v0Pos, v1Pos - v0Pos));
		
	Mesh::HalfedgeHandle leftHeh = m_decimatedMesh->opposite_halfedge_handle(m_decimatedMesh->prev_halfedge_handle(v1vl));
	auto middleVertex = vl;
	auto rightVertex = m_decimatedMesh->to_vertex_handle(leftHeh);
	auto middlePos = util::pun<ei::Vec3>(m_decimatedMesh->point(middleVertex));
	auto rightPos = util::pun<ei::Vec3>(m_decimatedMesh->point(rightVertex));
	float rightV0Area = ei::len(ei::cross(middlePos - v0Pos, rightPos - v0Pos));
	float rightV1Area = ei::len(ei::cross(middlePos - v1Pos, rightPos - v1Pos));

	// Interjection: check vl
	const float vlImp = m_importance[get_original_vertex_handle(vl).idx()].load();
	const float vlImpDensity = m_decimatedMesh->property(m_importanceDensity, vl);
	if(vlImpDensity == 0.f)
		return -1.f;
	const float vlArea = vlImp / vlImpDensity;
	const float vlImpLoss = vlImp * ei::lensq(vlPos - v0Pos) / distSqrSum;
	v0Importance += vlImpLoss;
	const float vlNewImp = vlImp - vlImpLoss;
	const float vlNewDensity = vlNewImp / (vlArea - rightV1Area + rightV0Area + v0vlv1Area);
	newDensities.push_back(std::make_pair(vlNewImp, vlNewDensity));
	if(newDensities.back().second < threshold)
		return -1.f;
	v1InnerArea += rightV1Area;

	// Iterate over the ring between vl and vr and determine the densities
	while(rightVertex != vr) {
		// Move over data
		const float leftV0Area = rightV0Area;
		const float leftV1Area = rightV1Area;
		middleVertex = rightVertex;
		middlePos = rightPos;

		const auto rightHeh = m_decimatedMesh->opposite_halfedge_handle(m_decimatedMesh->prev_halfedge_handle(leftHeh));
		if(rightVertex == m_decimatedMesh->to_vertex_handle(rightHeh)) {
			__debugbreak();
		}
		rightVertex = m_decimatedMesh->to_vertex_handle(rightHeh);
		rightPos = util::pun<ei::Vec3>(m_decimatedMesh->point(rightVertex));
		leftHeh = rightHeh;

		// Compute the new triangle area
		rightV0Area = ei::len(ei::cross(middlePos - v0Pos, rightPos - v0Pos));
		rightV1Area = ei::len(ei::cross(middlePos - v1Pos, rightPos - v1Pos));
		// Compare importance
		const float middleImp = m_importance[get_original_vertex_handle(middleVertex).idx()].load();
		const float middleImpDensity = m_decimatedMesh->property(m_importanceDensity, middleVertex);
		if(middleImpDensity == 0.f)
			return -1.f;
		const float middleArea = middleImp / middleImpDensity;
		const float middleImpLoss = middleImp * ei::lensq(middlePos - v0Pos) / distSqrSum;
		v0Importance += middleImpLoss;
		const float middleNewImp = middleImp - middleImpLoss;
		// TODO: buggy
		const float middleNewDensity = middleNewImp / (middleArea - leftV1Area - rightV1Area + leftV0Area + rightV0Area);
		newDensities.push_back(std::make_pair(middleNewImp, middleNewDensity));
		if(newDensities.back().second < threshold)
			return -1.f;
		v1InnerArea += rightV1Area;
	}

	// Check vr
	const float vrImp = m_importance[get_original_vertex_handle(vr).idx()].load();
	const float vrImpDensity = m_decimatedMesh->property(m_importanceDensity, vr);
	if(vrImpDensity == 0.f)
		return -1.f;
	const float vrArea = vrImp / vrImpDensity;
	const float vrImpLoss = vrImp * ei::lensq(vrPos - v0Pos) / distSqrSum;
	v0Importance += vrImpLoss;
	const float vrNewImp = vrImp - vrImpLoss;
	const float vrNewDensity = vrNewImp / (vrArea - rightV1Area + rightV0Area + v0vrv1Area);
	newDensities.push_back(std::make_pair(vlNewImp, vlNewDensity));
	if(newDensities.back().second < threshold)
		return -1.f;

	// Now check v1
	const float v1Imp = m_importance[get_original_vertex_handle(v1).idx()].load();
	const float v1ImpDensity = m_decimatedMesh->property(m_importanceDensity, v1);
	if(v1ImpDensity == 0.f)
		return -1.f;
	const float v1Area = v1Imp / v1ImpDensity;
	const float v1ImpLoss = v1Imp * ei::lensq(v1Pos - v0Pos) / distSqrSum;
	v0Importance += v1ImpLoss;
	const float v1NewImp = v1Imp - v1ImpLoss;
	const float v1NewDensity = v1NewImp / (v1Area - v1InnerArea + v0vlv1Area + v0vrv1Area);
	newDensities.push_back(std::make_pair(vlNewImp, vlNewDensity));
	if(newDensities.back().second < threshold)
		return -1.f;

	return v0Importance;
}
#endif

void ImportanceDecimater::record_silhouette_vertex_contribution(const u32 localIndex, const float importance) {
	// Reminder: local index will refer to the decimated mesh
	mAssert(localIndex < m_decimatedPoly->get_vertex_count());

	if(m_collapseMode == CollapseMode::DAMPENED_IMPORTANCE)
		atomic_add(m_importance[m_decimatedMesh->vertex_handle(localIndex).idx()], importance * m_damping);
	else
		atomic_add(m_importance[m_decimatedMesh->vertex_handle(localIndex).idx()], importance);

	if(m_collapseMode == CollapseMode::NO_SILHOUETTE)
		m_originalMesh.property(m_silhouette, this->get_original_vertex_handle(m_decimatedMesh->vertex_handle(localIndex))) = true;
}

void ImportanceDecimater::record_face_contribution(const u32* vertexIndices, const u32 vertexCount,
												   const ei::Vec3& hitpoint, const float importance) {
	mAssert(vertexIndices != nullptr);

	float distSqrSum = 0.f;
	for(u32 v = 0u; v < vertexCount; ++v) {
		const auto vh = m_decimatedMesh->vertex_handle(vertexIndices[v]);
		distSqrSum += ei::lensq(hitpoint - util::pun<ei::Vec3>(m_decimatedMesh->point(vh)));
	}
	const float distSqrSumInv = 1.f / distSqrSum;

	// Now do the actual attribution
	for(u32 v = 0u; v < vertexCount; ++v) {
		const auto vh = m_decimatedMesh->vertex_handle(vertexIndices[v]);
		const float distSqr = ei::lensq(hitpoint - util::pun<ei::Vec3>(m_decimatedMesh->point(vh)));
		const float weightedImportance = importance * distSqr * distSqrSumInv;

		if(m_collapseMode == CollapseMode::DAMPENED_IMPORTANCE)
			atomic_add(m_importance[vh.idx()], weightedImportance * m_damping);
		else
			atomic_add(m_importance[vh.idx()], weightedImportance);
	}
}

// Utility only
ImportanceDecimater::Mesh::VertexHandle ImportanceDecimater::get_original_vertex_handle(const Mesh::VertexHandle decimatedHandle) const {
	mAssert(decimatedHandle.is_valid());
	mAssert(static_cast<std::size_t>(decimatedHandle.idx()) < m_decimatedMesh->n_vertices());
	const auto originalHandle = m_decimatedMesh->property(m_originalVertex, decimatedHandle);
	mAssert(originalHandle.is_valid());
	mAssert(static_cast<std::size_t>(originalHandle.idx()) < m_originalPoly.get_vertex_count());
	return originalHandle;
}

float ImportanceDecimater::get_max_importance() const {
	float maxImp = 0.f;
	for(auto vertex : m_decimatedMesh->vertices())
		maxImp = std::max(maxImp, m_importance[vertex.idx()].load());
	return maxImp;
}

float ImportanceDecimater::get_max_importance_density() const {
	float maxImpDensity = 0.f;
	for(auto vertex : m_decimatedMesh->vertices())
		maxImpDensity = std::max(maxImpDensity, m_decimatedMesh->property(m_importanceDensity, vertex));
	return maxImpDensity;
}

float ImportanceDecimater::get_importance(const u32 localFaceIndex, const ei::Vec3& hitpoint) const {
	const auto faceHandle = m_decimatedMesh->face_handle(localFaceIndex);

	float importance = 0.f;
	float distSqrSum = 0.f;
	for(auto circIter = m_decimatedMesh->cfv_ccwbegin(faceHandle); circIter.is_valid(); ++circIter)
		distSqrSum += ei::lensq(hitpoint - util::pun<ei::Vec3>(m_decimatedMesh->point(*circIter)));

	for(auto circIter = m_decimatedMesh->cfv_ccwbegin(faceHandle); circIter.is_valid(); ++circIter) {
		const float distSqr = ei::lensq(hitpoint - util::pun<ei::Vec3>(m_decimatedMesh->point(*circIter)));
		importance += m_importance[circIter->idx()].load() * distSqr / distSqrSum;
	}

	return importance;
}

float ImportanceDecimater::get_importance_density(const u32 localFaceIndex, const ei::Vec3& hitpoint) const {
	const auto faceHandle = m_decimatedMesh->face_handle(localFaceIndex);

	float importance = 0.f;
	float distSqrSum = 0.f;
	for(auto circIter = m_decimatedMesh->cfv_ccwbegin(faceHandle); circIter.is_valid(); ++circIter)
		distSqrSum += ei::lensq(hitpoint - util::pun<ei::Vec3>(m_decimatedMesh->point(*circIter)));

	for(auto circIter = m_decimatedMesh->cfv_ccwbegin(faceHandle); circIter.is_valid(); ++circIter) {
		const float distSqr = ei::lensq(hitpoint - util::pun<ei::Vec3>(m_decimatedMesh->point(*circIter)));
		importance += m_decimatedMesh->property(m_importanceDensity, *circIter) * distSqr / distSqrSum;
	}

	return importance;
}

std::size_t ImportanceDecimater::get_original_vertex_count() const noexcept {
	return m_originalMesh.n_vertices();
}

std::size_t ImportanceDecimater::get_decimated_vertex_count() const noexcept {
	return m_decimatedMesh->n_vertices();
}

} // namespace mufflon::renderer::silhouette::decimation