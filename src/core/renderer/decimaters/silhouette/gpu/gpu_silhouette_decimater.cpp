#pragma once

#include "gpu_silhouette_decimater.hpp"
#include "util/parallel.hpp"
#include "util/punning.hpp"
#include "core/renderer/decimaters/silhouette/modules/importance_quadrics.hpp"
#include "core/renderer/decimaters/modules/collapse_tracker.hpp"
#include <OpenMesh/Tools/Decimater/DecimaterT.hh>
#include <OpenMesh/Tools/Decimater/ModQuadricT.hh>

namespace mufflon::renderer::decimaters::silhouette {

using namespace scene;
using namespace scene::geometry;
using namespace modules;
using namespace decimaters::modules;

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

GpuImportanceDecimater::GpuImportanceDecimater(Lod& original, Lod& decimated,
										 const std::size_t initialCollapses,
										 const float viewWeight, const float lightWeight,
										 const float shadowWeight, const float shadowSilhouetteWeight) :
	m_original(original),
	m_decimated(decimated),
	m_originalPoly(m_original.template get_geometry<Polygons>()),
	m_decimatedPoly(&m_decimated.template get_geometry<Polygons>()),
	m_originalMesh(m_originalPoly.get_mesh()),
	m_decimatedMesh(&m_decimatedPoly->get_mesh()),
	m_importances(nullptr),
	m_devImportances(nullptr),
	m_viewWeight(viewWeight),
	m_lightWeight(lightWeight),
	m_shadowWeight(shadowWeight),
	m_shadowSilhouetteWeight(shadowSilhouetteWeight)
{
	// Add necessary properties
	m_decimatedMesh->add_property(m_originalVertex);
	m_originalMesh.add_property(m_accumulatedImportanceDensity);
	m_originalMesh.add_property(m_collapsedTo);
	m_originalMesh.add_property(m_collapsed);
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
		decimater.add(modQuadricHandle);
		decimater.add(trackerHandle);
		decimater.module(trackerHandle).set_properties(m_originalMesh, m_collapsed, m_collapsedTo);
		// Possibly repeat until we reached the desired count
		const std::size_t targetCollapses = std::min(initialCollapses, m_originalPoly.get_vertex_count());
		const std::size_t targetVertexCount = m_originalPoly.get_vertex_count() - targetCollapses;
		std::size_t performedCollapses = m_decimatedPoly->decimate(decimater, targetVertexCount, false);
		if(performedCollapses < targetCollapses)
			logWarning("Not all decimations were performed: ", targetCollapses - performedCollapses, " missing");
		m_decimatedPoly->garbage_collect([this](Mesh::VertexHandle deletedVertex, Mesh::VertexHandle changedVertex) {
			// Adjust the reference from original to decimated mesh
			const auto originalVertex = this->get_original_vertex_handle(changedVertex);
			if(!m_originalMesh.property(m_collapsed, originalVertex))
				m_originalMesh.property(m_collapsedTo, originalVertex) = deletedVertex;
		});
		this->recompute_geometric_vertex_normals();

		m_decimated.clear_accel_structure();
	}
}

GpuImportanceDecimater::GpuImportanceDecimater(GpuImportanceDecimater&& dec) :
	m_original(dec.m_original),
	m_decimated(dec.m_decimated),
	m_originalPoly(dec.m_originalPoly),
	m_decimatedPoly(dec.m_decimatedPoly),
	m_originalMesh(dec.m_originalMesh),
	m_decimatedMesh(dec.m_decimatedMesh),
	m_importances(std::move(dec.m_importances)),
	m_originalVertex(dec.m_originalVertex),
	m_accumulatedImportanceDensity(dec.m_accumulatedImportanceDensity),
	m_collapsedTo(dec.m_collapsedTo),
	m_collapsed(dec.m_collapsed),
	m_viewWeight(dec.m_viewWeight),
	m_lightWeight(dec.m_lightWeight),
	m_shadowWeight(dec.m_shadowWeight),
	m_shadowSilhouetteWeight(dec.m_shadowSilhouetteWeight)
{
	// Request the status again since it will get removed once in the destructor
	m_decimatedMesh->request_vertex_status();
	m_decimatedMesh->request_edge_status();
	m_decimatedMesh->request_halfedge_status();
	m_decimatedMesh->request_face_status();
	// Invalidate the handles here so we know not to remove them in the destructor
	dec.m_originalVertex.invalidate();
	dec.m_accumulatedImportanceDensity.invalidate();
	dec.m_collapsedTo.invalidate();
	dec.m_collapsed.invalidate();
}

GpuImportanceDecimater::~GpuImportanceDecimater() {
	if(m_originalVertex.is_valid())
		m_decimatedMesh->remove_property(m_originalVertex);
	if(m_accumulatedImportanceDensity.is_valid())
		m_originalMesh.remove_property(m_accumulatedImportanceDensity);
	if(m_collapsedTo.is_valid())
		m_originalMesh.remove_property(m_collapsedTo);
	if(m_collapsed.is_valid())
		m_originalMesh.remove_property(m_collapsed);

	m_decimatedMesh->release_vertex_status();
	m_decimatedMesh->release_edge_status();
	m_decimatedMesh->release_halfedge_status();
	m_decimatedMesh->release_face_status();
}

void GpuImportanceDecimater::upload_normalized_importance() {
	copy(m_devImportances.get(), m_importances.get(), sizeof(Importances<Device::CUDA>) * m_decimatedPoly->get_vertex_count());
}

void GpuImportanceDecimater::udpate_importance_density(const ImportanceSums& sum) {
	// First get the data off the GPU
	copy(m_importances.get(), m_devImportances.get(), sizeof(Importances<Device::CUDA>) * m_decimatedPoly->get_vertex_count());

	// Update our statistics: the importance density of each vertex
	float importanceSum = 0.0;
#pragma PARALLEL_REDUCTION(+, importanceSum)
	for(i64 i = 0; i < static_cast<i64>(m_decimatedMesh->n_vertices()); ++i) {
		const auto vertex = m_decimatedMesh->vertex_handle(static_cast<u32>(i));
		// Important: only works for triangles!
		const float area = compute_area(*m_decimatedMesh, vertex);
		const float flux = m_importances[vertex.idx()].irradiance
			/ std::max(1.f, static_cast<float>(m_importances[vertex.idx()].hitCounter));
		const float viewImportance = m_importances[vertex.idx()].viewImportance;

		const float importance = viewImportance + m_lightWeight * flux;

		importanceSum += importance;
		m_importances[vertex.idx()].viewImportance = importance / area;
	}

	// Subtract the shadow silhouette importance and use shadow importance instead
	logPedantic("Importance sum/shadow/silhouette: ", importanceSum, " ", sum.shadowImportance, " ", sum.shadowSilhouetteImportance);
	m_importanceSum = importanceSum + m_shadowWeight * sum.shadowImportance - sum.shadowSilhouetteImportance;

	// Map the importance back to the original mesh
	for(auto iter = m_originalMesh.vertices_begin(); iter != m_originalMesh.vertices_end(); ++iter) {
		const auto vertex = *iter;
		// Traverse collapse chain and snatch importance
		auto v = vertex;
		while(m_originalMesh.property(m_collapsed, v))
			v = m_originalMesh.property(m_collapsedTo, v);

		// Put importance into temporary storage
		m_originalMesh.property(m_accumulatedImportanceDensity, vertex) = m_importances[m_originalMesh.property(m_collapsedTo, v).idx()].viewImportance;
	}
}

void GpuImportanceDecimater::recompute_geometric_vertex_normals() {
#pragma PARALLEL_FOR
	for(i64 i = 0; i < static_cast<i64>(m_decimatedMesh->n_vertices()); ++i) {
		const auto vertex = m_decimatedMesh->vertex_handle(static_cast<u32>(i));

		typename Mesh::Normal normal;
#pragma warning(push)
#pragma warning(disable : 4244)
		m_decimatedMesh->calc_vertex_normal_correct(vertex, normal);
#pragma warning(pop)
		m_decimatedMesh->set_normal(vertex, util::pun<typename Mesh::Normal>(ei::normalize(util::pun<ei::Vec3>(normal))));
	}
}

Importances<Device::CUDA>* GpuImportanceDecimater::start_iteration() {
	// Resize importance map
	m_importances = std::make_unique<Importances<Device::CUDA>[]>(m_decimatedPoly->get_vertex_count());
	m_devImportances = make_udevptr_array<Device::CUDA, Importances<Device::CUDA>, false>(m_decimatedPoly->get_vertex_count());
	cuda::check_error(cudaMemset(m_devImportances.get(), 0, sizeof(Importances<Device::CUDA>) * m_decimatedPoly->get_vertex_count()));
	return m_devImportances.get();
}

void GpuImportanceDecimater::iterate(const std::size_t minVertexCount, const float reduction) {
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
	ImportanceDecimationModule<>::Handle impHandle;
	decimater.add(trackerHandle);
	decimater.add(impHandle);
	decimater.module(trackerHandle).set_properties(m_originalMesh, m_collapsed, m_collapsedTo);
	decimater.module(impHandle).set_properties(m_originalMesh, m_accumulatedImportanceDensity);
	
	if(reduction != 0.f) {
		const std::size_t targetCount = (reduction == 1.f) ? 0u : static_cast<std::size_t>((1.f - reduction) * m_originalPoly.get_vertex_count());
		const auto collapses = m_decimatedPoly->decimate(decimater, targetCount, false);
		if(collapses > 0u) {
			m_decimatedPoly->garbage_collect([this](Mesh::VertexHandle deletedVertex, Mesh::VertexHandle changedVertex) {
				// Adjust the reference from original to decimated mesh
				const auto originalVertex = this->get_original_vertex_handle(changedVertex);
				if(!m_originalMesh.property(m_collapsed, originalVertex))
					m_originalMesh.property(m_collapsedTo, originalVertex) = deletedVertex;
			});
			this->recompute_geometric_vertex_normals();
			m_decimated.clear_accel_structure();
		}
		logPedantic("Performed ", collapses, " collapses, remaining vertices: ", m_decimatedMesh->n_vertices());
	}
}

// Utility only
GpuImportanceDecimater::Mesh::VertexHandle GpuImportanceDecimater::get_original_vertex_handle(const Mesh::VertexHandle decimatedHandle) const {
	mAssert(decimatedHandle.is_valid());
	mAssert(static_cast<std::size_t>(decimatedHandle.idx()) < m_decimatedMesh->n_vertices());
	const auto originalHandle = m_decimatedMesh->property(m_originalVertex, decimatedHandle);
	mAssert(originalHandle.is_valid());
	mAssert(static_cast<std::size_t>(originalHandle.idx()) < m_originalPoly.get_vertex_count());
	return originalHandle;
}

float GpuImportanceDecimater::get_current_max_importance() const {
	float maxImp = 0.f;
#pragma PARALLEL_FOR
	for(i64 v = 0; v < static_cast<i64>(m_decimatedPoly->get_vertex_count()); ++v) {
#pragma omp critical
		if(m_importances[v].viewImportance > maxImp)
			maxImp = m_importances[v].viewImportance;
	}
	return maxImp;
}

std::size_t GpuImportanceDecimater::get_original_vertex_count() const noexcept {
	return m_originalMesh.n_vertices();
}

std::size_t GpuImportanceDecimater::get_decimated_vertex_count() const noexcept {
	return m_decimatedMesh->n_vertices();
}

} // namespace mufflon::renderer::decimaters::silhouette