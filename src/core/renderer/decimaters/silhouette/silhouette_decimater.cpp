#pragma once

#include "silhouette_decimater.hpp"
#include "util/parallel.hpp"
#include "util/punning.hpp"
#include "modules/convex_edge.hpp"
#include "modules/importance_quadrics.hpp"
#include "modules/silhouette.hpp"
#include "core/renderer/decimaters/modules/collapse_tracker.hpp"
#include "core/renderer/decimaters/modules/normal_deviation.hpp"
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

ImportanceDecimater::ImportanceDecimater(Lod& original, Lod& decimated,
										 const std::size_t initialCollapses,
										 const Degrees maxNormalDeviation,
										 const float viewWeight, const float lightWeight,
										 const float shadowWeight, const float shadowSilhouetteWeight) :
	m_original(original),
	m_decimated(decimated),
	m_originalPoly(m_original.template get_geometry<Polygons>()),
	m_decimatedPoly(&m_decimated.template get_geometry<Polygons>()),
	m_originalMesh(m_originalPoly.get_mesh()),
	m_decimatedMesh(&m_decimatedPoly->get_mesh()),
	m_importances(nullptr),
	m_maxNormalDeviation(maxNormalDeviation),
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
		NormalDeviationModule<>::Handle normalHandle;
		decimater.add(modQuadricHandle);
		decimater.add(trackerHandle);
		decimater.add(normalHandle);
		decimater.module(trackerHandle).set_properties(m_originalMesh, m_collapsed, m_collapsedTo);
		decimater.module(normalHandle).set_max_deviation(m_maxNormalDeviation);
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

		m_decimated.clear_accel_structure();
	}

	// Initialize importance map
	m_shadowImportance.store(0.f);
	m_shadowSilhouetteImportance.store(0.f);
	m_importances = std::make_unique<Importances[]>(m_decimatedPoly->get_vertex_count());
	for(std::size_t i = 0u; i < m_decimatedPoly->get_vertex_count(); ++i) {
		m_importances[i].viewImportance.store(0.f);
		m_importances[i].irradiance.store(0.f);
		m_importances[i].hitCounter.store(0u);
	}
}

ImportanceDecimater::ImportanceDecimater(ImportanceDecimater&& dec) :
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
	m_maxNormalDeviation(dec.m_maxNormalDeviation),
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

	m_shadowImportance.store(dec.m_shadowImportance.load());
	m_shadowSilhouetteImportance.store(dec.m_shadowSilhouetteImportance.load());
}

ImportanceDecimater::~ImportanceDecimater() {
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

void ImportanceDecimater::udpate_importance_density() {
	// Update our statistics: the importance density of each vertex
	float importanceSum = 0.0;
#pragma PARALLEL_REDUCTION(+, importanceSum)
	for(i64 i = 0; i < static_cast<i64>(m_decimatedMesh->n_vertices()); ++i) {
		const auto vertex = m_decimatedMesh->vertex_handle(static_cast<u32>(i));
		// Important: only works for triangles!
		const float area = compute_area(*m_decimatedMesh, vertex);
		const float flux = m_importances[vertex.idx()].irradiance.load()
			/ std::max(1.f, static_cast<float>(m_importances[vertex.idx()].hitCounter.load()));
		const float viewImportance = m_importances[vertex.idx()].viewImportance;

		const float importance = viewImportance + m_lightWeight * flux;

		importanceSum += importance;
		m_importances[vertex.idx()].viewImportance.store(importance / area);
	}

	// Subtract the shadow silhouette importance and use shadow importance instead
	logPedantic("Importance sum/shadow/silhouette: ", importanceSum, " ", m_shadowImportance, " ", m_shadowSilhouetteImportance);
	m_importanceSum = importanceSum + m_shadowWeight * m_shadowImportance - m_shadowSilhouetteImportance;

	// Map the importance back to the original mesh
	for(auto iter = m_originalMesh.vertices_begin(); iter != m_originalMesh.vertices_end(); ++iter) {
		const auto vertex = *iter;
		// Traverse collapse chain and snatch importance
		auto v = vertex;
		while(m_originalMesh.property(m_collapsed, v))
			v = m_originalMesh.property(m_collapsedTo, v);

		// Put importance into temporary storage
		m_originalMesh.property(m_accumulatedImportanceDensity, vertex) = m_importances[m_originalMesh.property(m_collapsedTo, v).idx()].viewImportance.load();
	}
}

void ImportanceDecimater::iterate(const std::size_t minVertexCount, const float reduction) {
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
	NormalDeviationModule<>::Handle normalHandle;
	ImportanceDecimationModule<>::Handle impHandle;
	decimater.add(trackerHandle);
	decimater.add(normalHandle);
	decimater.add(impHandle);
	decimater.module(trackerHandle).set_properties(m_originalMesh, m_collapsed, m_collapsedTo);
	decimater.module(normalHandle).set_max_deviation(m_maxNormalDeviation);
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
			m_decimated.clear_accel_structure();
		}
		logPedantic("Performed ", collapses, " collapses, remaining vertices: ", m_decimatedMesh->n_vertices());
	}

	// Reinitialize importance map
	m_shadowImportance.store(0.f);
	m_shadowSilhouetteImportance.store(0.f);
	m_importances = std::make_unique<Importances[]>(m_decimatedPoly->get_vertex_count());
	for(std::size_t i = 0u; i < m_decimatedPoly->get_vertex_count(); ++i) {
		m_importances[i].viewImportance.store(0.f);
		m_importances[i].irradiance.store(0.f);
		m_importances[i].hitCounter.store(0u);
	}
}

void ImportanceDecimater::record_silhouette_vertex_contribution(const u32 localIndex, const float importance) {
	// Reminder: local index will refer to the decimated mesh
	mAssert(localIndex < m_decimatedPoly->get_vertex_count());

	const float weightedImportance = importance * m_shadowSilhouetteWeight;

	if(isnan(weightedImportance))
		__debugbreak();
	atomic_add(m_importances[m_decimatedMesh->vertex_handle(localIndex).idx()].viewImportance, weightedImportance);
	atomic_add(m_shadowSilhouetteImportance, weightedImportance);
}

void ImportanceDecimater::record_shadow(const float irradiance) {
	if(isnan(irradiance))
		__debugbreak();
	atomic_add(m_shadowImportance, irradiance);
}

void ImportanceDecimater::record_direct_hit(const u32* vertexIndices, const u32 vertexCount,
											const ei::Vec3& hitpoint, const float cosAngle,
											const float sharpness) {
	typename Mesh::VertexHandle min;
	float minDist = std::numeric_limits<float>::max();
	for(u32 v = 0u; v < vertexCount; ++v) {
		const auto vh = m_decimatedMesh->vertex_handle(vertexIndices[v]);
		const float dist = ei::lensq(hitpoint - util::pun<ei::Vec3>(m_decimatedMesh->point(vh)));
		if(dist < minDist) {
			minDist = dist;
			min = vh;
		}
	}

	if(isnan(sharpness))
		__debugbreak();
	if(isnan(cosAngle))
		__debugbreak();
	atomic_add(m_importances[min.idx()].viewImportance, m_viewWeight * sharpness * (1.f - ei::abs(cosAngle)));
}

void ImportanceDecimater::record_direct_irradiance(const u32* vertexIndices, const u32 vertexCount,
												   const ei::Vec3& hitpoint, const float irradiance) {
	typename Mesh::VertexHandle min;
	float minDist = std::numeric_limits<float>::max();
	for(u32 v = 0u; v < vertexCount; ++v) {
		const auto vh = m_decimatedMesh->vertex_handle(vertexIndices[v]);
		const float dist = ei::lensq(hitpoint - util::pun<ei::Vec3>(m_decimatedMesh->point(vh)));
		if(dist < minDist) {
			minDist = dist;
			min = vh;
		}
	}

	atomic_add(m_importances[min.idx()].irradiance, irradiance);
	++m_importances[min.idx()].hitCounter;
}

void ImportanceDecimater::record_indirect_irradiance(const u32* vertexIndices, const u32 vertexCount,
													 const ei::Vec3& hitpoint, const float irradiance) {
	typename Mesh::VertexHandle min;
	float minDist = std::numeric_limits<float>::max();
	for(u32 v = 0u; v < vertexCount; ++v) {
		const auto vh = m_decimatedMesh->vertex_handle(vertexIndices[v]);
		const float dist = ei::lensq(hitpoint - util::pun<ei::Vec3>(m_decimatedMesh->point(vh)));
		if(dist < minDist) {
			minDist = dist;
			min = vh;
		}
	}

	if(isnan(irradiance))
		__debugbreak();
	atomic_add(m_importances[min.idx()].irradiance, irradiance);
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

float ImportanceDecimater::get_current_max_importance() const {
	float maxImp = 0.f;
	for(auto vertex : m_decimatedMesh->vertices())
		maxImp = std::max(maxImp, m_originalMesh.property(m_accumulatedImportanceDensity, get_original_vertex_handle(vertex)));
	return maxImp;
}

float ImportanceDecimater::get_mapped_max_importance() const {
	float maxImp = 0.f;
	for(auto vertex : m_originalMesh.vertices())
		maxImp = std::max(maxImp, m_originalMesh.property(m_accumulatedImportanceDensity, vertex));
	return maxImp;
}

float ImportanceDecimater::get_current_importance(const u32 localFaceIndex, const ei::Vec3& hitpoint) const {
	const auto faceHandle = m_decimatedMesh->face_handle(localFaceIndex);

	float importance = 0.f;
	float distSqrSum = 0.f;
	for(auto circIter = m_decimatedMesh->cfv_ccwbegin(faceHandle); circIter.is_valid(); ++circIter)
		distSqrSum += ei::lensq(hitpoint - util::pun<ei::Vec3>(m_decimatedMesh->point(*circIter)));

	for(auto circIter = m_decimatedMesh->cfv_ccwbegin(faceHandle); circIter.is_valid(); ++circIter) {
		const float distSqr = ei::lensq(hitpoint - util::pun<ei::Vec3>(m_decimatedMesh->point(*circIter)));
		importance += m_originalMesh.property(m_accumulatedImportanceDensity, get_original_vertex_handle(*circIter)) * distSqr / distSqrSum;
	}

	return importance;
}

float ImportanceDecimater::get_mapped_importance(const u32 originalFaceIndex, const ei::Vec3& hitpoint) const {
	const auto faceHandle = m_originalMesh.face_handle(originalFaceIndex);

	float importance = 0.f;
	float distSqrSum = 0.f;
	for(auto circIter = m_originalMesh.cfv_ccwbegin(faceHandle); circIter.is_valid(); ++circIter)
		distSqrSum += ei::lensq(hitpoint - util::pun<ei::Vec3>(m_originalMesh.point(*circIter)));

	for(auto circIter = m_originalMesh.cfv_ccwbegin(faceHandle); circIter.is_valid(); ++circIter) {
		const float distSqr = ei::lensq(hitpoint - util::pun<ei::Vec3>(m_originalMesh.point(*circIter)));
		importance += m_originalMesh.property(m_accumulatedImportanceDensity, *circIter) * distSqr / distSqrSum;
	}

	return importance;
}

std::size_t ImportanceDecimater::get_original_vertex_count() const noexcept {
	return m_originalMesh.n_vertices();
}

std::size_t ImportanceDecimater::get_decimated_vertex_count() const noexcept {
	return m_decimatedMesh->n_vertices();
}

} // namespace mufflon::renderer::decimaters::silhouette