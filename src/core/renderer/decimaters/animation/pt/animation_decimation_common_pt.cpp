#include "animation_decimation_common_pt.hpp"
#include "util/parallel.hpp"
#include "util/punning.hpp"
#include "core/renderer/decimaters/silhouette/modules/importance_quadrics.hpp"
#include "core/renderer/decimaters/modules/collapse_tracker.hpp"
#include "core/scene/geometry/polygon.hpp"
#include <ei/vector.hpp>
#include <OpenMesh/Tools/Decimater/DecimaterT.hh>
#include <OpenMesh/Tools/Decimater/ModQuadricT.hh>

namespace mufflon::renderer::decimaters::animation::pt {

using namespace modules;
using namespace mufflon::scene;
using namespace mufflon::scene::geometry;
using namespace mufflon::renderer::decimaters::modules;

namespace {

inline float compute_area(const PolygonMeshType& mesh, const OpenMesh::VertexHandle vertex) {
	float area = 0.f;
	for(auto fIter = mesh.cvf_ccwbegin(vertex); fIter.is_valid(); ++fIter) {
		auto vIter = mesh.cfv_ccwbegin(*fIter);
		const auto a = *vIter; ++vIter;
		const auto b = *vIter; ++vIter;
		const auto c = *vIter; ++vIter;
		const auto pA = util::pun<ei::Vec3>(mesh.point(a));
		const auto pB = util::pun<ei::Vec3>(mesh.point(b));
		const auto pC = util::pun<ei::Vec3>(mesh.point(c));
		area += ei::len(ei::cross(pB - pA, pC - pA));
		if(vIter.is_valid()) {
			const auto d = *vIter;
			const auto pD = util::pun<ei::Vec3>(mesh.point(d));
			area += ei::len(ei::cross(pC - pA, pD - pA));
		}
	}
	return 0.5f * area;
}

} // namespace

ImportanceDecimater::ImportanceDecimater(StringView objectName, ArrayDevHandle_t<DEVICE, silhouette::pt::Importances<DEVICE>> impBuffer,
										 Lod& original, Lod& decimated, const std::size_t initialCollapses,
										 const u32 frameCount, const float viewWeight, const float lightWeight,
										 const float shadowWeight, const float shadowSilhouetteWeight) :
	m_objectName(objectName),
	m_original(original),
	m_decimated(decimated),
	m_originalPoly(m_original.template get_geometry<Polygons>()),
	m_decimatedPoly(&m_decimated.template get_geometry<Polygons>()),
	m_originalMesh(m_originalPoly.get_mesh()),
	m_decimatedMesh(&m_decimatedPoly->get_mesh()),
	m_frameCount{ frameCount },
	m_importanceBuffer{ impBuffer },
	m_currImpBuffer{ nullptr },
	m_viewWeight(viewWeight),
	m_lightWeight(lightWeight),
	m_shadowWeight(shadowWeight),
	m_shadowSilhouetteWeight(shadowSilhouetteWeight) {
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

	// Add curvature
	m_originalPoly.compute_curvature();

	// Perform initial decimation
	this->decimate_with_error_quadrics(initialCollapses);
}

ImportanceDecimater::ImportanceDecimater(ImportanceDecimater&& dec) :
	m_original(dec.m_original),
	m_decimated(dec.m_decimated),
	m_originalPoly(dec.m_originalPoly),
	m_decimatedPoly(dec.m_decimatedPoly),
	m_originalMesh(dec.m_originalMesh),
	m_frameCount{ dec.m_frameCount },
	m_decimatedMesh(dec.m_decimatedMesh),
	m_importanceBuffer(std::move(dec.m_importanceBuffer)),
	m_currImpBuffer{ dec.m_currImpBuffer },
	m_originalVertex(dec.m_originalVertex),
	m_accumulatedImportanceDensity(dec.m_accumulatedImportanceDensity),
	m_collapsedTo(dec.m_collapsedTo),
	m_collapsed(dec.m_collapsed),
	m_viewWeight(dec.m_viewWeight),
	m_lightWeight(dec.m_lightWeight),
	m_shadowWeight(dec.m_shadowWeight),
	m_shadowSilhouetteWeight(dec.m_shadowSilhouetteWeight) {
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

void ImportanceDecimater::update_importance_density(const silhouette::pt::ImportanceSums& sum) {
	// Update our statistics: the importance density of each vertex
	float importanceSum = 0.0;
#pragma PARALLEL_REDUCTION(+, importanceSum)
	for(i64 i = 0; i < static_cast<i64>(m_decimatedMesh->n_vertices()); ++i) {
		const auto vertex = m_decimatedMesh->vertex_handle(static_cast<u32>(i));
		// Important: only works for triangles!
		const float area = compute_area(*m_decimatedMesh, vertex);
		const float flux = m_currImpBuffer[vertex.idx()].irradiance
			/ std::max(1.f, static_cast<float>(m_currImpBuffer[vertex.idx()].hitCounter));
		const float viewImportance = m_currImpBuffer[vertex.idx()].viewImportance;

		const float importance = viewImportance + m_lightWeight * flux;

		importanceSum += importance;
		m_currImpBuffer[vertex.idx()].viewImportance = importance;
	}

	// Subtract the shadow silhouette importance and use shadow importance instead
	logPedantic("Importance sum/shadow/silhouette(", m_objectName, "): ", importanceSum, " ", sum.shadowImportance, " ", sum.shadowSilhouetteImportance);
	m_importanceSum.back() = importanceSum + m_shadowWeight * sum.shadowImportance - sum.shadowSilhouetteImportance;
}

void ImportanceDecimater::upload_importance(const PImpWeightMethod::Values weighting,
											u32 startFrame, u32 endFrame) {
	// Map the importance back to the original mesh
	const auto* curvature = m_originalPoly.template acquire_const<Device::CPU, float>(m_originalPoly.get_curvature_hdl().value());
	for(auto iter = m_originalMesh.vertices_begin(); iter != m_originalMesh.vertices_end(); ++iter) {
		const auto vertex = *iter;
		// Traverse collapse chain and snatch importance
		auto v = vertex;
		while(m_originalMesh.property(m_collapsed, v))
			v = m_originalMesh.property(m_collapsedTo, v);

		// Put importance into temporary storage
		// TODO: end of frame sequence!
		float importance = 0.f;
		const auto area = compute_area(m_originalMesh, vertex);
		const auto curv = std::abs(curvature[vertex.idx()]);

		switch(weighting) {
			case PImpWeightMethod::Values::AVERAGE_ALL:
				startFrame = 0u;
				endFrame = m_frameCount - 1u;
				[[fallthrough]];
			case PImpWeightMethod::Values::AVERAGE:
				for(u32 f = startFrame; f <= endFrame; ++f) {
					// TODO: the importance comes not from the original, but the successively decimated mesh
					//const auto offset = m_originalMesh.n_vertices() * f + m_originalMesh.property(m_collapsedTo, v).idx();
					const auto offset = m_originalMesh.n_vertices() * f + vertex.idx();
					importance += m_importanceBuffer[offset].viewImportance;
				}
				importance /= static_cast<float>(endFrame - startFrame + 1u);
				break;
			case PImpWeightMethod::Values::MAX_ALL:
				startFrame = 0u;
				endFrame = m_frameCount - 1u;
				[[fallthrough]];
			case PImpWeightMethod::Values::MAX:
				for(u32 f = startFrame; f <= endFrame; ++f) {
					//const auto offset = m_originalMesh.n_vertices() * f + m_originalMesh.property(m_collapsedTo, v).idx();
					const auto offset = m_originalMesh.n_vertices() * f + vertex.idx();
					importance = std::max<float>(
						m_originalMesh.property(m_accumulatedImportanceDensity, vertex),
						m_importanceBuffer[offset].viewImportance);
				}
				break;
		}
		const auto weightedImportance = std::sqrt(importance * curv);
		m_originalMesh.property(m_accumulatedImportanceDensity, vertex) = weightedImportance;
	}
}

void ImportanceDecimater::recompute_geometric_vertex_normals() {
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

ArrayDevHandle_t<Device::CPU, silhouette::pt::Importances<Device::CPU>> ImportanceDecimater::start_iteration() {
	// TODO: implement sliding window

	// For now, we assume that we only start iterations for valid frames
	// TODO: in theory, decimatedMesh vertices should suffice, but that may change between frames
	if(m_currImpBuffer == nullptr)
		m_currImpBuffer = m_importanceBuffer;
	else
		m_currImpBuffer += m_originalMesh.n_vertices();
	m_importanceSum.emplace_back();
	std::memset(m_currImpBuffer, 0, sizeof(*m_currImpBuffer) * m_originalMesh.n_vertices());
	return m_currImpBuffer;
}


void ImportanceDecimater::iterate(const std::size_t targetCount) {
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
	silhouette::modules::ImportanceDecimationModule<>::Handle impHandle;
	decimater.add(trackerHandle);
	decimater.add(impHandle);
	decimater.module(trackerHandle).set_properties(m_originalMesh, m_collapsed, m_collapsedTo);
	decimater.module(impHandle).set_properties(m_originalMesh, m_accumulatedImportanceDensity);

	if(targetCount < get_original_vertex_count()) {
		const auto t0 = std::chrono::high_resolution_clock::now();
		const auto collapses = m_decimatedPoly->decimate(decimater, targetCount, false);
		const auto t1 = std::chrono::high_resolution_clock::now();
		const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0);
		logInfo("Collapse duration: ", duration.count(), "ms");
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
		logPedantic("Performed ", collapses, " collapses for object '", m_objectName, "', remaining vertices: ", m_decimatedMesh->n_vertices());
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


float ImportanceDecimater::get_current_max_importance() const {
	float maxImp = 0.f;
#pragma PARALLEL_FOR
	for(i64 v = 0; v < static_cast<i64>(m_decimatedPoly->get_vertex_count()); ++v) {
#pragma omp critical
		if(m_currImpBuffer[v].viewImportance > maxImp)
			maxImp = m_currImpBuffer[v].viewImportance;
	}
	return maxImp;
}


std::size_t ImportanceDecimater::get_original_vertex_count() const noexcept {
	return m_originalMesh.n_vertices();
}


std::size_t ImportanceDecimater::get_decimated_vertex_count() const noexcept {
	return m_decimatedMesh->n_vertices();
}


void ImportanceDecimater::decimate_with_error_quadrics(const std::size_t collapses) {
	if(collapses > 0u) {
		auto decimater = m_decimatedPoly->create_decimater();
		OpenMesh::Decimater::ModQuadricT<scene::geometry::PolygonMeshType>::Handle modQuadricHandle;
		CollapseTrackerModule<>::Handle trackerHandle;
		decimater.add(modQuadricHandle);
		decimater.add(trackerHandle);
		decimater.module(trackerHandle).set_properties(m_originalMesh, m_collapsed, m_collapsedTo);
		// Possibly repeat until we reached the desired count
		const std::size_t targetCollapses = std::min(collapses, m_originalPoly.get_vertex_count());
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

} // namespace mufflon::renderer::animation::silhouette::pt