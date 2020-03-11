#include "combined_decimater.hpp"
#include "combined_params.hpp"
#include "core/renderer/decimaters/util/octree.inl"
#include "core/renderer/decimaters/util/float_octree.inl"
#include "core/renderer/decimaters/modules/collapse_tracker.hpp"
#include "core/renderer/decimaters/combined/modules/importance_quadrics.hpp"
#include "core/scene/lod.hpp"
#include "core/scene/geometry/polygon.hpp"
#include "core/scene/geometry/util.hpp"
#include <OpenMesh/Tools/Decimater/DecimaterT.hh>

namespace mufflon::renderer::decimaters::combined {

CombinedDecimater::CombinedDecimater(StringView objectName, scene::Lod& original,
									 scene::Lod& decimated, const u32 frameCount,
									 ArrayDevHandle_t<Device::CPU, FloatOctree*> view,
									 ArrayDevHandle_t<Device::CPU, SampleOctree*> irradiance,
									 ArrayDevHandle_t<Device::CPU, double> importanceSums,
									 const float lightWeight) :
	m_objectName{ objectName },
	m_original{ original },
	m_decimated{ decimated },
	m_originalPoly{ m_original.template get_geometry<scene::geometry::Polygons>() },
	m_decimatedPoly{ &m_decimated.template get_geometry<scene::geometry::Polygons>() },
	m_originalMesh{ m_originalPoly.get_mesh() },
	m_decimatedMesh{ &m_decimatedPoly->get_mesh() },
	m_viewImportance{ view },
	m_irradianceImportance{ irradiance },
	m_frameCount{ frameCount },
	m_importanceSums{ importanceSums },
	m_originalVertex{},
	m_accumulatedImportanceDensity{},
	m_collapsedTo{},
	m_collapsed{},
	m_lightWeight{ lightWeight }
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

	// Add curvature
	m_originalPoly.compute_curvature();

	// Perform initial decimation (TODO)
	//this->decimate_with_error_quadrics(m_clusterGridRes);
}

CombinedDecimater::CombinedDecimater(CombinedDecimater&& other) :
	m_objectName{ other.m_objectName },
	m_original{ other.m_original},
	m_decimated{ other.m_decimated },
	m_originalPoly{ other.m_originalPoly },
	m_decimatedPoly{ other.m_decimatedPoly },
	m_originalMesh{ other.m_originalMesh },
	m_decimatedMesh{ other.m_decimatedMesh },
	m_viewImportance{ other.m_viewImportance },
	m_irradianceImportance{ other.m_irradianceImportance },
	m_frameCount{ other.m_frameCount },
	m_importanceSums{ other.m_importanceSums },
	m_originalVertex{ other .m_originalVertex },
	m_accumulatedImportanceDensity{ other.m_accumulatedImportanceDensity },
	m_collapsedTo{ other.m_collapsedTo },
	m_collapsed{ other.m_collapsed },
	m_lightWeight{ other.m_lightWeight }
{
	// Request the status again since it will get removed once in the destructor
	m_decimatedMesh->request_vertex_status();
	m_decimatedMesh->request_edge_status();
	m_decimatedMesh->request_halfedge_status();
	m_decimatedMesh->request_face_status();
	// Invalidate the handles here so we know not to remove them in the destructor
	other.m_originalVertex.invalidate();
	other.m_accumulatedImportanceDensity.invalidate();
	other.m_collapsedTo.invalidate();
	other.m_collapsed.invalidate();

	// Make sure that the curvature isn't actually removed
	m_originalPoly.compute_curvature();
}

CombinedDecimater::~CombinedDecimater() {
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

	// Remove curvature reference again
	m_originalPoly.remove_curvature();
}

void CombinedDecimater::finish_gather(const u32 frame) {
	mAssert(frame < m_frameCount);
	// Join the irradiance octree into our main octree
	const auto t0 = std::chrono::high_resolution_clock::now();
	m_viewImportance[frame]->join(*m_irradianceImportance[frame], m_lightWeight);
	const auto t1 = std::chrono::high_resolution_clock::now();
	logPedantic(m_objectName, ": join time ", std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count(), "ms");
	// Also compute the importance sums
	// TODO
	m_importanceSums[frame] = m_viewImportance[frame]->compute_leaf_sum();
}

void CombinedDecimater::update(const PImpWeightMethod::Values weighting,
							   u32 startFrame, u32 endFrame) {
	mAssert(endFrame < m_frameCount);
	const auto* curvature = m_originalPoly.template acquire_const<Device::CPU, float>(m_originalPoly.get_curvature_hdl().value());

	// Map the importance back to the original mesh
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
					// Fetch octree value (TODO)
					const auto imp = m_viewImportance[f]->get_density(util::pun<ei::Vec3>(m_originalMesh.point(vertex)),
																	  util::pun<ei::Vec3>(m_originalMesh.normal(vertex)));
					importance += imp;
				}
				importance /= static_cast<float>(endFrame - startFrame + 1u);
				break;
			case PImpWeightMethod::Values::MAX_ALL:
				startFrame = 0u;
				endFrame = m_frameCount - 1u;
				[[fallthrough]];
			case PImpWeightMethod::Values::MAX:
				for(u32 f = startFrame; f <= endFrame; ++f) {
					// Fetch octree value (TODO)
					const auto imp = m_viewImportance[f]->get_density(util::pun<ei::Vec3>(m_originalMesh.point(vertex)),
																	  util::pun<ei::Vec3>(m_originalMesh.normal(vertex)));
					importance = std::max<float>(importance, imp);
				}
				break;
		}
		const auto weightedImportance = std::sqrt(importance * curv) / area;
		m_originalMesh.property(m_accumulatedImportanceDensity, vertex) = weightedImportance;
	}
}

void CombinedDecimater::reduce(const std::size_t targetVertexCount, const float maxDensity,
							   const u32 frame) {
	// Reset the collapse property
	for(auto vertex : m_originalMesh.vertices()) {
		m_originalMesh.property(m_collapsed, vertex) = false;
		m_originalMesh.property(m_collapsedTo, vertex) = vertex;
	}

	// Recreate the LoD for decimation
	m_decimated.~Lod();
	new(&m_decimated) scene::Lod(m_original);
	// Refetch the affected pointers
	m_decimatedPoly = &m_decimated.template get_geometry<scene::geometry::Polygons>();
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
	decimaters::modules::CollapseTrackerModule<>::Handle trackerHandle;
	modules::ImportanceDecimationModule<>::Handle impHandle;
	decimater.add(trackerHandle);
	decimater.add(impHandle);
	decimater.module(trackerHandle).set_properties(m_originalMesh, m_collapsed, m_collapsedTo);
	decimater.module(impHandle).set_properties(m_originalMesh, m_accumulatedImportanceDensity);

	if(targetVertexCount < get_original_vertex_count()) {
		const auto t0 = std::chrono::high_resolution_clock::now();
		std::size_t collapses = 0u;
		// TODO
		/*if(view != nullptr)
			collapses = m_decimatedPoly->cluster(*view, targetCount, false);
		else*/
		collapses = m_decimatedPoly->cluster_decimate(*m_viewImportance[frame], decimater,
													  targetVertexCount, maxDensity);
		//collapses = m_decimatedPoly->decimate(decimater, targetVertexCount, false);
		//const auto collapses = m_decimatedPoly->decimate(decimater, targetCount, false);
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
			//this->recompute_geometric_vertex_normals();
			m_decimated.clear_accel_structure();
		}
		logPedantic("Performed ", collapses, " collapses for object '", m_objectName, "', remaining vertices: ", m_decimatedMesh->n_vertices());
	}
}


std::size_t CombinedDecimater::get_original_vertex_count() const noexcept {
	return m_originalMesh.n_vertices();
}
std::size_t CombinedDecimater::get_decimated_vertex_count() const noexcept {
	return m_decimatedMesh->n_vertices();
}

CombinedDecimater::VertexHandle CombinedDecimater::get_original_vertex_handle(const VertexHandle decimatedHandle) const noexcept {
	mAssert(decimatedHandle.is_valid());
	mAssert(static_cast<std::size_t>(decimatedHandle.idx()) < m_decimatedMesh->n_vertices());
	const auto originalHandle = m_decimatedMesh->property(m_originalVertex, decimatedHandle);
	mAssert(originalHandle.is_valid());
	mAssert(static_cast<std::size_t>(originalHandle.idx()) < m_originalPoly.get_vertex_count());
	return originalHandle;
}

} // namespace mufflon::renderer::decimaters::combined