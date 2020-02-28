#include "decimation_common_pt.hpp"
#include "util/parallel.hpp"
#include "util/punning.hpp"
#include "core/renderer/decimaters/silhouette/modules/importance_quadrics.hpp"
#include "core/renderer/decimaters/modules/collapse_tracker.hpp"
#include "core/renderer/decimaters/util/octree.inl"
#include "core/renderer/decimaters/util/float_octree.inl"
#include "core/scene/geometry/polygon.hpp"
#include <ei/vector.hpp>
#include <OpenMesh/Tools/Decimater/DecimaterT.hh>
#include <OpenMesh/Tools/Decimater/ModQuadricT.hh>

#include <gli/gli/texture3d.hpp>
#include <gli/gli/save.hpp>

namespace mufflon::renderer::decimaters::silhouette::pt {

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

template < Device dev >
ImportanceDecimater<dev>::ImportanceDecimater(StringView objectName, Lod& original, Lod& decimated,
											  const u32 clusterGridRes,
											  const float viewWeight, const float lightWeight,
											  const float shadowWeight, const float shadowSilhouetteWeight) :
	m_objectName(objectName),
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
	m_shadowSilhouetteWeight(shadowSilhouetteWeight),
	m_clusterGridRes{ clusterGridRes }
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

	// Perform initial decimation
	this->decimate_with_error_quadrics(m_clusterGridRes);
}

template < Device dev >
ImportanceDecimater<dev>::ImportanceDecimater(ImportanceDecimater<dev>&& dec) :
	m_original(dec.m_original),
	m_decimated(dec.m_decimated),
	m_originalPoly(dec.m_originalPoly),
	m_decimatedPoly(dec.m_decimatedPoly),
	m_originalMesh(dec.m_originalMesh),
	m_decimatedMesh(dec.m_decimatedMesh),
	m_importances(std::move(dec.m_importances)),
	m_devImportances(std::move(dec.m_devImportances)),
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

template < Device dev >
ImportanceDecimater<dev>::~ImportanceDecimater() {
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

template < Device dev >
void ImportanceDecimater<dev>::copy_back_normalized_importance() {
	(void)0;
	copy(m_devImportances.get(), m_importances.get(), sizeof(Importances<Device::CUDA>) * m_decimatedPoly->get_vertex_count());
}

template <>
void ImportanceDecimater<Device::CPU>::copy_back_normalized_importance() {
	(void)0;
	// No need to copy anything
}

template < Device dev >
void ImportanceDecimater<dev>::pull_importance_from_device() {
	copy(m_importances.get(), m_devImportances.get(), sizeof(Importances<Device::CUDA>) * m_decimatedPoly->get_vertex_count());
}

template < >
void ImportanceDecimater<Device::CPU>::pull_importance_from_device() {
	(void)0;
	// No need to copy anything
}

template < Device dev >
void ImportanceDecimater<dev>::update_importance_density(const ImportanceSums& sum, const bool useCurvature) {
	// First get the data off the GPU
	this->pull_importance_from_device();


	double importanceSum = 0.0;
	// Update our statistics: the importance density of each vertex
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

	const float* curvature = nullptr;
	if(useCurvature) {
		curvature = m_originalPoly.template acquire_const<Device::CPU, float>(m_originalPoly.get_curvature_hdl().value());
		importanceSum = 0.0;
	}
	// Map the importance back to the original mesh
	for(auto iter = m_originalMesh.vertices_begin(); iter != m_originalMesh.vertices_end(); ++iter) {
		const auto vertex = *iter;
		// Traverse collapse chain and snatch importance
		auto v = vertex;
		while(m_originalMesh.property(m_collapsed, v))
			v = m_originalMesh.property(m_collapsedTo, v);

		const auto importance = std::max(0.f, cuda::atomic_load<dev, float>(m_importances[m_originalMesh.property(m_collapsedTo, v).idx()].viewImportance));
		if(useCurvature) {
			const auto area = compute_area(m_originalMesh, vertex);
			const auto curv = std::abs(curvature[vertex.idx()]);

			const auto weightedImportance = std::sqrt(importance * curv);
			importanceSum += std::sqrt(importance * area * curv);
			//m_importances[m_originalMesh.property(m_collapsedTo, v).idx()].viewImportance = weightedImportance;
			// Put importance into temporary storage
			//m_originalMesh.property(m_accumulatedImportanceDensity, vertex) = weightedImportance;
		}
		//} else {
			m_originalMesh.property(m_accumulatedImportanceDensity, vertex) = importance;
		//}
	}

	// Subtract the shadow silhouette importance and use shadow importance instead
	logPedantic("Importance sum/shadow/silhouette(", m_objectName, "): ", importanceSum,
				" ", sum.shadowImportance, " ", sum.shadowSilhouetteImportance);
	m_importanceSum = importanceSum + m_shadowWeight * sum.shadowImportance - sum.shadowSilhouetteImportance;
}

template < Device dev >
void ImportanceDecimater<dev>::update_importance_density(const ImportanceSums& sum,
														 FloatOctree& viewGrid,
														 const SampleOctree& irradianceGrid) {
	// First get the data off the GPU
	this->pull_importance_from_device();

	viewGrid.join(irradianceGrid, m_lightWeight);
	viewGrid.export_to_file(std::string(m_objectName) + "-merged.ktx", 8u);

	// TODO: proper reservation
	std::vector<u32> vertexCount(viewGrid.capacity());

	for(i64 i = 0; i < static_cast<i64>(m_decimatedMesh->n_vertices()); ++i) {
		const auto vertex = m_decimatedMesh->vertex_handle(static_cast<u32>(i));
		const auto pos = util::pun<ei::Vec3>(m_decimatedMesh->point(vertex));
		//const auto id = viewGrid.get_node_id(pos);
		//vertexCount[id.index] += 1u;
	}

	// Update our statistics: the importance density of each vertex
//#pragma PARALLEL_REDUCTION(+, importanceSum)
	for(i64 i = 0; i < static_cast<i64>(m_decimatedMesh->n_vertices()); ++i) {
		const auto vertex = m_decimatedMesh->vertex_handle(static_cast<u32>(i));
		// Important: only works for triangles!
		const float area = compute_area(*m_decimatedMesh, vertex);
		const auto pos = util::pun<ei::Vec3>(m_decimatedMesh->point(vertex));
		const auto normal = util::pun<ei::Vec3>(m_decimatedMesh->normal(vertex));

		// TODO

		// In contrast to saving importance per vertex, we have to divide by the sample count here
		// because we have to distribute the value across many vertices
		//const auto id = viewGrid.get_node_id(pos);
		//const auto viewSample = viewGrid.get_density(pos, normal, id, true);
		//const auto viewSample = viewGrid.get_samples(id).value;
		//const auto sum = viewSample / static_cast<float>(vertexCount[id.index]);

		//m_importances[vertex.idx()].viewImportance = sum;// / area;
	}

	double importanceSum = 0.0;
	const auto* curvature = m_originalPoly.template acquire_const<Device::CPU, float>(m_originalPoly.get_curvature_hdl().value());
	// Map the importance back to the original mesh
	for(auto iter = m_originalMesh.vertices_begin(); iter != m_originalMesh.vertices_end(); ++iter) {
		const auto vertex = *iter;
		// Traverse collapse chain and snatch importance
		auto v = vertex;
		while(m_originalMesh.property(m_collapsed, v))
			v = m_originalMesh.property(m_collapsedTo, v);

		const auto importance = cuda::atomic_load<dev, float>(m_importances[m_originalMesh.property(m_collapsedTo, v).idx()].viewImportance);
		const auto area = compute_area(m_originalMesh, vertex);
		const auto curv = std::abs(curvature[vertex.idx()]);

		const auto weightedImportance = std::sqrt(importance * curv);
		importanceSum += std::sqrt(importance * area * curv);

		// Put importance into temporary storage
		m_originalMesh.property(m_accumulatedImportanceDensity, vertex) = weightedImportance;
	}

	// Subtract the shadow silhouette importance and use shadow importance instead
	logPedantic("Importance sum/shadow/silhouette(", m_objectName, "): ", importanceSum,
				" ", sum.shadowImportance, " ", sum.shadowSilhouetteImportance);
	m_importanceSum = importanceSum + m_shadowWeight * sum.shadowImportance - sum.shadowSilhouetteImportance;
}

template < Device dev >
void ImportanceDecimater<dev>::update_importance_density(const ImportanceSums& sum,
														 const data_structs::DmHashGrid<float>& viewGrid,
														 const data_structs::DmHashGrid<float>& irradianceGrid,
														 const data_structs::DmHashGrid<u32>& irradianceCount) {
	// First get the data off the GPU
	this->pull_importance_from_device();

	// TODO: proper reservation
	std::vector<u32> vertexCount(viewGrid.capacity());

	for(i64 i = 0; i < static_cast<i64>(m_decimatedMesh->n_vertices()); ++i) {
		const auto vertex = m_decimatedMesh->vertex_handle(static_cast<u32>(i));
		const auto pos = util::pun<ei::Vec3>(m_decimatedMesh->point(vertex));
		const auto id = viewGrid.get_cell_index(pos);
		vertexCount[id] += 1u;
	}

	// Update our statistics: the importance density of each vertex
	float importanceSum = 0.0;
//#pragma PARALLEL_REDUCTION(+, importanceSum)
	for(i64 i = 0; i < static_cast<i64>(m_decimatedMesh->n_vertices()); ++i) {
		const auto vertex = m_decimatedMesh->vertex_handle(static_cast<u32>(i));
		// Important: only works for triangles!
		const float area = compute_area(*m_decimatedMesh, vertex);
		const auto pos = util::pun<ei::Vec3>(m_decimatedMesh->point(vertex));
		const auto normal = util::pun<ei::Vec3>(m_decimatedMesh->normal(vertex));
		const auto flux = irradianceGrid.get_density(pos, normal)
			/ std::max(1.f, static_cast<float>(irradianceCount.get_density(pos, normal)));

		const auto id = viewGrid.get_cell_index(pos);
		const auto count = viewGrid.get_count(id);
		const auto viewImportance = count / std::max(1.f, static_cast<float>(vertexCount[id]));
		if(isnan(viewImportance)) {
			printf("%f %d %d\n", count, id, vertexCount[id]);
			fflush(stdout);
			__debugbreak();
		}
		//const auto viewImportance = viewGrid.get_density(pos, normal);

		const float importance = viewImportance;// +m_lightWeight * flux;

		importanceSum += importance;
		m_importances[vertex.idx()].viewImportance = importance;
	}

	// Subtract the shadow silhouette importance and use shadow importance instead
	logPedantic("Importance sum/shadow/silhouette(", m_objectName, "): ", importanceSum, " ", sum.shadowImportance, " ", sum.shadowSilhouetteImportance);
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

template < Device dev >
void ImportanceDecimater<dev>::recompute_geometric_vertex_normals() {
#pragma PARALLEL_FOR
	for(i64 i = 0; i < static_cast<i64>(m_decimatedMesh->n_vertices()); ++i) {
		const auto vertex = m_decimatedMesh->vertex_handle(static_cast<u32>(i));

		typename Mesh::Normal normal;
#pragma warning(push)
#pragma warning(disable : 4244)
		m_decimatedMesh->calc_vertex_normal_correct(vertex, normal);
#pragma warning(pop)
		const auto n = ei::normalize(util::pun<ei::Vec3>(normal));
		m_decimatedMesh->set_normal(vertex, util::pun<typename Mesh::Normal>(n));
	}
}

template < Device dev >
ArrayDevHandle_t<dev, Importances<dev>> ImportanceDecimater<dev>::start_iteration() {
	// Resize importance map
	m_importances = std::make_unique<Importances<dev>[]>(m_decimatedPoly->get_vertex_count());
	m_devImportances = make_udevptr_array<dev, Importances<dev>, false>(m_decimatedPoly->get_vertex_count());
	cuda::check_error(cudaMemset(m_devImportances.get(), 0, sizeof(Importances<dev>) * m_decimatedPoly->get_vertex_count()));
	return m_devImportances.get();
}

template < >
ArrayDevHandle_t<Device::CPU, Importances<Device::CPU>> ImportanceDecimater<Device::CPU>::start_iteration() {
	// Resize importance map (only one on the CPU since we got direct access)
	m_importances = std::make_unique<Importances<Device::CPU>[]>(m_decimatedPoly->get_vertex_count());
	std::memset(m_importances.get(), 0, sizeof(Importances<Device::CPU>) * m_decimatedPoly->get_vertex_count());
	return m_importances.get();
}

template < Device dev >
void ImportanceDecimater<dev>::iterate(const std::size_t targetCount, const FloatOctree* view) {
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

	if(targetCount < get_original_vertex_count()) {
		const auto t0 = std::chrono::high_resolution_clock::now();
		std::size_t collapses = 0u;
		// TODO
		/*if(view != nullptr)
			collapses = m_decimatedPoly->cluster(*view, targetCount, false);
		else*/
			collapses = m_decimatedPoly->decimate(decimater, targetCount, false);
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
			this->recompute_geometric_vertex_normals();
			m_decimated.clear_accel_structure();
		}
		logPedantic("Performed ", collapses, " collapses for object '", m_objectName, "', remaining vertices: ", m_decimatedMesh->n_vertices());
	}
}

// Utility only
template < Device dev >
ImportanceDecimater<dev>::Mesh::VertexHandle ImportanceDecimater<dev>::get_original_vertex_handle(const Mesh::VertexHandle decimatedHandle) const {
	mAssert(decimatedHandle.is_valid());
	mAssert(static_cast<std::size_t>(decimatedHandle.idx()) < m_decimatedMesh->n_vertices());
	const auto originalHandle = m_decimatedMesh->property(m_originalVertex, decimatedHandle);
	mAssert(originalHandle.is_valid());
	mAssert(static_cast<std::size_t>(originalHandle.idx()) < m_originalPoly.get_vertex_count());
	return originalHandle;
}

template < Device dev >
float ImportanceDecimater<dev>::get_current_max_importance() const {
	float maxImp = 0.f;
#pragma PARALLEL_FOR
	for(i64 v = 0; v < static_cast<i64>(m_decimatedPoly->get_vertex_count()); ++v) {
		const auto imp = cuda::atomic_load<dev, float>(m_importances[v].viewImportance);
		if(!std::isinf(imp) && !std::isnan(imp)) {
#pragma omp critical
			if(m_importances[v].viewImportance > maxImp)
				maxImp = m_importances[v].viewImportance;
		}
	}
	return maxImp;
}

template < Device dev >
std::size_t ImportanceDecimater<dev>::get_original_vertex_count() const noexcept {
	return m_originalMesh.n_vertices();
}

template < Device dev >
std::size_t ImportanceDecimater<dev>::get_decimated_vertex_count() const noexcept {
	return m_decimatedMesh->n_vertices();
}

template < Device dev >
void ImportanceDecimater<dev>::decimate_with_error_quadrics(const u32 clusterGridRes) {
	if(clusterGridRes > 0u) {
		auto decimater = m_decimatedPoly->create_decimater();
		OpenMesh::Decimater::ModQuadricT<scene::geometry::PolygonMeshType>::Handle modQuadricHandle;
		CollapseTrackerModule<>::Handle trackerHandle;
		decimater.add(modQuadricHandle);
		decimater.add(trackerHandle);
		decimater.module(trackerHandle).set_properties(m_originalMesh, m_collapsed, m_collapsedTo);
		// Possibly repeat until we reached the desired count
		//const std::size_t targetCollapses = std::min(collapses, m_originalPoly.get_vertex_count());
		//const std::size_t targetVertexCount = m_originalPoly.get_vertex_count() - targetCollapses;
		//std::size_t performedCollapses = m_decimatedPoly->decimate(decimater, targetVertexCount, false);
		const auto performedCollapses = m_decimatedPoly->cluster(clusterGridRes, false);
		/*if(performedCollapses < targetCollapses)
			logWarning("Not all decimations were performed: ", targetCollapses - performedCollapses, " missing");*/
		m_decimatedPoly->garbage_collect([this](Mesh::VertexHandle deletedVertex, Mesh::VertexHandle changedVertex) {
			// Adjust the reference from original to decimated mesh
			const auto originalVertex = this->get_original_vertex_handle(changedVertex);
			if(!m_originalMesh.property(m_collapsed, originalVertex))
				m_originalMesh.property(m_collapsedTo, originalVertex) = deletedVertex;
		});
		//this->recompute_geometric_vertex_normals();

		m_decimated.clear_accel_structure();
	}
}

template class ImportanceDecimater<Device::CPU>;
template class ImportanceDecimater<Device::CUDA>;

} // namespace mufflon::renderer::decimaters::silhouette::pt