#include "sil_decimater.hpp"

namespace mufflon::renderer::silhouette {

template < class MeshT >
ImportanceModule<MeshT>::ImportanceModule(MeshT &mesh) :
	Base(mesh, false) {}

template < class MeshT >
void ImportanceModule<MeshT>::initialize() {
	// TODO
}

template < class MeshT >
float ImportanceModule<MeshT>::collapse_priority(const CollapseInfo& ci) {
	const auto& propHandle = m_importanceMap->get_importance_property(m_meshIndex);
	float importance = Base::mesh().property(propHandle, ci.v0);
	u32 count = 0u;
	for(auto ringVertexHandle = Base::mesh().vv_iter(ci.v0); ringVertexHandle.is_valid(); ++ringVertexHandle) {
		importance += Base::mesh().property(propHandle, *ringVertexHandle);
		++count;
	}
	importance /= static_cast<float>(count);
	if(importance > m_threshold)
		return -1.f;
	return importance;
}

template < class MeshT >
void ImportanceModule<MeshT>::use_collapse_history(bool val) {
	m_useCollapseHistory = val;
}

template < class MeshT >
void ImportanceModule<MeshT>::set_importance_map(ImportanceMap& importanceMap, const u32 meshIndex, const float threshold) {
	m_importanceMap = &importanceMap;
	m_meshIndex = meshIndex;
	m_threshold = threshold;
}

template < class MeshT >
void ImportanceModule<MeshT>::preprocess_collapse(const CollapseInfo& ci) {
}

// Post-process halfedge collapse (accumulate importance)
template < class MeshT >
void ImportanceModule<MeshT>::postprocess_collapse(const CollapseInfo& ci) {
	vertex_split(Base::mesh(), ci.v0, ci.v1, ci.vl, ci.vr);
	m_importanceMap->collapse(m_meshIndex, ci);
}

OpenMesh::HalfedgeHandle insert_loop(scene::geometry::PolygonMeshType& mesh, const OpenMesh::HalfedgeHandle hh) {
	using namespace OpenMesh;
	HalfedgeHandle  h0(hh);
	HalfedgeHandle  o0(mesh.opposite_halfedge_handle(h0));

	VertexHandle    v0(mesh.to_vertex_handle(o0));
	VertexHandle    v1(mesh.to_vertex_handle(h0));

	HalfedgeHandle  h1 = mesh.new_edge(v1, v0);
	HalfedgeHandle  o1 = mesh.opposite_halfedge_handle(h1);

	FaceHandle      f0 = mesh.face_handle(h0);
	FaceHandle      f1 = mesh.new_face();

	// halfedge -> halfedge
	mesh.set_next_halfedge_handle(mesh.prev_halfedge_handle(h0), o1);
	mesh.set_next_halfedge_handle(o1, mesh.next_halfedge_handle(h0));
	mesh.set_next_halfedge_handle(h1, h0);
	mesh.set_next_halfedge_handle(h0, h1);

	// halfedge -> face
	mesh.set_face_handle(o1, f0);
	mesh.set_face_handle(h0, f1);
	mesh.set_face_handle(h1, f1);

	// face -> halfedge
	mesh.set_halfedge_handle(f1, h0);
	if(f0.is_valid())
		mesh.set_halfedge_handle(f0, o1);


	// vertex -> halfedge
	mesh.adjust_outgoing_halfedge(v0);
	mesh.adjust_outgoing_halfedge(v1);

	return h1;
}

OpenMesh::HalfedgeHandle insert_edge(scene::geometry::PolygonMeshType& mesh, const OpenMesh::VertexHandle _vh,
									 const OpenMesh::HalfedgeHandle _h0, const OpenMesh::HalfedgeHandle _h1) {
	using namespace OpenMesh;
	mAssert(_h0.is_valid() && _h1.is_valid());

	VertexHandle  v0 = _vh;
	VertexHandle  v1 = mesh.to_vertex_handle(_h0);

	mAssert(v1 == mesh.to_vertex_handle(_h1));

	HalfedgeHandle v0v1 = mesh.new_edge(v0, v1);
	HalfedgeHandle v1v0 = mesh.opposite_halfedge_handle(v0v1);



	// vertex -> halfedge
	mesh.set_halfedge_handle(v0, v0v1);
	mesh.set_halfedge_handle(v1, v1v0);


	// halfedge -> halfedge
	mesh.set_next_halfedge_handle(v0v1, mesh.next_halfedge_handle(_h0));
	mesh.set_next_halfedge_handle(_h0, v0v1);
	mesh.set_next_halfedge_handle(v1v0, mesh.next_halfedge_handle(_h1));
	mesh.set_next_halfedge_handle(_h1, v1v0);


	// halfedge -> vertex
	for(auto vih_it = mesh.vih_ccwiter(v0); vih_it.is_valid(); ++vih_it)
		mesh.set_vertex_handle(*vih_it, v0);


	// halfedge -> face
	mesh.set_face_handle(v0v1, mesh.face_handle(_h0));
	mesh.set_face_handle(v1v0, mesh.face_handle(_h1));


	// face -> halfedge
	if(mesh.face_handle(v0v1).is_valid())
		mesh.set_halfedge_handle(mesh.face_handle(v0v1), v0v1);
	if(mesh.face_handle(v1v0).is_valid())
		mesh.set_halfedge_handle(mesh.face_handle(v1v0), v1v0);


	// vertex -> halfedge
	mesh.adjust_outgoing_halfedge(v0);
	mesh.adjust_outgoing_halfedge(v1);


	return v0v1;
}

OpenMesh::HalfedgeHandle vertex_split(scene::geometry::PolygonMeshType& mesh,
									  const OpenMesh::VertexHandle v0, const OpenMesh::VertexHandle v1,
									  const OpenMesh::VertexHandle vl, const OpenMesh::VertexHandle vr) {
	using namespace OpenMesh;
	HalfedgeHandle v1vl, vlv1, vrv1, v0v1;

	// build loop from halfedge v1->vl
	if(vl.is_valid()) {
		v1vl = mesh.find_halfedge(v1, vl);
		mAssert(v1vl.is_valid());
		vlv1 = insert_loop(mesh, v1vl);
	}

	// build loop from halfedge vr->v1
	if(vr.is_valid()) {
		vrv1 = mesh.find_halfedge(vr, v1);
		mAssert(vrv1.is_valid());
		insert_loop(mesh, vrv1);
	}

	// handle boundary cases
	if(!vl.is_valid())
		vlv1 = mesh.prev_halfedge_handle(mesh.halfedge_handle(v1));
	if(!vr.is_valid())
		vrv1 = mesh.prev_halfedge_handle(mesh.halfedge_handle(v1));


	// split vertex v1 into edge v0v1
	v0v1 = insert_edge(mesh, v0, vlv1, vrv1);


	return v0v1;
}

// TODO: maybe one day I'll manage to get it to work properly even for multiple deleted vertices in a chain...
#if 0
inline void restore_collapsed_edge(scene::geometry::PolygonMeshType& mesh,
								   const OpenMesh::VertexHandle v0, const OpenMesh::HalfedgeHandle v1v0,
								   const OpenMesh::HalfedgeHandle v0vl, const OpenMesh::HalfedgeHandle vrv0) {

	// How things used to be before the collapse
	// TODO: what if vl/vr collapse away?
	const auto v0vl = ci.v0vl;
	const auto vrv0 = ci.vrv0;
	const auto v1v0 = ci.v1v0;

	const auto v0 = ci.v0;
	const auto vl = mesh.to_vertex_handle(v0vl);
	const auto vr = mesh.from_vertex_handle(vrv0);
	const auto v1 = mesh.to_vertex_handle(v1v0);

	const auto v0v1 = mesh.opposite_halfedge_handle(v1v0);
	const auto vlv0 = mesh.opposite_halfedge_handle(v0vl);
	const auto v0vr = mesh.opposite_halfedge_handle(vrv0);
	const auto vlv1 = mesh.find_halfedge(vl, v1);
	const auto vrv1 = mesh.find_halfedge(vr, v1);
	const auto v1vl = mesh.opposite_halfedge_handle(vlv1);
	const auto v1vr = mesh.opposite_halfedge_handle(vrv1);

	const auto fl = mesh.face_handle(v0v1);
	const auto fr = mesh.face_handle(ci.v1v0);

	// Restore left loop
	// Adjust halfedge -> halfedge
	mesh.set_next_halfedge_handle(v0vl, mesh.next_halfedge_handle(v1vl));
	mesh.set_next_halfedge_handle(mesh.prev_halfedge_handle(v1vl), v0vl);
	mesh.set_next_halfedge_handle(v0v1, v1vl);
	mesh.set_next_halfedge_handle(v1vl, vlv0);
	mesh.set_next_halfedge_handle(vlv0, v0v1);
	// Adjust halfedge -> face
	mesh.set_face_handle(v1vl, fl);
	mesh.set_face_handle(vlv0, fl);
	mesh.set_face_handle(v0v1, fl);
	// Adjust face -> halfedge
	mesh.set_halfedge_handle(fl, v0v1);

	// Restore right loop
	// Adjust halfedge -> halfedge
	mesh.set_next_halfedge_handle(vrv0, mesh.next_halfedge_handle(vrv1));
	mesh.set_next_halfedge_handle(mesh.prev_halfedge_handle(vrv1), vrv0);
	mesh.set_next_halfedge_handle(v1v0, v0vr);
	mesh.set_next_halfedge_handle(v0vr, vrv1);
	mesh.set_next_halfedge_handle(vrv1, v1v0);
	// Adjust halfedge -> face
	mesh.set_face_handle(v1v0, fr);
	mesh.set_face_handle(v0vr, fr);
	mesh.set_face_handle(vrv1, fr);
	// Adjust face -> halfedge
	mesh.set_halfedge_handle(fr, v1v0);

	// Fix some other things
	// Adjust vertex -> halfedge
	mesh.set_halfedge_handle(v0, v0v1);
	mesh.set_halfedge_handle(v1, v1v0);
	// Adjust halfedge -> vertex
	mesh.set_vertex_handle(v0v1, v1);
	mesh.set_vertex_handle(v1v0, v0);

	// Adjust the halfedge -> vertex between vl and vr
	mesh.set_vertex_handle(mesh.opposite_halfedge_handle(v0vl), v0);
	mesh.set_vertex_handle(mesh.opposite_halfedge_handle(v0vr), v0);
	for(auto curr = mesh.prev_halfedge_handle(v0vl); curr != vrv0;
		curr = mesh.prev_halfedge_handle(mesh.opposite_halfedge_handle(curr))) {
		const auto to = mesh.to_vertex_handle(curr);
		const auto from = mesh.from_vertex_handle(curr);
		mesh.set_vertex_handle(curr, v0);
	}

	// Adjust vertex -> halfedge
	mesh.adjust_outgoing_halfedge(v0);
	mesh.adjust_outgoing_halfedge(v1);

	mesh.status(v0v1).set_deleted(false);
	mesh.status(v1v0).set_deleted(false);
	mesh.status(v0).set_deleted(false);
	mesh.status(fl).set_deleted(false);
	mesh.status(fr).set_deleted(false);
}
#endif // 0

template class ImportanceModule<scene::geometry::PolygonMeshType>;

} // namespace mufflon::renderer::silhouette