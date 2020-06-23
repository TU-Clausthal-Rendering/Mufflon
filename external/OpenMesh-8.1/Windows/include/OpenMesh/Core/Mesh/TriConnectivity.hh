/* ========================================================================= *
 *                                                                           *
 *                               OpenMesh                                    *
 *           Copyright (c) 2001-2015, RWTH-Aachen University                 *
 *           Department of Computer Graphics and Multimedia                  *
 *                          All rights reserved.                             *
 *                            www.openmesh.org                               *
 *                                                                           *
 *---------------------------------------------------------------------------*
 * This file is part of OpenMesh.                                            *
 *---------------------------------------------------------------------------*
 *                                                                           *
 * Redistribution and use in source and binary forms, with or without        *
 * modification, are permitted provided that the following conditions        *
 * are met:                                                                  *
 *                                                                           *
 * 1. Redistributions of source code must retain the above copyright notice, *
 *    this list of conditions and the following disclaimer.                  *
 *                                                                           *
 * 2. Redistributions in binary form must reproduce the above copyright      *
 *    notice, this list of conditions and the following disclaimer in the    *
 *    documentation and/or other materials provided with the distribution.   *
 *                                                                           *
 * 3. Neither the name of the copyright holder nor the names of its          *
 *    contributors may be used to endorse or promote products derived from   *
 *    this software without specific prior written permission.               *
 *                                                                           *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS       *
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED *
 * TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A           *
 * PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER *
 * OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,  *
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,       *
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR        *
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF    *
 * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING      *
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS        *
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.              *
 *                                                                           *
 * ========================================================================= */



#ifndef OPENMESH_TRICONNECTIVITY_HH
#define OPENMESH_TRICONNECTIVITY_HH

#include <OpenMesh/Core/Mesh/PolyConnectivity.hh>

namespace OpenMesh {

/** \brief Connectivity Class for Triangle Meshes
*/
class OPENMESHDLLEXPORT TriConnectivity : public PolyConnectivity
{
public:

  TriConnectivity() {}
  virtual ~TriConnectivity() {}

  inline static bool is_triangles()
  { return true; }

  /** assign_connectivity() methods. See ArrayKernel::assign_connectivity()
      for more details. When the source connectivity is not triangles, in
      addition "fan" connectivity triangulation is performed*/
  inline void assign_connectivity(const TriConnectivity& _other)
  { PolyConnectivity::assign_connectivity(_other); }
  
  inline void assign_connectivity(const PolyConnectivity& _other)
  { 
    PolyConnectivity::assign_connectivity(_other); 
    triangulate();
  }
  
  /** \name Addding items to a mesh
  */

  //@{

  /** \brief Add a face with arbitrary valence to the triangle mesh
   *
   * Override OpenMesh::Mesh::PolyMeshT::add_face(). Faces that aren't
   * triangles will be triangulated and added. In this case an
   * invalid face handle will be returned.
   *
   *
   * */
  SmartFaceHandle add_face(const VertexHandle* _vhandles, size_t _vhs_size);

  /** \brief Add a face with arbitrary valence to the triangle mesh
     *
     * Override OpenMesh::Mesh::PolyMeshT::add_face(). Faces that aren't
     * triangles will be triangulated and added. In this case an
     * invalid face handle will be returned.
     *
     *
     * */
  SmartFaceHandle add_face(const std::vector<VertexHandle>& _vhandles);

  /** \brief Add a face with arbitrary valence to the triangle mesh
     *
     * Override OpenMesh::Mesh::PolyMeshT::add_face(). Faces that aren't
     * triangles will be triangulated and added. In this case an
     * invalid face handle will be returned.
     *
     *
     * */
  SmartFaceHandle add_face(const std::vector<SmartVertexHandle>& _vhandles);

  /** \brief Add a face to the mesh (triangle)
   *
   * This function adds a triangle to the mesh. The triangle is passed directly
   * to the underlying PolyConnectivity as we don't explicitly need to triangulate something.
   *
   * @param _vh0 VertexHandle 1
   * @param _vh1 VertexHandle 2
   * @param _vh2 VertexHandle 3
   * @return FaceHandle of the added face (invalid, if the operation failed)
   */
  SmartFaceHandle add_face(VertexHandle _vh0, VertexHandle _vh1, VertexHandle _vh2);
  
  //@}

  /** Returns the opposite vertex to the halfedge _heh in the face
      referenced by _heh returns InvalidVertexHandle if the _heh is
      boundary  */
  inline VertexHandle opposite_vh(HalfedgeHandle _heh) const
  {
    return is_boundary(_heh) ? InvalidVertexHandle :
                               to_vertex_handle(next_halfedge_handle(_heh));
  }

  /** Returns the opposite vertex to the opposite halfedge of _heh in
      the face referenced by it returns InvalidVertexHandle if the
      opposite halfedge is boundary  */
  VertexHandle opposite_he_opposite_vh(HalfedgeHandle _heh) const
  { return opposite_vh(opposite_halfedge_handle(_heh)); }

  /** \name Topology modifying operators
  */
  //@{


  /** Returns whether collapsing halfedge _heh is ok or would lead to
      topological inconsistencies.
      \attention This method need the Attributes::Status attribute and
      changes the \em tagged bit.  */
  bool is_collapse_ok(HalfedgeHandle v0v1) {
    // is the edge already deleted?
    if ( status(edge_handle(v0v1)).deleted() )
      return false;

    HalfedgeHandle  v1v0(opposite_halfedge_handle(v0v1));
    VertexHandle    v0(to_vertex_handle(v1v0));
    VertexHandle    v1(to_vertex_handle(v0v1));

    // are vertices already deleted ?
    if (status(v0).deleted() || status(v1).deleted())
      return false;

    VertexHandle    vl, vr;
    HalfedgeHandle  h1, h2;

    // the edges v1-vl and vl-v0 must not be both boundary edges
    if (!is_boundary(v0v1))
    {

      h1 = next_halfedge_handle(v0v1);
      h2 = next_halfedge_handle(h1);

      vl = to_vertex_handle(h1);

      if (is_boundary(opposite_halfedge_handle(h1)) &&
          is_boundary(opposite_halfedge_handle(h2)))
      {
        return false;
      }
    }


    // the edges v0-vr and vr-v1 must not be both boundary edges
    if (!is_boundary(v1v0))
    {

      h1 = next_halfedge_handle(v1v0);
      h2 = next_halfedge_handle(h1);

      vr = to_vertex_handle(h1);

      if (is_boundary(opposite_halfedge_handle(h1)) &&
          is_boundary(opposite_halfedge_handle(h2)))
        return false;
    }

    // if vl and vr are equal or both invalid -> fail
    if (vl == vr) return false;

    VertexVertexIter  vv_it;

    // test intersection of the one-rings of v0 and v1
    for (vv_it = vv_iter(v0); vv_it.is_valid(); ++vv_it)
      status(*vv_it).set_tagged(false);

    for (vv_it = vv_iter(v1); vv_it.is_valid(); ++vv_it)
      status(*vv_it).set_tagged(true);

    for (vv_it = vv_iter(v0); vv_it.is_valid(); ++vv_it)
      if (status(*vv_it).tagged() && *vv_it != vl && *vv_it != vr)
        return false;


    // edge between two boundary vertices should be a boundary edge
    if ( is_boundary(v0) && is_boundary(v1) &&
        !is_boundary(v0v1) && !is_boundary(v1v0))
      return false;

    // passed all tests
    return true;
  }

  /// Vertex Split: inverse operation to collapse().
  HalfedgeHandle vertex_split(VertexHandle v0, VertexHandle v1,
                              VertexHandle vl, VertexHandle vr);

  /// Check whether flipping _eh is topologically correct.
  bool is_flip_ok(EdgeHandle _eh) const;

  /** Flip edge _eh.
      Check for topological correctness first using is_flip_ok(). */
  void flip(EdgeHandle _eh);


  /** \brief Edge split (= 2-to-4 split)
   *
   *
   * The function will introduce two new faces ( non-boundary case) or
   * one additional face (if edge is boundary)
   *
   * \note The properties of the new edges, halfedges, and faces will be undefined!
   *
   * @param _eh Edge handle that should be split
   * @param _vh Vertex handle that will be inserted at the edge
   */
  void split(EdgeHandle _eh, VertexHandle _vh);

  /** \brief Edge split (= 2-to-4 split)
     *
     *
     * The function will introduce two new faces ( non-boundary case) or
     * one additional face (if edge is boundary)
     *
     * \note The properties of the new edges, halfedges, and faces will be undefined!
     *
     * \note This is an override to prevent a direct call to PolyConnectivity split_edge,
     *       which would introduce a singular vertex with valence 2 which is not allowed
     *       on TriMeshes
     *
     * @param _eh Edge handle that should be split
     * @param _vh Vertex handle that will be inserted at the edge
     */
  inline void split_edge(EdgeHandle _eh, VertexHandle _vh) { TriConnectivity::split(_eh, _vh); }

  /** \brief Edge split (= 2-to-4 split)
   *
   * The function will introduce two new faces ( non-boundary case) or
   * one additional face (if edge is boundary)
   *
   * \note The properties of the new edges and faces will be adjusted to the
   *       properties of the original edge and face
   * \note The properties of the new halfedges will be undefined
   *
   * @param _eh Edge handle that should be split
   * @param _vh Vertex handle that will be inserted at the edge
   */
  void split_copy(EdgeHandle _eh, VertexHandle _vh);

  /** \brief Edge split (= 2-to-4 split)
   *
   * The function will introduce two new faces ( non-boundary case) or
   * one additional face (if edge is boundary)
   *
   * \note The properties of the new edges and faces will be adjusted to the
   *       properties of the original edge and face
   * \note The properties of the new halfedges will be undefined
   *
   * \note This is an override to prevent a direct call to PolyConnectivity split_edge_copy,
   *       which would introduce a singular vertex with valence 2 which is not allowed
   *       on TriMeshes
   *
   * @param _eh Edge handle that should be split
   * @param _vh Vertex handle that will be inserted at the edge
   */
  inline void split_edge_copy(EdgeHandle _eh, VertexHandle _vh) { TriConnectivity::split_copy(_eh, _vh); }

  /** \brief Face split (= 1-to-3) split, calls corresponding PolyMeshT function).
   *
   * @param _fh Face handle that should be split
   * @param _vh Vertex handle that will be inserted at the face
   */
  inline void split(FaceHandle _fh, VertexHandle _vh)
  { PolyConnectivity::split(_fh, _vh); }

  /** \brief Face split (= 1-to-3) split, calls corresponding PolyMeshT function).
   *
   * @param _fh Face handle that should be split
   * @param _vh Vertex handle that will be inserted at the face
   */
  inline void split_copy(FaceHandle _fh, VertexHandle _vh)
  { PolyConnectivity::split_copy(_fh, _vh); }

  //@}

private:
  /// Helper for vertex split
  HalfedgeHandle insert_loop(HalfedgeHandle _hh);
  /// Helper for vertex split
  HalfedgeHandle insert_edge(VertexHandle _vh,
                             HalfedgeHandle _h0, HalfedgeHandle _h1);
};

}

#endif//OPENMESH_TRICONNECTIVITY_HH
