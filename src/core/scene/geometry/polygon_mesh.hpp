#pragma once

#include <OpenMesh/Core/Mesh/Traits.hh>
#include <OpenMesh/Core/Mesh/PolyMesh_ArrayKernelT.hh>
#include <OpenMesh/Core/Mesh/ArrayKernel.hh>

namespace mufflon::scene::geometry {

// Traits for a polygon mesh - has to have normals and 2D UV coordinates per vertex.
struct PolygonTraits : public OpenMesh::DefaultTraits {
	VertexAttributes(OpenMesh::Attributes::Normal | OpenMesh::Attributes::TexCoord2D);
};

using PolygonMeshType = OpenMesh::PolyMesh_ArrayKernelT<PolygonTraits>;

}