#pragma once

#include <OpenMesh/Core/Mesh/PolyMesh_ArrayKernelT.hh>
#include <OpenMesh/Core/Mesh/Traits.hh>

namespace mufflon::scene {

struct PolyTraits : public OpenMesh::DefaultTraits {
	// TODO: possibly change coordinate types

	// Per default, every vertex has a normal and that's it
	VertexAttributes(OpenMesh::Attributes::Normal);
};

using PolyMesh = OpenMesh::PolyMesh_ArrayKernelT<PolyTraits>;

} // namespace mufflon::scene