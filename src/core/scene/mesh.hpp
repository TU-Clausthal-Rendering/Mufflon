#pragma once

#include <OpenMesh/Core/Mesh/PolyMesh_ArrayKernelT.hh>
#include <OpenMesh/Core/Mesh/Traits.hh>
#include <OpenMesh/Tools/Subdivider/Adaptive/Composite/CompositeTraits.hh>

namespace mufflon::scene {

struct PolyTraits : public OpenMesh::DefaultTraits {
	// TODO: possibly change coordinate types

	// Per default, every vertex has a normal and that's it
	VertexAttributes(OpenMesh::Attributes::Normal);
};

struct AdaptivePolyTraits : public OpenMesh::Subdivider::Adaptive::CompositeTraits {};

using PolyMesh = OpenMesh::PolyMesh_ArrayKernelT<PolyTraits>;
using AdaptivePolyMesh = OpenMesh::PolyMesh_ArrayKernelT<AdaptivePolyTraits>;

} // namespace mufflon::scene