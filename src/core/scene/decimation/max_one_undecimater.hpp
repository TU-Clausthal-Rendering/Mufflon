#pragma once

#include "core/scene/geometry/polygon_mesh.hpp"
#include <OpenMesh/Tools/Decimater/BaseDecimaterT.hh>

namespace mufflon::scene::decimation {

// TODO: introduce some sort of module system?

/* Unlike its counterpart, this does not take a full-fletched module system (yet),
 * but instead fixely determines the local per-vertex density of the given property
 * and restores collapsed vertices if it surpasses the given threshold.
 * Only vertices that are of collapse-depth one, ie. who have not been collapsed further,
 * are considered.
 */
class MaxOneUndecimater {
public:
	using Mesh = geometry::PolygonMeshType;

	struct CollapseHistory {
		Mesh::VertexHandle v1;
		Mesh::VertexHandle vl;
		Mesh::VertexHandle vr;
	};

	MaxOneUndecimater();
	~MaxOneUndecimater();

	std::size_t undecimate(Mesh& mesh, const float threshold);

private:
	OpenMesh::VPropHandleT<float> m_importanceDensity;
	OpenMesh::VPropHandleT<CollapseHistory> m_collapseHistory;
};

} // namespace mufflon::scene::decimation