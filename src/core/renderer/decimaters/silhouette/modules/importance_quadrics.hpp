#pragma once

#include "util/int_types.hpp"
#include "core/scene/geometry/polygon_mesh.hpp"
#include <OpenMesh/Tools/Decimater/ModBaseT.hh>
#include <OpenMesh/Core/Geometry/QuadricT.hh>

namespace mufflon::renderer::decimaters::silhouette::modules {

// General importance decimater based on threshold
template < class MeshT = scene::geometry::PolygonMeshType >
class ImportanceDecimationModule : public OpenMesh::Decimater::ModBaseT<MeshT> {
public:
	DECIMATING_MODULE(ImportanceDecimationModule, MeshT, ImportanceDecimationModule);

	ImportanceDecimationModule(MeshT& mesh) :
		Base(mesh, false) {
		Base::mesh().add_property(m_quadrics);
	}
	virtual ~ImportanceDecimationModule() {
		Base::mesh().remove_property(m_quadrics);
	}
	ImportanceDecimationModule(const ImportanceDecimationModule&) = delete;
	ImportanceDecimationModule(ImportanceDecimationModule&&) = delete;
	ImportanceDecimationModule& operator=(const ImportanceDecimationModule&) = delete;
	ImportanceDecimationModule& operator=(ImportanceDecimationModule&&) = delete;

	void set_properties(MeshT& originalMesh,
						OpenMesh::VPropHandleT<float> importanceDensity) {
		m_originalMesh = &originalMesh;
		m_importanceDensity = importanceDensity;
	}

	float collapse_priority(const CollapseInfo& ci) final {
		// We rely on the assumption that the original and the target mesh are identical
		// Gather in ring
		float importance = m_originalMesh->property(m_importanceDensity, ci.v0);

		for(auto circIter = Base::mesh().cvv_iter(ci.v0); circIter.is_valid(); ++circIter) {
			float factor = 1.0f;
			// TODO: what's up with that factor?
			if(*circIter == ci.v1)
				factor += 0.001f;

			importance += m_originalMesh->property(m_importanceDensity, *circIter) * factor;
		}

		const auto q = Base::mesh().property(m_quadrics, ci.v0) + Base::mesh().property(m_quadrics, ci.v1);
		const auto err = q(ci.p1);

		return static_cast<float>(importance * err);
	}

	void postprocess_collapse(const CollapseInfo& ci) final {
		// TODO: proper redistribution!
		m_originalMesh->property(m_importanceDensity, ci.v1) += m_originalMesh->property(m_importanceDensity, ci.v0);

		Base::mesh().property(m_quadrics, ci.v1) += Base::mesh().property(m_quadrics, ci.v0);
	}

	void initialize() {
		using OpenMesh::Geometry::Quadricd;
		// alloc quadrics
		if(!m_quadrics.is_valid())
			Base::mesh().add_property(m_quadrics);

		// clear quadrics
		for(auto vertex : Base::mesh().vertices())
			Base::mesh().property(m_quadrics, vertex).clear();

		// calc (normal weighted) quadric
		for(auto face : Base::mesh().faces()) {
			auto fvIter = Base::mesh().fv_iter(face);
			const auto vh0 = *fvIter;  ++fvIter;
			const auto vh1 = *fvIter;  ++fvIter;
			const auto vh2 = *fvIter;

			const auto v0 = OpenMesh::vector_cast<OpenMesh::Vec3d>(Base::mesh().point(vh0));
			const auto v1 = OpenMesh::vector_cast<OpenMesh::Vec3d>(Base::mesh().point(vh1));
			const auto v2 = OpenMesh::vector_cast<OpenMesh::Vec3d>(Base::mesh().point(vh2));

			auto n = (v1 - v0) % (v2 - v0);
			double area = n.norm();
			if(area > std::numeric_limits<double>::min()) {
				n /= area;
				area *= 0.5;
			}

			const double a = n[0];
			const double b = n[1];
			const double c = n[2];
			const double d = -(OpenMesh::vector_cast<OpenMesh::Vec3d>(Base::mesh().point(vh0)) | n);

			OpenMesh::Geometry::QuadricT<double> q(a, b, c, d);
			q *= area;

			Base::mesh().property(m_quadrics, vh0) += q;
			Base::mesh().property(m_quadrics, vh1) += q;
			Base::mesh().property(m_quadrics, vh2) += q;
		}
	}

private:
	MeshT* m_originalMesh;

	// Original mesh properties
	OpenMesh::VPropHandleT<float> m_importanceDensity;								// Temporary storage to keep track of the remapped importance
	OpenMesh::VPropHandleT<OpenMesh::Geometry::QuadricT<double>> m_quadrics;
};

} // namespace mufflon::renderer::decimaters::silhouette::modules