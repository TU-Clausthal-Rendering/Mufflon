#pragma once

#include "util/degrad.hpp"
#include "util/int_types.hpp"
#include "core/scene/geometry/polygon_mesh.hpp"
#include <OpenMesh/Tools/Decimater/ModBaseT.hh>

namespace mufflon::renderer::silhouette::decimation::modules {

// Makes sure there are no normal flips
template < class MeshT = scene::geometry::PolygonMeshType >
class NormalDeviationModule : public OpenMesh::Decimater::ModBaseT<MeshT> {
public:
	DECIMATING_MODULE(NormalDeviationModule, MeshT, NormalDeviationModule);

	NormalDeviationModule(MeshT& mesh) :
		Base(mesh, true) {}
	virtual ~NormalDeviationModule() = default;
	NormalDeviationModule(const NormalDeviationModule&) = delete;
	NormalDeviationModule(NormalDeviationModule&&) = delete;
	NormalDeviationModule& operator=(const NormalDeviationModule&) = delete;
	NormalDeviationModule& operator=(NormalDeviationModule&&) = delete;

	float collapse_priority(const CollapseInfo& ci) final {
		static thread_local std::vector<typename MeshT::Normal> normalStorage;

		// Compute the face normals before the collapse
		normalStorage.clear();
		for(typename MeshT::ConstVertexFaceIter iter(Base::mesh(), ci.v0); iter.is_valid(); ++iter) {
			typename MeshT::FaceHandle fh = *iter;
			if(fh != ci.fl && fh != ci.fr) {
				normalStorage.push_back(Base::mesh().calc_face_normal(fh));
			}
		}

		// simulate collapse
		Base::mesh().set_point(ci.v0, ci.p1);

		// check for flipping normals
		typename MeshT::Scalar c(1.0);
		u32 index = 0u;
		for(typename MeshT::ConstVertexFaceIter iter(Base::mesh(), ci.v0); iter.is_valid(); ++iter) {
			typename MeshT::FaceHandle fh = *iter;
			if(fh != ci.fl && fh != ci.fr) {
				typename const MeshT::Normal& n1 = normalStorage[index];
				typename MeshT::Normal n2 = Base::mesh().calc_face_normal(fh);

				c = dot(n1, n2);

				if(c < m_minCos)
					break;

				++index;
			}
		}

		// undo simulation changes
		Base::mesh().set_point(ci.v0, ci.p0);

		return static_cast<float>((c < m_minCos) ? Base::ILLEGAL_COLLAPSE : Base::LEGAL_COLLAPSE);
	}

	void set_max_deviation(const Degrees deviation) {
		m_minCos = std::cos(static_cast<float>(static_cast<Radians>(deviation)));
	}

private:
	float m_minCos;
};

} // namespace mufflon::renderer::silhouette::decimation::modules