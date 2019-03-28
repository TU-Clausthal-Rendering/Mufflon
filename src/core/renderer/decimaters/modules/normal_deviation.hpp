#pragma once

#include "util/degrad.hpp"
#include "util/int_types.hpp"
#include "util/punning.hpp"
#include "core/scene/geometry/polygon_mesh.hpp"
#include <ei/vector.hpp>
#include <OpenMesh/Tools/Decimater/ModBaseT.hh>

namespace mufflon::renderer::decimaters::modules {

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
		for(auto iter = Base::mesh().cvf_ccwbegin(ci.v1); iter.is_valid(); ++iter) {
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
		for(auto iter = Base::mesh().cvf_ccwbegin(ci.v1); iter.is_valid(); ++iter) {
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

	void postprocess_collapse(const CollapseInfo& ci) final {
		// Adjust the vertex normal
		// TODO: find better way to adjust normal (e.g. compute from collapsed vertex normal)
		auto computeAndSetVertexNormal = [](MeshT& mesh, typename MeshT::VertexHandle v) {
			ei::Vec3 normal{};
			for(auto faceIter = mesh.cvf_ccwbegin(v); faceIter.is_valid(); ++faceIter)
				normal += util::pun<ei::Vec3>(mesh.calc_face_normal(*faceIter));
			mesh.set_normal(v, util::pun<typename MeshT::Normal>(ei::normalize(normal)));
		};

		computeAndSetVertexNormal(Base::mesh(), ci.v1);
		for(auto circIter = Base::mesh().cvv_ccwbegin(ci.v1); circIter.is_valid(); ++circIter)
			computeAndSetVertexNormal(Base::mesh(), *circIter);
	}

	void set_max_deviation(const Degrees deviation) {
		m_minCos = std::cos(static_cast<float>(static_cast<Radians>(deviation)));
	}

private:
	float m_minCos;
};

} // namespace mufflon::renderer::decimaters::modules