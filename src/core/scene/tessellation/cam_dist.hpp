	#pragma once

#include "tessellater.hpp"
#include "util/punning.hpp"
#include <vector>

namespace mufflon::scene::tessellation {

// Tessellates purely based on distance to camera
class CameraDistanceTessellater : public Tessellater {
public:
	CameraDistanceTessellater(const u32 maxLevel, const ei::Vec3& cameraPos,
							  const std::vector<ei::Mat3x4>& instTrans) :
		m_maxLevel(static_cast<float>(maxLevel)),
		m_camPos(cameraPos),
		m_instanceTransformations(instTrans) {}
	CameraDistanceTessellater(const CameraDistanceTessellater&) = delete;
	CameraDistanceTessellater(CameraDistanceTessellater&&) = delete;
	CameraDistanceTessellater& operator=(const CameraDistanceTessellater&) = delete;
	CameraDistanceTessellater& operator=(CameraDistanceTessellater&&) = delete;
	virtual ~CameraDistanceTessellater() = default;

protected:
	virtual u32 get_edge_tessellation_level(const OpenMesh::EdgeHandle edge) const override {
		const auto halfedge = m_mesh->halfedge_handle(edge, 0);
		const auto p0 = util::pun<ei::Vec3>(m_mesh->point(m_mesh->to_vertex_handle(halfedge)));
		const auto p1 = util::pun<ei::Vec3>(m_mesh->point(m_mesh->from_vertex_handle(halfedge)));

		float maxFactor = 0.f;
		for(const auto& instTrans : m_instanceTransformations) {
			const auto instP0 = ei::transform(p0, instTrans);
			const auto instP1 = ei::transform(p1, instTrans);
			const float distSq = std::min(ei::lensq(instP0 - m_camPos), ei::lensq(instP1 - m_camPos));
			const float edgeLen = ei::len(instP0 - instP1);
			// TODO
			const float factor = edgeLen / distSq;

			maxFactor = std::max(maxFactor, factor);
		}

		return static_cast<u32>(m_maxLevel * std::min(1.f, maxFactor));
	}

	virtual u32 get_inner_tessellation_level(const OpenMesh::FaceHandle face) const override {
		static thread_local std::vector<ei::Vec3> transformedPoints;

		float maxFactor = 0.f;
		for(const auto& instTrans : m_instanceTransformations) {
			ei::Vec3 centre{ 0.f };
			transformedPoints.clear();
			for(auto iter = m_mesh->cfv_ccwbegin(face); iter.is_valid(); ++iter) {
				transformedPoints.push_back(ei::transform(util::pun<ei::Vec3>(m_mesh->point(*iter)), instTrans));
				centre += transformedPoints.back();
			}
			const float distSq = ei::lensq((centre / static_cast<float>(transformedPoints.size())) - m_camPos);

			// Compute area as another factor
			float area;
			if(transformedPoints.size() == 3u) {
				const auto& p0 = transformedPoints[0u];
				const auto& p1 = transformedPoints[1u];
				const auto& p2 = transformedPoints[2u];
				// Triangle
				area = 0.5f * ei::len(ei::cross(p1 - p0, p2 - p0));
			} else {
				// Quad; we assume that the quad is planar, if not we overestimate the area
				const auto& p0 = transformedPoints[0u];
				const auto& p1 = transformedPoints[1u];
				const auto& p2 = transformedPoints[2u];
				const auto& p3 = transformedPoints[3u];
				area = 0.5f * (ei::len(ei::cross(p1 - p0, p2 - p0)) + ei::len(ei::cross(p2 - p0, p3 - p0)));
			}

			// TODO
			const float factor = std::sqrt(area) / distSq;

			maxFactor = std::max(maxFactor, factor);
		}
		

		// TODO: implement tessellation for no inner level
		return static_cast<u32>(m_maxLevel * std::min(1.f, maxFactor));
	}

private:
	float m_maxLevel;
	ei::Vec3 m_camPos;
	std::vector<ei::Mat3x4> m_instanceTransformations;
};

} // namespace mufflon::scene::tessellation 