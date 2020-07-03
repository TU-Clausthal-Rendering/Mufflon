#pragma once

#include "octree_nodes.hpp"
#include "util/int_types.hpp"
#include "core/export/core_api.h"
#include <ei/vector.hpp>
#include <cuda_runtime.h>

namespace mufflon { namespace renderer { namespace decimaters {

template < class N >
class ReadOnlyOctree {
public:
	using NodeType = N;

	__host__ ReadOnlyOctree(const ei::Vec3& invDiag, const ei::Vec3& minBound,
							const NodeType* nodes, const NodeType& root,
							const u32 depth) noexcept :
		m_diagonalInv{ invDiag },
		m_minBound{ minBound },
		m_nodes{ nodes },
		m_root{ root },
		m_depth{ depth }
	{}
	ReadOnlyOctree(const ReadOnlyOctree&) = default;
	ReadOnlyOctree(ReadOnlyOctree&&) = default;
	ReadOnlyOctree& operator=(const ReadOnlyOctree&) = delete;
	ReadOnlyOctree& operator=(ReadOnlyOctree&&) = delete;
	~ReadOnlyOctree() = default;

	__host__ __device__ float get_samples(const ei::Vec3& pos) const noexcept {
		const ei::Vec3 offPos = pos - m_minBound;
		const ei::Vec3 normPos = offPos * m_diagonalInv;
		// Get the integer position on the finest level.
		// Get/set of samples never happens at the same time, so having no barriers is fine
		const decltype(m_depth) gridRes = 1u << m_depth;
		const ei::UVec3 iPos{ normPos * gridRes };
		auto currentLvlMask = gridRes;
		auto currVal = m_root;
		while(currVal.is_parent()) {
			currentLvlMask >>= 1;
			const auto offset = currVal.get_child_offset();
			// Get the relative index of the child [0,7] plus the child offset for the node index
			const auto idx = ((iPos.x & currentLvlMask) ? 1 : 0)
				+ ((iPos.y & currentLvlMask) ? 2 : 0)
				+ ((iPos.z & currentLvlMask) ? 4 : 0)
				+ offset;
			currVal = m_nodes[idx];
		}
		return currVal.get_sample();
	}

private:
	ei::Vec3 m_diagonalInv;
	ei::Vec3 m_minBound;
	const NodeType* m_nodes;
	const NodeType& m_root;
	u32 m_depth;
};

}}} // namespace mufflon::renderer::decimaters