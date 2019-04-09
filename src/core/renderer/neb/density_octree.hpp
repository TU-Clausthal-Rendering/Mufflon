#pragma once

#include <ei/3dtypes.hpp>
#include <memory>
#include <atomic>

namespace mufflon::renderer {

	// TODO: move codesnipped somewhere else (epsilon?)
	// Compute the area of the plane-box intersection
	// https://math.stackexchange.com/questions/885546/area-of-the-polygon-formed-by-cutting-a-cube-with-a-plane
	// https://math.stackexchange.com/a/885662
	inline float intersection_area(const ei::Vec3& bmin, const ei::Vec3& bmax, const ei::Vec3& pos, const ei::Vec3& normal) {
		ei::Vec3 cellSize = bmax - bmin;
		ei::Vec3 absN = abs(normal);
		// 1D cases
		if(ei::abs(absN.x - 1.0f) < 1e-3f) return cellSize.y * cellSize.z;
		if(ei::abs(absN.y - 1.0f) < 1e-3f) return cellSize.x * cellSize.z;
		if(ei::abs(absN.z - 1.0f) < 1e-3f) return cellSize.x * cellSize.y;
		// 2D cases
		for(int d = 0; d < 3; ++d) if(absN[d] < 1e-4f) {
			int x = (d + 1) % 3;
			int y = (d + 2) % 3;
			// Use the formula from stackexchange: phi(t) = max(0,t)^2 / 2 m_1 m_2
			// -> l(t) = sum^4 s max(0,t-dot(m,v)) / m_1 m_2
			// -> A(t) = l(t) * h_3
			float t = normal[x] * pos[x] + normal[y] * pos[y];
			float sum = 0.0f;
			sum += ei::max(0.0f, t - (normal[x] * bmin[x] + normal[y] * bmin[y]));
			sum -= ei::max(0.0f, t - (normal[x] * bmin[x] + normal[y] * bmax[y]));
			sum -= ei::max(0.0f, t - (normal[x] * bmax[x] + normal[y] * bmin[y]));
			sum += ei::max(0.0f, t - (normal[x] * bmax[x] + normal[y] * bmax[y]));
			return cellSize[d] * ei::abs(sum / (normal[x] * normal[y]));
		}
		// 3D cases
		float t = dot(normal, pos);
		float sum = 0.0f;
		sum += ei::sq(ei::max(0.0f, t - dot(normal, bmin)));
		sum -= ei::sq(ei::max(0.0f, t - dot(normal, ei::Vec3{bmin.x, bmin.y, bmax.z})));
		sum += ei::sq(ei::max(0.0f, t - dot(normal, ei::Vec3{bmin.x, bmax.y, bmax.z})));
		sum -= ei::sq(ei::max(0.0f, t - dot(normal, ei::Vec3{bmin.x, bmax.y, bmin.z})));
		sum += ei::sq(ei::max(0.0f, t - dot(normal, ei::Vec3{bmax.x, bmax.y, bmin.z})));
		sum -= ei::sq(ei::max(0.0f, t - dot(normal, ei::Vec3{bmax.x, bmin.y, bmin.z})));
		sum += ei::sq(ei::max(0.0f, t - dot(normal, ei::Vec3{bmax.x, bmin.y, bmax.z})));
		sum -= ei::sq(ei::max(0.0f, t - dot(normal, bmax)));
		return ei::abs(sum / (2.0f * normal.x * normal.y * normal.z));
	}

	// A sparse octree with atomic insertion to measure the density of elements in space.
	class DensityOctree {
		static constexpr int LVL0_N = 4;
		static constexpr float SPLIT_FACTOR = 0.5f;
	public:
		void set_iteration(int iter) {
			m_densityScale = 1.0f / iter;
			m_splitCountDensity = ei::ceil(SPLIT_FACTOR * 8) * iter;
		}

		void initialize(const ei::Box& sceneBounds, int capacity) {
			// Slightly enlarge the volume to avoid numerical issues on the boundary
			ei::Vec3 sceneSize = (sceneBounds.max - sceneBounds.min) * 1.002f;
			m_sceneSizeInv = 1.0f / sceneSize;
			m_sceneScale = len(sceneSize);
			m_minBound = sceneBounds.min - sceneSize * (0.001f / 1.002f);
			m_capacity = LVL0_N * LVL0_N * LVL0_N + ((capacity + 7) & (~7));
			m_nodes = std::make_unique<std::atomic_int32_t[]>(m_capacity);;
			// Root nodes have a count of 0
			for(int i = 0; i < LVL0_N * LVL0_N * LVL0_N; ++i)
				m_nodes[i].store(0);
			// TODO: parallelize?
			// The other nodes are only used if the parent is split
			for(int i = LVL0_N * LVL0_N * LVL0_N; i < m_capacity; ++i)
				m_nodes[i].store(ei::ceil(SPLIT_FACTOR));
			m_allocationCounter.store(LVL0_N * LVL0_N * LVL0_N);
		}

		void increment(const ei::Vec3& pos) {
			ei::Vec3 normPos = (pos - m_minBound) * m_sceneSizeInv;
			// The first level is a uniform grid of size LVL0_N�
			ei::IVec3 rootPos { normPos * LVL0_N };
			int idx = rootPos.x + LVL0_N * (rootPos.y + LVL0_N * rootPos.z);
			int countOrChild = increment_if_positive(idx);
			countOrChild = split_node_if_necessary(idx, countOrChild);
			int edgeL = LVL0_N;
			while(countOrChild < 0) {
				edgeL *= 2;
				// Get the relative index of the child [0,7]
				ei::IVec3 intPos = (ei::IVec3{ normPos * edgeL }) & 1;
				idx = intPos.x + 2 * (intPos.y + 2 * intPos.z);
				idx -= countOrChild;	// 'Add' global offset (which is stored negative)
				countOrChild = increment_if_positive(idx);
				countOrChild = split_node_if_necessary(idx, countOrChild);
			}
		}

		float get_density(const ei::Vec3& pos, const ei::Vec3& normal) {
			ei::Vec3 offPos = pos - m_minBound;
			ei::Vec3 normPos = offPos * m_sceneSizeInv;
			// The first level is a uniform grid of size LVL0_N�
			ei::IVec3 rootPos { normPos * LVL0_N };
			int idx = rootPos.x + LVL0_N * (rootPos.y + LVL0_N * rootPos.z);
			int countOrChild = m_nodes[idx].load();
			int edgeL = LVL0_N;
			while(countOrChild < 0) {
				edgeL *= 2;
				// Get the relative index of the child [0,7]
				ei::IVec3 intPos = (ei::IVec3{ normPos * edgeL }) & 1;
				idx = intPos.x + 2 * (intPos.y + 2 * intPos.z);
				idx -= countOrChild;	// 'Add' global offset (which is stored negative)
				countOrChild = m_nodes[idx].load();
			}
			if(countOrChild > 0) {
				//float countPerIteration = m_densityScale * countOrChild;
				// Get the world space cell boundaries
				ei::IVec3 intPos { normPos * edgeL };
				ei::Vec3 cellMin = intPos / (edgeL * m_sceneSizeInv);
				ei::Vec3 cellMax = (intPos+1) / (edgeL * m_sceneSizeInv);
				float area = intersection_area(cellMin, cellMax, offPos, normal);
				// Sometimes the above method returns zero. Therefore we restrict the
				// area to something larger then a thousands part of an approximate cell area.
				float minArea = 1e-3f * ei::sq(m_sceneScale / edgeL);
				return m_densityScale * countOrChild / ei::max(minArea, area);
			}
			return 0.0f;
		}

		int capacity() const { return m_capacity; }
		int size() const { return ei::min(m_capacity, m_allocationCounter.load()); }
		// Get the size of the associated memory excluding this instance.
		std::size_t mem_size() const { return sizeof(std::atomic_int32_t) * m_capacity; }
	private:
		float m_densityScale;		// 1/#iterations to normalize the counters into a density
		int m_splitCountDensity;	// The number when a node is split must be a multiple of 8 and must grow proportional to #iterations
		ei::Vec3 m_minBound;
		ei::Vec3 m_sceneSizeInv;
		// Nodes consist of 8 atomic counters OR child indices. Each number is either a
		// counter (positive) or a negated child index.
		std::unique_ptr<std::atomic_int32_t[]> m_nodes;
		std::atomic_int32_t m_allocationCounter;
		int m_capacity;
		float m_sceneScale;

		// Returns the new value
		int increment_if_positive(int idx) {
			int oldV = m_nodes[idx].load();
			int newV;
			do {
				if(oldV < 0) return oldV;	// Do nothing, the value is a child pointer
				newV = oldV + 1;			// Increment
			} while(!m_nodes[idx].compare_exchange_weak(oldV, newV));	// Write if nobody destroyed the value
			return newV;
		}

		// Returns the new child pointer or 0
		int split_node_if_necessary(int idx, int count) {
			// The node must be split if its density gets too high
			if(count >= m_splitCountDensity) {
				// Only one thread is responsible to do the allocation
				if(count == m_splitCountDensity) {
					int child = m_allocationCounter.fetch_add(8);
					if(child >= m_capacity) { // Allocation overflow
						m_allocationCounter.store(int(m_capacity + 1));	// Avoid overflow of the counter (but keep a large number)
						return 0;
					}
					m_nodes[idx].store(-child);
					// All eight children are initilized with SPLIT_FACTOR on clear().
					return -child;
				} else {
					// Spin-lock until the responsible thread has set the child pointer
					int child = m_nodes[idx].load();
					while(child > 0) {
						// Check for allocation overflow
						if(m_allocationCounter.load() > m_capacity)
							return 0;
						child = m_nodes[idx].load();
					}
					return child;
				}
			}
			return count;
		}
	};

} // namespace mufflon::renderer