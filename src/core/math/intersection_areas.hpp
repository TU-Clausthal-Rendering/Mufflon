#pragma once

#include <ei/3dtypes.hpp>

namespace mufflon { namespace math {

// Compute the area of the plane-box intersection
// https://math.stackexchange.com/questions/885546/area-of-the-polygon-formed-by-cutting-a-cube-with-a-plane
// https://math.stackexchange.com/a/885662
inline float intersection_area(const ei::Vec3& bmin, const ei::Vec3& bmax, const ei::Vec3& pos, const ei::Vec3& normal) {
	ei::Vec3 cellSize = bmax - bmin;
	ei::Vec3 absN = abs(normal);
	// 1D cases
	if(ei::abs(absN.x - 1.0f) < 1e-3f) return (pos.x >= bmin.x && pos.x <= bmax.x) ? cellSize.y * cellSize.z : 0.0f;
	if(ei::abs(absN.y - 1.0f) < 1e-3f) return (pos.y >= bmin.y && pos.y <= bmax.y) ? cellSize.x * cellSize.z : 0.0f;
	if(ei::abs(absN.z - 1.0f) < 1e-3f) return (pos.z >= bmin.z && pos.z <= bmax.z) ? cellSize.x * cellSize.y : 0.0f;
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

// Normalized variant of the above method. Both the box and the position are moved
// by bMin. I.e. the origin is the min-coordinate of the box.
// The size of the box is not normalized.
inline float intersection_area_nrm(const ei::Vec3& cellSize, const ei::Vec3& pos, const ei::Vec3& normal) {
	ei::Vec3 absN = abs(normal);
	// 1D cases
	if(ei::abs(absN.x - 1.0f) < 1e-3f) return (pos.x >= 0.0f && pos.x <= cellSize.x) ? cellSize.y * cellSize.z : 0.0f;
	if(ei::abs(absN.y - 1.0f) < 1e-3f) return (pos.y >= 0.0f && pos.y <= cellSize.y) ? cellSize.x * cellSize.z : 0.0f;
	if(ei::abs(absN.z - 1.0f) < 1e-3f) return (pos.z >= 0.0f && pos.z <= cellSize.z) ? cellSize.x * cellSize.y : 0.0f;
	// 2D cases
	for(int d = 0; d < 3; ++d) if(absN[d] < 1e-4f) {
		int x = (d + 1) % 3;
		int y = (d + 2) % 3;
		// Use the formula from stackexchange: phi(t) = max(0,t)^2 / 2 m_1 m_2
		// -> l(t) = sum^4 s max(0,t-dot(m,v)) / m_1 m_2
		// -> A(t) = l(t) * h_3
		float t = normal[x] * pos[x] + normal[y] * pos[y];
		float sum = 0.0f;
		sum += ei::max(0.0f, t);
		sum -= ei::max(0.0f, t - normal[y] * cellSize[y]);
		sum -= ei::max(0.0f, t - normal[x] * cellSize[x]);
		sum += ei::max(0.0f, t - (normal[x] * cellSize[x] + normal[y] * cellSize[y]));
		return cellSize[d] * ei::abs(sum / (normal[x] * normal[y]));
	}
	// 3D cases
	float t = dot(normal, pos);
	float sum = 0.0f;
	const ei::Vec3 ns = normal * cellSize;
	sum += ei::sq(ei::max(0.0f, t));
	sum -= ei::sq(ei::max(0.0f, t - ns.z));
	sum += ei::sq(ei::max(0.0f, t - ns.y - ns.z));
	sum -= ei::sq(ei::max(0.0f, t - ns.y));
	sum += ei::sq(ei::max(0.0f, t - ns.x - ns.y));
	sum -= ei::sq(ei::max(0.0f, t - ns.x));
	sum += ei::sq(ei::max(0.0f, t - ns.x - ns.z));
	sum -= ei::sq(ei::max(0.0f, t - ns.x - ns.y - ns.z));
	return ei::abs(sum / (2.0f * normal.x * normal.y * normal.z));
}

}} // namespace mufflon::math