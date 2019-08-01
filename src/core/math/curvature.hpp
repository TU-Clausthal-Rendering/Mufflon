#pragma once

#include "util/assert.hpp"
#include "core/export/api.h"
#include "core/scene/types.hpp"
#include <ei/vector.hpp>

namespace mufflon { namespace math {

// An info struct which d
struct Curvature {
	scene::Direction dirU;
	scene::Direction dirV;
	ei::Vec3 efg;			// Weingarten matrix (2x2 symmetric)

	CUDA_FUNCTION float get_gauss_curvature() const {
		// Assumption: first fundamental form is the identity
		return efg.x * efg.z - efg.y * efg.y;
	}

	CUDA_FUNCTION float get_mean_curvature() const {
		// Assumption: first fundamental form is the identity
		return (efg.x + efg.z) * 0.5f;
	}

	// After this call, dirU and dirV will be the principal directions.
	CUDA_FUNCTION void compute_principal_directions() {
		// If f=0, the tensor is already an diagonal matrix and thus, dirU
		// and dirV are the eigenvectors in global space (locally they are
		// (1,0) and (0,1)).
		if(efg.y != 0.0f) {
			ei::Mat2x2 Q;
			ei::Vec2 l;
			ei::decomposeQl(ei::Mat2x2(efg.x, efg.y, efg.y, efg.z), Q, l);
			scene::Direction newU = Q[0] * dirU + Q[1] * dirV;
			scene::Direction newV = Q[2] * dirU + Q[3] * dirV;
			efg = ei::Vec3{l.x, 0.0f, l.y};
			dirU = newU;
			dirV = newV;
		}
	}
};

/* 
 * Compute the Gaussian curvature (product of principal curvature values).
 * v0, v1, v2: positions of triangle vertices.
 * normal0, normal1, normal2: vertex normals of the triangle
 */
CUDA_FUNCTION
float compute_gauss_curvature(const scene::Point& v0,
							  const scene::Point& v1,
							  const scene::Point& v2,
							  const scene::Direction& normal0,
							  const scene::Direction& normal1,
							  const scene::Direction& normal2) {
	ei::Vec3 e0 = v1 - v0;
	ei::Vec3 e1 = v2 - v0;
	ei::Vec3 en0 = normal1 - normal0;
	ei::Vec3 en1 = normal2 - normal0;
	/*float triArea2 = len(cross(e0, e1));
	float nrmArea2 = len(cross(en0, en1));
	return nrmArea2 / (triArea2 + 1e-5f);
	float triArea2Sq = lensq(cross(e0, e1));
	float nrmArea2Sq = lensq(cross(en0, en1));
	return sqrt(nrmArea2Sq / (triArea2Sq + 1e-25f));*/
	ei::Vec3 triNormalScaled = cross(e0, e1);
	ei::Vec3 normalNormalScaled = cross(en0, en1);
	return dot(normalNormalScaled, triNormalScaled) / dot(triNormalScaled, triNormalScaled);
}


/* 
 * Compute the Gaussian curvature of a quad (product of principal curvature values).
 * v0, v1, v2, v3: positions of quad vertices in the order:
 *
 *     2┌­────┐1
 *      │    │
 *     3└────┘0
 *
 * normal0, normal1, normal2: vertex normals of the quad
 */
CUDA_FUNCTION
float compute_gauss_curvature(const scene::Point& v0,
							  const scene::Point& v1,
							  const scene::Point& v2,
							  const scene::Point& v3,
							  const scene::Direction& normal0,
							  const scene::Direction& normal1,
							  const scene::Direction& normal2,
							  const scene::Direction& normal3) {
	// Average over both possible triangulations
	return (compute_gauss_curvature(v0, v1, v2, normal0, normal1, normal2)
		+ compute_gauss_curvature(v0, v3, v2, normal0, normal3, normal2)
		+ compute_gauss_curvature(v0, v1, v3, normal0, normal1, normal3)
		+ compute_gauss_curvature(v1, v2, v3, normal1, normal2, normal3)) / 4.0f;
}


/* 
 * Compute the Mean curvature (average of principal curvature values).
 * v0, v1, v2: positions of triangle vertices.
 * normal0, normal1, normal2: vertex normals of the triangle
 */
CUDA_FUNCTION
float compute_mean_curvature(const scene::Point& v0,
							 const scene::Point& v1,
							 const scene::Point& v2,
							 const scene::Direction& normal0,
							 const scene::Direction& normal1,
							 const scene::Direction& normal2) {
	ei::Vec3 e0 = v1 - v0;
	ei::Vec3 e1 = v2 - v0;
	ei::Vec3 e2 = v2 - v1;
	ei::Vec3 en0 = normal1 - normal0;
	ei::Vec3 en1 = normal2 - normal0;
	ei::Vec3 en2 = normal2 - normal1;
	/*return (sqrt(lensq(en0) / (lensq(e0) + 1e-5f))
		  + sqrt(lensq(en1) / (lensq(e1) + 1e-5f))
		  + sqrt(lensq(en2) / (lensq(e2) + 1e-5f))) / 3.0f;*/
	/*return (dot(en0, e0) / dot(e0, e0)
		+ dot(en1, e1) / dot(e1, e1)
		+ dot(en2, e2) / dot(e2, e2)) / 3.0f;*/
	float w0 = 1.0f / lensq(e0);//ei::abs(dot(e1, e2) / len(cross(e1, e2)));
	float w1 = 1.0f / lensq(e1);//ei::abs(dot(e0, e2) / len(cross(e0, e2)));
	float w2 = 1.0f / lensq(e2);//ei::abs(dot(e1, e0) / len(cross(e1, e0)));
	return (dot(en0, e0) / dot(e0, e0) * w0
		  + dot(en1, e1) / dot(e1, e1) * w1
		  + dot(en2, e2) / dot(e2, e2) * w2) / (w0 + w1 + w2);
}

/* 
 * Compute the Gaussian curvature of a quad (product of principal curvature values).
 * v0, v1, v2, v3: positions of quad vertices in the order:
 *
 *     2┌­────┐1
 *      │    │
 *     3└────┘0
 *
 * normal0, normal1, normal2: vertex normals of the quad
 */
CUDA_FUNCTION
float compute_mean_curvature(const scene::Point& v0,
							 const scene::Point& v1,
							 const scene::Point& v2,
							 const scene::Point& v3,
							 const scene::Direction& normal0,
							 const scene::Direction& normal1,
							 const scene::Direction& normal2,
							 const scene::Direction& normal3) {
	// Average over both possible triangulations
	return (compute_mean_curvature(v0, v1, v2, normal0, normal1, normal2)
		+ compute_mean_curvature(v0, v3, v2, normal0, normal3, normal2)
		+ compute_mean_curvature(v0, v1, v3, normal0, normal1, normal3)
		+ compute_mean_curvature(v1, v2, v3, normal1, normal2, normal3)) / 4.0f;
}


/*
 * Compute both principal curvature values and their tangent directions in
 * object space:
 * "Estimating Curvatures and Their Derivatives on Triangle Meshes"
 * Rusinkiewicz Szymon
 * https://gfx.cs.princeton.edu/pubs/_2004_ECA/curvpaper.pdf
 */
CUDA_FUNCTION
Curvature compute_curvature(const scene::Direction& geoNormal,
							const scene::Point& v0,
							const scene::Point& v1,
							const scene::Point& v2,
							const scene::Direction& normal0,
							const scene::Direction& normal1,
							const scene::Direction& normal2) {
	const ei::Vec3 e01 = v1 - v0;
	const ei::Vec3 e02 = v2 - v0;
	const ei::Vec3 e12 = v2 - v1;
	// 1. Get an orthonormal tangent base
	/*float minusSinTheta = -sqrt((1.0f - geoNormal.z) * (1.0f + geoNormal.z)); // cos(π/2 + acos(z)) = -sinθ
	float sinZratio = ei::abs(geoNormal.z / minusSinTheta);
	scene::Direction u { geoNormal.x * sinZratio, geoNormal.y * sinZratio, orthoZ };
	mAssert(ei::approx(len(u), 1.0f));*/
	scene::Direction u = normalize(e01);
	scene::Direction v = cross(geoNormal, u);
	// 2. Determine the 6 constraints from finite differences
	ei::Matrix<float, 6, 3> A;
	ei::Vec<float, 6> b;
	ei::Vec3 en0 = normal1 - normal0;
	A(1,1) = A(0,0) = dot(u, e01);
	A(1,2) = A(0,1) = dot(v, e01);
	A(1,0) = A(0,2) = 0.0f;
	b[0] = dot(u, en0);
	b[1] = dot(v, en0);
	ei::Vec3 en1 = normal2 - normal0;
	A(3,1) = A(2,0) = dot(u, e02);
	A(3,2) = A(2,1) = dot(v, e02);
	A(3,0) = A(2,2) = 0.0f;
	b[2] = dot(u, en1);
	b[3] = dot(v, en1);
	ei::Vec3 en2 = normal2 - normal1;
	A(5,1) = A(4,0) = dot(u, e12);
	A(5,2) = A(4,1) = dot(v, e12);
	A(5,0) = A(4,2) = 0.0f;
	b[4] = dot(u, en2);
	b[5] = dot(v, en2);
	// 3. solve least squares for curvature tensor (Weingarten matrix)
	// | e f |
	// | f g |
	ei::Vec3 bsq = transpose(A) * b;
	ei::Mat3x3 Asq = transpose(A) * A;
	ei::Mat3x3 LU;
	ei::UVec3 p;
	ei::decomposeLUp(Asq, LU, p);
	ei::Vec3 efg = ei::solveLUp(LU, p, bsq);
	return { u, v, efg };
}

/*
 * Compute both principal curvature values and their tangent directions in
 * object space:
 * "Estimating Curvatures and Their Derivatives on Triangle Meshes"
 * Rusinkiewicz Szymon
 * https://gfx.cs.princeton.edu/pubs/_2004_ECA/curvpaper.pdf
 */
CUDA_FUNCTION
Curvature compute_curvature(const scene::Direction& geoNormal,
							const scene::Point& v0,
							const scene::Point& v1,
							const scene::Point& v2,
							const scene::Point& v3,
							const scene::Direction& normal0,
							const scene::Direction& normal1,
							const scene::Direction& normal2,
							const scene::Direction& normal3) {
	const ei::Vec3 e01 = v1 - v0;
	const ei::Vec3 e02 = v2 - v0;
	const ei::Vec3 e12 = v2 - v1;
	const ei::Vec3 e23 = v3 - v2;
	// 1. Get an orthonormal tangent base
	/*float minusSinTheta = -sqrt((1.0f - geoNormal.z) * (1.0f + geoNormal.z)); // cos(π/2 + acos(z)) = -sinθ
	float sinZratio = ei::abs(geoNormal.z / minusSinTheta);
	scene::Direction u { geoNormal.x * sinZratio, geoNormal.y * sinZratio, orthoZ };
	mAssert(ei::approx(len(u), 1.0f));*/
	scene::Direction u = normalize(e01);
	scene::Direction v = cross(geoNormal, u);
	// 2. Determine the 8 constraints from finite differences
	ei::Matrix<float, 8, 3> A;
	ei::Vec<float, 8> b;
	ei::Vec3 en0 = normal1 - normal0;
	A(1,1) = A(0,0) = dot(u, e01);
	A(1,2) = A(0,1) = dot(v, e01);
	A(1,0) = A(0,2) = 0.0f;
	b[0] = dot(u, en0);
	b[1] = dot(v, en0);
	ei::Vec3 en1 = normal2 - normal0;
	A(3,1) = A(2,0) = dot(u, e02);
	A(3,2) = A(2,1) = dot(v, e02);
	A(3,0) = A(2,2) = 0.0f;
	b[2] = dot(u, en1);
	b[3] = dot(v, en1);
	ei::Vec3 en2 = normal2 - normal1;
	A(5,1) = A(4,0) = dot(u, e12);
	A(5,2) = A(4,1) = dot(v, e12);
	A(5,0) = A(4,2) = 0.0f;
	b[4] = dot(u, en2);
	b[5] = dot(v, en2);
	ei::Vec3 en3 = normal3 - normal2;
	A(7,1) = A(6,0) = dot(u, e23);
	A(7,2) = A(6,1) = dot(v, e23);
	A(7,0) = A(6,2) = 0.0f;
	b[6] = dot(u, en3);
	b[7] = dot(v, en3);
	// 3. solve least squares for curvature tensor (Weingarten matrix)
	// | e f |
	// | f g |
	ei::Vec3 bsq = transpose(A) * b;
	ei::Mat3x3 Asq = transpose(A) * A;
	ei::Mat3x3 LU;
	ei::UVec3 p;
	ei::decomposeLUp(Asq, LU, p);
	ei::Vec3 efg = ei::solveLUp(LU, p, bsq);
	return { u, v, efg };
}

}} // namespace mufflon::math