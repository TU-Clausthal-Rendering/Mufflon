#pragma once

#include <OpenMesh/Core/Mesh/PolyConnectivity.hh>
#include <OpenMesh/Core/Mesh/PolyMeshT.hh>
#include "util/assert.hpp"

namespace mufflon::scene {

/// Utility class which represents a range of iterators.
template < class Iter >
class IteratorRange {
public:
	IteratorRange(Iter begin, Iter end) :
		m_begin(std::move(begin)),
		m_end(std::move(end)) {}

	const Iter& begin() { return m_begin; }
	const Iter& end() { return m_end; }

private:
	Iter m_begin;
	Iter m_end;
};

/**
 * Iterator for PolyMeshes.
 * Iterates over all vertex indices of the mesh in clockwise order per face.
 * Follows the interface of a ForwardIterator.
 * Does NOT skip deleted faces in the mesh.
 */
template < class Kernel, std::size_t N = 0u >
class PolygonCWIterator {
public:
	static constexpr std::size_t VERTICES_PER_FACE = N;

	/// Creates the beginning iterator.
	static PolygonCWIterator begin(const OpenMesh::PolyMeshT<Kernel>& mesh) {
		return PolygonCWIterator(mesh, mesh.faces_begin());
	}

	/// Creates the ending iterator
	static PolygonCWIterator end(const OpenMesh::PolyMeshT<Kernel>& mesh) {
		return PolygonCWIterator(mesh, mesh.faces_end());
	}

	/// Post-increment operator
	virtual PolygonCWIterator& operator++() {
		mAssert(m_faceIter != m_mesh.faces_end());
		if(m_vertexIter != m_mesh.cfv_cwend(*m_faceIter)) {
			++m_vertexIter;
		} else {
			++m_faceIter;
			this->skip_until_n_poly();
		}
		return *this;
	}

	/// Pre-increment operator
	virtual PolygonCWIterator operator++(int) {
		PolygonCWIterator temp(*this);
		++(*this);
		return temp;
	}

	/// Checks iterators for equality, ie. point towards the same vertex
	bool operator==(const PolygonCWIterator& iter) const {
		if(m_mesh == iter.m_mesh)
			if(m_faceIter == iter.m_faceIter)
				if(m_faceIter == m_mesh.faces_end())
					return true;
				else
					return m_vertexIter == iter.m_vertexIter;
		return false;
	}

	bool operator!=(const PolygonCWIterator& iter) const {
		return !this->operator==(iter);
	}

	/// Dereference operator - returns a (valid) vertex handle
	const OpenMesh::VertexHandle &operator*() const {
		mAssert(m_faceIter != m_mesh.faces_end()
				&& m_vertexIter != m_mesh.cfv_cwend(*m_faceIter));
		return *m_vertexIter;
	}

	/// Dereference operator - returns a pointer to a (valid) vertex handle
	const OpenMesh::FaceHandle *operator->() const {
		mAssert(m_faceIter != m_mesh.faces_end()
				&& m_vertexIter != m_mesh.cfv_cwend(*m_faceIter));
		return m_vertexIter;
	}

private:
	PolygonCWIterator(const OpenMesh::PolyMeshT<Kernel>& mesh,
					  const OpenMesh::PolyConnectivity::ConstFaceIter& face_iter) :
		m_mesh(mesh),
		m_faceIter(face_iter),
		m_vertexIter() // Don't initialize yet since the mesh might be empty
	{
		this->skip_until_n_poly();
	}

	/**
	 * Skips the iterator until the first vertex of the next n-polygon.
	 * Alternatively stops at the end of the mesh.
	 * Assumes that the face iterator already has been increased to avoid
	 * special casing in the constructor.
	 */
	virtual void skip_until_n_poly() {
		while(m_faceIter != m_mesh.faces_end()) {
			// Get first vertex of current polygon
			m_vertexIter = m_mesh.cfv_cwbegin(*m_faceIter);
			if constexpr(VERTICES_PER_FACE != 0u) {
				// If we want only specific polygons, check the iterator distance to
				// know how many vertices this face has
				if(std::distance(m_vertexIter, m_mesh.cfv_cwend(*m_faceIter)) == VERTICES_PER_FACE)
					break;
				++m_faceIter;
			} else {
				// Any polygon allowed -> just get out
				break;
			}
		}
	}

	const OpenMesh::PolyMeshT<Kernel>& m_mesh;
	OpenMesh::PolyConnectivity::ConstFaceIter m_faceIter;
	OpenMesh::PolyConnectivity::ConstFaceVertexCWIter m_vertexIter;
};

} // namespace mufflon::scene