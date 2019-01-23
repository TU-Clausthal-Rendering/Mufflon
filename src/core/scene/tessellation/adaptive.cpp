#include "adaptive.hpp"
#include "util/punning.hpp"
#include "util/log.hpp"

namespace mufflon::scene::tessellation {

namespace {

// Returns the index of a vertex in the list of inner vertices based
// on the edge we're on
constexpr inline u32 get_inner_vertex_index(const u32 edgeIndex, const u32 index, const u32 innerLevel) {
	switch(edgeIndex) {
		case 0u: return index + innerLevel * 0u;
		case 1u: return (innerLevel - 1u) + innerLevel * index;
		case 2u: return (innerLevel - index - 1u) + innerLevel * (innerLevel - 1u);
		case 3u: return 0u + (innerLevel - index - 1u) * innerLevel;
	}
	return 0u;
}

} // namespace

AdaptiveTessellater::AdaptiveTessellater(geometry::PolygonMeshType& mesh) :
	m_mesh(mesh)
{}

bool AdaptiveTessellater::tessellate() {
	// Store the vertex handles of the newly created vertices per edge
	m_edgeVertexHandles.clear();
	m_innerVertices.clear();
	m_mesh.request_face_status();
	m_mesh.request_vertex_status();

	// TODO: reserve proper amount

	// Prepare the necessary attributes per edge
	m_mesh.add_property(m_addedVertexProp);
	if(!m_addedVertexProp.is_valid())
		return false;

	for(auto edge : m_mesh.edges()) {
		mAssert(edge.is_valid());

		const OpenMesh::VertexHandle from = m_mesh.from_vertex_handle(m_mesh.halfedge_handle(edge, 0));
		const OpenMesh::VertexHandle to = m_mesh.to_vertex_handle(m_mesh.halfedge_handle(edge, 0));

		// Ask how many new vertices this edge should get
		u32 newVertices = this->get_edge_tessellation_level(edge);
		m_mesh.property(m_addedVertexProp, edge) = AddedVertices{
			static_cast<u32>(m_edgeVertexHandles.size()),
			newVertices
		};
		// For each new vertex, ask to set the position, normal etc. (whatever they want)
		for(u32 i = 0u; i < newVertices; ++i) {
			m_edgeVertexHandles.push_back(m_mesh.add_vertex(OpenMesh::Vec3f{}));
			this->set_edge_vertex(i, newVertices, edge, m_edgeVertexHandles.back());
		}
	}

	// We need a temporary face to copy over properties in case we have a tessellation level of zero
	// at some edge
	const OpenMesh::FaceHandle tempFace = m_mesh.add_face(m_mesh.add_vertex(OpenMesh::Vec3f{}),
														  m_mesh.add_vertex(OpenMesh::Vec3f{}),
														  m_mesh.add_vertex(OpenMesh::Vec3f{}));

	// The inner primitive will be tessellated differently based on primitive type.
	// The inner and outer vertices will then be connected. It follows the following scheme:
	// If the primitive is a triangle, always spawn triangles
	// If the primitive is a quad and either level is less than 2 or evenness is not equal, spawn triangles
	// If the primitive is a quad and both levels have equal evenness (and are both >= 2), spawn min(inner, outer) quads, then triangles

	// Now iterate over the faces to create the inner tesselation
	for(auto face : m_mesh.faces()) {
		// Skip the temporary face
		if(face == tempFace)
			continue;

		// Check if we're in a triangle or quad
		const std::size_t vertexCount = std::distance(m_mesh.cfv_ccwbegin(face), m_mesh.cfv_ccwend(face));
		// Find out the inner tessellation level
		const u32 innerLevel = this->get_inner_tessellation_level(face);

		// Case distinction: do we have inner tessellation at all
		if(innerLevel == 0u) {
			return true;
		} else {
			mAssert(vertexCount <= 4u);
			// We gotta remove the face here already since non-tessellated edges make problems otherwise
			// For that we need to grab some info
			OpenMesh::VertexHandle vertices[4u];
			AddedVertices edgeVertices[4u];
			auto feIter = m_mesh.cfe_ccwbegin(face);
			for(u32 i = 0u; i < vertexCount; ++i, ++feIter) {
				vertices[i] = m_mesh.from_vertex_handle(m_mesh.halfedge_handle(*feIter, 0));
				edgeVertices[i] = m_mesh.property(m_addedVertexProp, *feIter);
			}

			// Copy over the properties and remove the face
			m_mesh.copy_all_properties(face, tempFace, false);
			m_mesh.delete_face(face);

			// Create the faces for the inner vertices
			if(vertexCount == 3u) {
				// Triangles
				// TODO
			} else {
				// Quads

				// Spawn the inner vertices
				m_innerVertices.clear();
				for(u32 y = 0u; y < innerLevel; ++y) {
					for(u32 x = 0u; x < innerLevel; ++x) {
						m_innerVertices.push_back(m_mesh.add_vertex(OpenMesh::Vec3f{}));
						this->set_quad_inner_vertex(x, y, innerLevel, m_innerVertices.back(), tempFace, vertices);
					}
				}

				this->tessellate_inner_quads(innerLevel, tempFace);

				// Create the faces between outer and inner tessellation
				for(u32 edgeIndex = 0u; edgeIndex < vertexCount; ++edgeIndex) {
					const OpenMesh::VertexHandle edgeFrom = vertices[edgeIndex];
					const OpenMesh::VertexHandle edgeTo = vertices[(edgeIndex + 1u < vertexCount) ? edgeIndex + 1u : 0u];

					const AddedVertices& outerVertices = edgeVertices[edgeIndex];

					// Catch edge cases
					if(outerVertices.count == 0u) {
						// Just triangulate strip
						m_stripVertices.clear();
						m_stripVertices.push_back(edgeTo);
						for(u32 i = 0u; i < innerLevel; ++i)
							m_stripVertices.push_back(m_innerVertices[get_inner_vertex_index(edgeIndex, innerLevel - i - 1u, innerLevel)]);
						m_stripVertices.push_back(edgeFrom);
						const OpenMesh::FaceHandle stripPoly = m_mesh.add_face(m_stripVertices);
						if(!stripPoly.is_valid())
							throw std::runtime_error("Failed to add tessellated outer face");
						if(m_stripVertices.size() > 4)
							m_mesh.triangulate(stripPoly);
						// TODO: post-processing!
					} else {
						// TODO: condition is wrong - match in evenness and more than 1 for each
						if(outerVertices.count > 1 && innerLevel > 1 && outerVertices.count % 2 == innerLevel % 2) {
							// Create as many good-looking quads as possible
							const u32 stripeQuadCount = std::min(innerLevel, outerVertices.count) - 1u;
							// Determine what vertex the quads start at
							u32 startInner, startOuter;
							if(innerLevel <= outerVertices.count) {
								startInner = 0u;
								startOuter = (outerVertices.count - innerLevel) / 2u;
							} else {
								startInner = (innerLevel - outerVertices.count) / 2u;
								startOuter = 0u;
							}


							for(u32 i = 0u; i < stripeQuadCount; ++i) {
								// Compute what edge of the inner quad the vertices are on
								const u32 i2 = get_inner_vertex_index(edgeIndex, startInner + i + 1u, innerLevel);
								const u32 i3 = get_inner_vertex_index(edgeIndex, startInner + i, innerLevel);

								const OpenMesh::VertexHandle& v0 = m_edgeVertexHandles[outerVertices.offset + startOuter + i];
								const OpenMesh::VertexHandle& v1 = m_edgeVertexHandles[outerVertices.offset + startOuter + i + 1u];
								const OpenMesh::VertexHandle& v2 = m_innerVertices[i2];
								const OpenMesh::VertexHandle& v3 = m_innerVertices[i3];

								// Add the face and post-process
								const OpenMesh::FaceHandle newFace = m_mesh.add_face(v0, v1, v2, v3);
								if(!newFace.is_valid())
									throw std::runtime_error("Failed to add tessellated inner face");
								this->set_quad_face_outer(tempFace, newFace);
							}

							// The rest just gets triangulated
							// TODO: think about whether corners, where quads are possible, should get them

							// Left corner vertices (all that are not part of the quads)
							m_stripVertices.clear();
							m_stripVertices.push_back(edgeFrom);
							for(u32 i = 0u; i < startOuter + 1u; ++i)
								m_stripVertices.push_back(m_edgeVertexHandles[outerVertices.offset + i]);
							for(u32 i = 0u; i < startInner + 1u; ++i)
								m_stripVertices.push_back(m_innerVertices[get_inner_vertex_index(edgeIndex, startInner - i, innerLevel)]);
							const OpenMesh::FaceHandle leftFace = m_mesh.add_face(m_stripVertices);
							// Right corner vertices (all that are not part of the quads)
							m_stripVertices.clear();
							m_stripVertices.push_back(edgeTo);
							const u32 startIndexInner = startInner + stripeQuadCount;
							for(u32 i = startIndexInner; i < innerLevel; ++i)
								m_stripVertices.push_back(m_innerVertices[get_inner_vertex_index(edgeIndex, innerLevel - i - 1u + startIndexInner, innerLevel)]);
							for(u32 i = startOuter + stripeQuadCount; i < outerVertices.count; ++i)
								m_stripVertices.push_back(m_edgeVertexHandles[outerVertices.offset + i]);
							const OpenMesh::FaceHandle rightFace = m_mesh.add_face(m_stripVertices);

							if(!leftFace.is_valid())
								throw std::runtime_error("Failed to add tessellated outer face");
							if(!rightFace.is_valid())
								throw std::runtime_error("Failed to add tessellated outer face");
							m_mesh.triangulate(leftFace);
							m_mesh.triangulate(rightFace);
							this->set_quad_face_outer(tempFace, leftFace);
							this->set_quad_face_outer(tempFace, rightFace);
						} else {
							// Triangle strip only
							m_stripVertices.clear();
							// First the edge vertices...
							m_stripVertices.push_back(edgeFrom);
							for(u32 i = 0u; i < outerVertices.count; ++i)
								m_stripVertices.push_back(m_edgeVertexHandles[outerVertices.offset + i]);
							m_stripVertices.push_back(edgeTo);
							// ...then in reversed order the inner vertices
							for(u32 i = 0u; i < innerLevel; ++i)
								m_stripVertices.push_back(m_innerVertices[get_inner_vertex_index(edgeIndex, innerLevel - i - 1u, innerLevel)]);
							const OpenMesh::FaceHandle stripPoly = m_mesh.add_face(m_stripVertices);
							if(!stripPoly.is_valid())
								throw std::runtime_error("Failed to add tessellated outer face");
							m_mesh.triangulate(stripPoly);
							// TODO: post-processing!
						}
					}
				}
			}
		}

	}

	// Clean up
	m_mesh.remove_property(m_addedVertexProp);
	m_mesh.delete_face(tempFace, true);
	m_mesh.garbage_collection();
	m_mesh.release_face_status();
	m_mesh.release_vertex_status();
	return true;
}

u32 AdaptiveTessellater::get_edge_tessellation_level(const OpenMesh::EdgeHandle edge) const {
	return 7u;
}

u32 AdaptiveTessellater::get_inner_tessellation_level(const OpenMesh::FaceHandle face) const {
	return 4u;
}

void AdaptiveTessellater::set_edge_vertex(const u32 i, const u32 count,
								const OpenMesh::EdgeHandle edge,
								const OpenMesh::VertexHandle vertex) {
	const OpenMesh::VertexHandle from = m_mesh.from_vertex_handle(m_mesh.halfedge_handle(edge, 0u));
	const OpenMesh::VertexHandle to = m_mesh.to_vertex_handle(m_mesh.halfedge_handle(edge, 0u));
	const float interP = (i + 1u) / static_cast<float>(count + 1u);
	const ei::Vec3& p0 = util::pun<ei::Vec3>(m_mesh.point(from));
	const ei::Vec3& p1 = util::pun<ei::Vec3>(m_mesh.point(to));
	const ei::Vec3& n0 = util::pun<ei::Vec3>(m_mesh.normal(from));
	const ei::Vec3& n1 = util::pun<ei::Vec3>(m_mesh.normal(to));
	const ei::Vec2& u0 = util::pun<ei::Vec2>(m_mesh.texcoord2D(from));
	const ei::Vec2& u1 = util::pun<ei::Vec2>(m_mesh.texcoord2D(to));
	m_mesh.set_point(vertex, util::pun<OpenMesh::Vec3f>(p0 * (1 - interP) + p1 * interP));
	m_mesh.set_normal(vertex, util::pun<OpenMesh::Vec3f>(n0 * (1 - interP) + n1 * interP));
	m_mesh.set_texcoord2D(vertex, util::pun<OpenMesh::Vec2f>(u0 * (1 - interP) + u1 * interP));
}

void AdaptiveTessellater::set_quad_inner_vertex(const u32 x, const u32 y, const u32 level,
									const OpenMesh::VertexHandle vertex,
									const OpenMesh::FaceHandle face,
									const OpenMesh::VertexHandle(&vertices)[4u]) {
	ei::Vec3 pos = ei::bilerp(util::pun<ei::Vec3>(m_mesh.point(vertices[0u])),
								util::pun<ei::Vec3>(m_mesh.point(vertices[1u])),
								util::pun<ei::Vec3>(m_mesh.point(vertices[3u])),
								util::pun<ei::Vec3>(m_mesh.point(vertices[2u])),
								(x + 1u) / static_cast<float>(level + 1u),
								(y + 1u) / static_cast<float>(level + 1u));
	ei::Vec3 normal = ei::bilerp(util::pun<ei::Vec3>(m_mesh.normal(vertices[0u])),
									util::pun<ei::Vec3>(m_mesh.normal(vertices[1u])),
									util::pun<ei::Vec3>(m_mesh.normal(vertices[3u])),
									util::pun<ei::Vec3>(m_mesh.normal(vertices[2u])),
									(x + 1u) / static_cast<float>(level + 1u),
									(y + 1u) / static_cast<float>(level + 1u));
	ei::Vec2 uv = ei::bilerp(util::pun<ei::Vec2>(m_mesh.texcoord2D(vertices[0u])),
								util::pun<ei::Vec2>(m_mesh.texcoord2D(vertices[1u])),
								util::pun<ei::Vec2>(m_mesh.texcoord2D(vertices[3u])),
								util::pun<ei::Vec2>(m_mesh.texcoord2D(vertices[2u])),
								(x + 1u) / static_cast<float>(level + 1u),
								(y + 1u) / static_cast<float>(level + 1u));
	m_mesh.set_point(vertex, util::pun<OpenMesh::Vec3f>(pos));
	m_mesh.set_normal(vertex, util::pun<OpenMesh::Vec3f>(normal));
	m_mesh.set_texcoord2D(vertex, util::pun<OpenMesh::Vec2f>(uv));
}

void AdaptiveTessellater::set_quad_face_inner(const u32 x, const u32 y, const u32 innerLevel,
									const OpenMesh::FaceHandle original,
									const OpenMesh::FaceHandle newInner) {
	m_mesh.copy_all_properties(original, newInner);
}

void AdaptiveTessellater::set_quad_face_outer(const OpenMesh::FaceHandle original,
											  const OpenMesh::FaceHandle newOuter) {
	m_mesh.copy_all_properties(original, newOuter);
}

void AdaptiveTessellater::set_triangle_face_outer(const OpenMesh::FaceHandle original,
												  const OpenMesh::FaceHandle newOuter) {
	m_mesh.copy_all_properties(original, newOuter);
}

// Perfoms tessellation for quad inner level
void AdaptiveTessellater::tessellate_inner_quads(const u32 innerLevel, const OpenMesh::FaceHandle original) {
	for(u32 y = 0u; y < innerLevel - 1u; ++y) {
		for(u32 x = 0u; x < innerLevel - 1u; ++x) {
			const OpenMesh::VertexHandle& v0 = m_innerVertices[y * innerLevel + x];
			const OpenMesh::VertexHandle& v1 = m_innerVertices[y * innerLevel + x + 1u];
			const OpenMesh::VertexHandle& v2 = m_innerVertices[(y + 1u) * innerLevel + x + 1u];
			const OpenMesh::VertexHandle& v3 = m_innerVertices[(y + 1u) * innerLevel + x];

			const ei::Vec3 p0 = util::pun<ei::Vec3>(m_mesh.point(v0));
			const ei::Vec3 p1 = util::pun<ei::Vec3>(m_mesh.point(v1));
			const ei::Vec3 p2 = util::pun<ei::Vec3>(m_mesh.point(v2));
			const ei::Vec3 p3 = util::pun<ei::Vec3>(m_mesh.point(v3));

			const OpenMesh::FaceHandle newFace = m_mesh.add_face(v0, v1, v2, v3);
			this->set_quad_face_inner(x, y, innerLevel, original, newFace);
		}
	}
}

} // namespace mufflon::scene::tessellation