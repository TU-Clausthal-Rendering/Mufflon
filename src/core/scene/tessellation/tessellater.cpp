#include "tessellater.hpp"
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

void Tessellater::tessellate(geometry::PolygonMeshType& mesh) {
	m_mesh = &mesh;

	// Store the vertex handles of the newly created vertices per edge
	m_edgeVertexHandles.clear();
	m_innerVertices.clear();
	// We need to be able to delete faces and vertices
	m_mesh->request_face_status();
	m_mesh->request_vertex_status();

	// TODO: reserve proper amount

	// Prepare the necessary attributes per edge
	m_mesh->add_property(m_addedVertexProp);
	if (!m_addedVertexProp.is_valid())
		throw std::runtime_error("Failed to add edge vertex property to mesh");


	for(auto edge : m_mesh->edges()) {
		mAssert(edge.is_valid());

		const OpenMesh::VertexHandle from = m_mesh->from_vertex_handle(m_mesh->halfedge_handle(edge, 0));
		const OpenMesh::VertexHandle to = m_mesh->to_vertex_handle(m_mesh->halfedge_handle(edge, 0));

		// Ask how many new vertices this edge should get
		u32 newVertices = this->get_edge_tessellation_level(edge);
		m_mesh->property(m_addedVertexProp, edge) = AddedVertices{
			from, to,
			static_cast<u32>(m_edgeVertexHandles.size()),
			newVertices
		};
		// For each new vertex, ask to set the position, normal etc. (whatever they want)
		for(u32 i = 0u; i < newVertices; ++i) {
			const OpenMesh::VertexHandle newVertex = m_mesh->add_vertex(OpenMesh::Vec3f{});
			if(!newVertex.is_valid())
				throw std::runtime_error("Failed to add edge vertex");
			m_edgeVertexHandles.push_back(newVertex);
			this->set_edge_vertex(static_cast<float>(i + 1u) / static_cast<float>(newVertices + 1u),
								  edge, m_edgeVertexHandles.back());
		}
	}

	// We need a temporary face to copy over properties in case we have a tessellation level of zero
	// at some edge
	const OpenMesh::VertexHandle tempV0 = m_mesh->add_vertex(OpenMesh::Vec3f{});
	const OpenMesh::VertexHandle tempV1 = m_mesh->add_vertex(OpenMesh::Vec3f{});
	const OpenMesh::VertexHandle tempV2 = m_mesh->add_vertex(OpenMesh::Vec3f{});
	if(!tempV0.is_valid() || !tempV1.is_valid() || !tempV2.is_valid())
		throw std::runtime_error("Failed to add temporary vertices");
	const OpenMesh::FaceHandle tempFace = m_mesh->add_face(tempV0, tempV1, tempV2);
	if(!tempFace.is_valid())
		throw std::runtime_error("Failed to add temporary face");

	// The inner primitive will be tessellated differently based on primitive type.
	// The inner and outer vertices will then be connected. It follows the following scheme:
	// If the primitive is a triangle, always spawn triangles
	// If the primitive is a quad and either level is less than 2 or evenness is not equal, spawn triangles
	// If the primitive is a quad and both levels have equal evenness (and are both >= 2), spawn min(inner, outer) quads, then triangles

	// Now iterate over the faces to create the inner tesselation
	for(auto face : m_mesh->faces()) {
		// Skip the temporary face
		if(face == tempFace)
			continue;

		// Check if we're in a triangle or quad
		const std::size_t vertexCount = std::distance(m_mesh->cfv_ccwbegin(face), m_mesh->cfv_ccwend(face));
		// Find out the inner tessellation level
		const u32 innerLevel = this->get_inner_tessellation_level(face);

		// Case distinction: do we have inner tessellation at all
		if(innerLevel == 0u) {
			m_stripVertices.clear();
			// Fetch the additional vertices from the edges
			auto fhIter = m_mesh->cfh_ccwbegin(face);
			for(u32 i = 0u; i < vertexCount; ++i, ++fhIter) {
				OpenMesh::VertexHandle vertex = m_mesh->from_vertex_handle(*fhIter);
				mAssert(vertex.is_valid());
				const OpenMesh::EdgeHandle eh = m_mesh->edge_handle(*fhIter);
				mAssert(eh.is_valid());
				const AddedVertices& outerVertices = m_mesh->property(m_addedVertexProp, eh);
				m_stripVertices.push_back(vertex);
				if(vertex == outerVertices.from) {
					for(u32 v = 0u; v < outerVertices.count; ++v)
						m_stripVertices.push_back(m_edgeVertexHandles[outerVertices.offset + v]);
				} else {
					const u32 maxOffset = outerVertices.offset + outerVertices.count;
					for(u32 v = 0u; v < outerVertices.count; ++v)
						m_stripVertices.push_back(m_edgeVertexHandles[maxOffset - v - 1u]);
				}
			}

			if(m_stripVertices.size() != vertexCount) {
				// TODO: define a better tessellation pattern?
				// Copy over the properties and remove the face
				m_mesh->copy_all_properties(face, tempFace, false);
				m_mesh->delete_face(face);

				const OpenMesh::FaceHandle newFace = m_mesh->add_face(m_stripVertices);
				if(!newFace.is_valid())
					throw std::runtime_error("Any: failed to add tessellated face");
				this->triangulate(newFace);
			}
		} else {
			mAssert(vertexCount <= 4u);
			// We gotta remove the face here already since non-tessellated edges make problems otherwise
			// For that we need to grab some info
			OpenMesh::VertexHandle vertices[4u];
			AddedVertices edgeVertices[4u];
			auto fhIter = m_mesh->cfh_ccwbegin(face);
			for(u32 i = 0u; i < vertexCount; ++i, ++fhIter) {
				vertices[i] = m_mesh->from_vertex_handle(*fhIter);
				mAssert(vertices[i].is_valid());
				const OpenMesh::EdgeHandle eh = m_mesh->edge_handle(*fhIter);
				mAssert(eh.is_valid());
				edgeVertices[i] = m_mesh->property(m_addedVertexProp, eh);
			}

			// Copy over the properties and remove the face
			m_mesh->copy_all_properties(face, tempFace, false);
			m_mesh->delete_face(face);

			// Create the faces for the inner vertices
			if(vertexCount == 3u) {
				// Triangles

				// Spawn the inner vertices
				m_innerVertices.clear();
				this->spawn_inner_triangle_vertices(innerLevel, tempFace, vertices);
				this->tessellate_inner_triangles(innerLevel, tempFace);

				// Bridge the gap between outer and inner tessellation
				m_stripVertices.clear();
				for(u32 edgeIndex = 0u; edgeIndex < vertexCount; ++edgeIndex) {
					m_stripVertices.clear();
					const AddedVertices& outerVertices = edgeVertices[edgeIndex];
					const OpenMesh::VertexHandle from = vertices[edgeIndex];
					const OpenMesh::VertexHandle to = vertices[(edgeIndex + 1u < vertexCount) ? edgeIndex + 1u : 0u];

					// First all the edge vertices
					// Take care that we stay counter-clockwise
					m_stripVertices.push_back(from);
					if(from == outerVertices.from) {
						for(u32 i = 0u; i < outerVertices.count; ++i)
							m_stripVertices.push_back(m_edgeVertexHandles[outerVertices.offset + i]);
					} else {
						const u32 upperOffset = outerVertices.offset + outerVertices.count;
						for(u32 i = 0u; i < outerVertices.count; ++i)
							m_stripVertices.push_back(m_edgeVertexHandles[upperOffset - i - 1u]);
					}
					m_stripVertices.push_back(to);

					// Then all the inner vertices (on the border) in reverse order
					// For that we need to traverse the borders of the inner vertex set
					// Select the proper inner "edge"
					switch(edgeIndex) {
						case 0u: {
							u32 offset = innerLevel - 1u;
							for(u32 i = 0u; i < innerLevel; ++i) {
								m_stripVertices.push_back(m_innerVertices[offset]);
								--offset;
							}
						}	break;
						case 1u: {
							u32 offset = (innerLevel * innerLevel + innerLevel) / 2u - 1u;
							for(u32 i = 0u; i < innerLevel; ++i) {
								m_stripVertices.push_back(m_innerVertices[offset]);
								offset -= i + 1u;
							}
						}	break;
						case 2u: {
							u32 offset = 0u;
							for(u32 i = 0u; i < innerLevel; ++i) {
								m_stripVertices.push_back(m_innerVertices[offset]);
								offset += innerLevel - i;
							}
						}	break;
					}


					// Add the strip and triangulate
					const OpenMesh::FaceHandle newFace = m_mesh->add_face(m_stripVertices);
					if(!newFace.is_valid())
						throw std::runtime_error("Triangle: failed to add tessellated outer face");
					this->triangulate(newFace);
				}
			} else {
				// Quads

				// Spawn the inner vertices and create the quads
				m_innerVertices.clear();
				this->spawn_inner_quad_vertices(innerLevel, tempFace, vertices);
				this->tessellate_inner_quads(innerLevel, tempFace);

				// Create the faces between outer and inner tessellation
				for(u32 edgeIndex = 0u; edgeIndex < vertexCount; ++edgeIndex) {
					const OpenMesh::VertexHandle from = vertices[edgeIndex];
					const OpenMesh::VertexHandle to = vertices[(edgeIndex + 1u < vertexCount) ? edgeIndex + 1u : 0u];

					const AddedVertices& outerVertices = edgeVertices[edgeIndex];

					// Catch edge cases
					if(outerVertices.count == 0u) {
						// Just triangulate strip
						m_stripVertices.clear();
						m_stripVertices.push_back(to);
						for(u32 i = 0u; i < innerLevel; ++i)
							m_stripVertices.push_back(m_innerVertices[get_inner_vertex_index(edgeIndex, innerLevel - i - 1u, innerLevel)]);
						m_stripVertices.push_back(from);
						const OpenMesh::FaceHandle stripPoly = m_mesh->add_face(m_stripVertices);
						if(!stripPoly.is_valid())
							throw std::runtime_error("Quad: Failed to add tessellated outer face (single inner vertex)");
						if(m_stripVertices.size() > 4)
							this->triangulate(stripPoly);
					} else {
						if(outerVertices.count > 1 && innerLevel > 1 && outerVertices.count % 2 == innerLevel % 2) {
							// Determine what vertex the quads start at
							u32 startInner, startOuter;
							if(innerLevel <= outerVertices.count) {
								startInner = 0u;
								startOuter = (outerVertices.count - innerLevel) / 2u;
							} else {
								startInner = (innerLevel - outerVertices.count) / 2u;
								startOuter = 0u;
							}

							// Create as many good-looking quads as possible
							const u32 outerQuadCount = this->spawn_outer_quads(innerLevel, outerVertices.count, startInner,
																			   startOuter,outerVertices.offset, edgeIndex,
																			   from == outerVertices.from, tempFace);

							// The rest just gets triangulated
							// TODO: think about whether corners, where quads are possible, should get them
							this->spawn_outer_corner_triangles(innerLevel, startInner, startOuter, outerQuadCount,
															   edgeIndex, outerVertices, from, to, tempFace);
						} else {
							// Triangle strip only
							m_stripVertices.clear();
							// First the edge vertices...
							// Take care that we stay counter-clockwise
							m_stripVertices.push_back(from);
							if(from == outerVertices.from) {
								for(u32 i = 0u; i < outerVertices.count; ++i)
									m_stripVertices.push_back(m_edgeVertexHandles[outerVertices.offset + i]);
							} else {
								const u32 upperOffset = outerVertices.offset + outerVertices.count;
								for(u32 i = 0u; i < outerVertices.count; ++i)
									m_stripVertices.push_back(m_edgeVertexHandles[upperOffset - i - 1u]);
							}
							m_stripVertices.push_back(to);
							// ...then in reversed order the inner vertices
							for(u32 i = 0u; i < innerLevel; ++i)
								m_stripVertices.push_back(m_innerVertices[get_inner_vertex_index(edgeIndex, innerLevel - i - 1u, innerLevel)]);
							const OpenMesh::FaceHandle stripPoly = m_mesh->add_face(m_stripVertices);
							if(!stripPoly.is_valid())
								throw std::runtime_error("Quad: failed to add tessellated outer face (triangles only)");
							this->triangulate(stripPoly);
						}
					}
				}
			}
		}

	}

	// Clean up
	m_mesh->remove_property(m_addedVertexProp);
	m_mesh->delete_face(tempFace, true);
	m_mesh->garbage_collection();
	m_mesh->release_face_status();
	m_mesh->release_vertex_status();
}

void Tessellater::set_edge_vertex(const float x,
										  const OpenMesh::EdgeHandle edge,
										  const OpenMesh::VertexHandle vertex) {
	mAssert(x >= 0.f && x <= 1.f);
	const OpenMesh::VertexHandle from = m_mesh->from_vertex_handle(m_mesh->halfedge_handle(edge, 0u));
	const OpenMesh::VertexHandle to = m_mesh->to_vertex_handle(m_mesh->halfedge_handle(edge, 0u));
	const ei::Vec3& p0 = util::pun<ei::Vec3>(m_mesh->point(from));
	const ei::Vec3& p1 = util::pun<ei::Vec3>(m_mesh->point(to));
	const ei::Vec3& n0 = util::pun<ei::Vec3>(m_mesh->normal(from));
	const ei::Vec3& n1 = util::pun<ei::Vec3>(m_mesh->normal(to));
	// Use phong tessellation to have the tessellation do something useful in absence
	// of displacement mapping
	const ei::Vec3 pos = ei::lerp(p0, p1, x);
	const ei::Vec3 normal = ei::lerp(n0, n1, x);
	const ei::Vec2 uv = ei::lerp(util::pun<ei::Vec2>(m_mesh->texcoord2D(from)),
								 util::pun<ei::Vec2>(m_mesh->texcoord2D(to)), x);

	if(m_usePhongTessellation) {
		const ei::Vec3 plane0 = pos - ei::dot(pos - p0, n0) * n0;
		const ei::Vec3 plane1 = pos - ei::dot(pos - p1, n1) * n1;
		const ei::Vec3 phongPos = ei::lerp(plane0, plane1, x);
		m_mesh->set_point(vertex, util::pun<OpenMesh::Vec3f>(phongPos));
	} else {
		m_mesh->set_point(vertex, util::pun<OpenMesh::Vec3f>(pos));
	}
	m_mesh->set_normal(vertex, util::pun<OpenMesh::Vec3f>(normal));
	m_mesh->set_texcoord2D(vertex, util::pun<OpenMesh::Vec2f>(uv));
}

void Tessellater::set_quad_inner_vertex(const float x, const float y,
									const OpenMesh::VertexHandle vertex,
									const OpenMesh::FaceHandle face,
									const OpenMesh::VertexHandle(&vertices)[4u]) {
	mAssert(x >= 0.f && x <= 1.f && y >= 0.f && y <= 1.f);
	const ei::Vec3 p0 = util::pun<ei::Vec3>(m_mesh->point(vertices[0u]));
	const ei::Vec3 p1 = util::pun<ei::Vec3>(m_mesh->point(vertices[1u]));
	const ei::Vec3 p2 = util::pun<ei::Vec3>(m_mesh->point(vertices[2u]));
	const ei::Vec3 p3 = util::pun<ei::Vec3>(m_mesh->point(vertices[3u]));
	const ei::Vec3 n0 = util::pun<ei::Vec3>(m_mesh->normal(vertices[0u]));
	const ei::Vec3 n1 = util::pun<ei::Vec3>(m_mesh->normal(vertices[1u]));
	const ei::Vec3 n2 = util::pun<ei::Vec3>(m_mesh->normal(vertices[2u]));
	const ei::Vec3 n3 = util::pun<ei::Vec3>(m_mesh->normal(vertices[3u]));

	// TODO: use shading normals to adjust point
	const ei::Vec3 pos = ei::bilerp(n0, n1, n3, n2, x, y);
	const ei::Vec3 normal = ei::bilerp(p0, p1, p3, p2, x, y);
	const ei::Vec2 uv = ei::bilerp(util::pun<ei::Vec2>(m_mesh->texcoord2D(vertices[0u])),
								   util::pun<ei::Vec2>(m_mesh->texcoord2D(vertices[1u])),
								   util::pun<ei::Vec2>(m_mesh->texcoord2D(vertices[3u])),
								   util::pun<ei::Vec2>(m_mesh->texcoord2D(vertices[2u])),
								   x, y);

	// Use phong tessellation to have the tessellation do something useful in absence
	// of displacement mapping

	if(m_usePhongTessellation) {
		const ei::Vec3 plane0 = pos - ei::dot(pos - p0, n0) * n0;
		const ei::Vec3 plane1 = pos - ei::dot(pos - p1, n1) * n1;
		const ei::Vec3 plane2 = pos - ei::dot(pos - p2, n2) * n2;
		const ei::Vec3 plane3 = pos - ei::dot(pos - p3, n3) * n3;
		const ei::Vec3 phongPos = ei::bilerp(plane0, plane1, plane3, plane2, x, y);
		m_mesh->set_point(vertex, util::pun<OpenMesh::Vec3f>(phongPos));
	} else {
		m_mesh->set_point(vertex, util::pun<OpenMesh::Vec3f>(pos));
	}

	m_mesh->set_normal(vertex, util::pun<OpenMesh::Vec3f>(normal));
	m_mesh->set_texcoord2D(vertex, util::pun<OpenMesh::Vec2f>(uv));
}

void Tessellater::set_triangle_inner_vertex(const float x, const float y,
													const OpenMesh::VertexHandle vertex,
													const OpenMesh::FaceHandle face,
													const OpenMesh::VertexHandle(&vertices)[4u]) {
	mAssert(x >= 0.f && y >= 0.f && x + y <= 1.f);
	const ei::Vec3 p0 = util::pun<ei::Vec3>(m_mesh->point(vertices[0u]));
	const ei::Vec3 p1 = util::pun<ei::Vec3>(m_mesh->point(vertices[1u]));
	const ei::Vec3 p2 = util::pun<ei::Vec3>(m_mesh->point(vertices[2u]));
	const ei::Vec3 n0 = util::pun<ei::Vec3>(m_mesh->normal(vertices[0u]));
	const ei::Vec3 n1 = util::pun<ei::Vec3>(m_mesh->normal(vertices[1u]));
	const ei::Vec3 n2 = util::pun<ei::Vec3>(m_mesh->normal(vertices[2u]));

	const ei::Vec3 pos = x * p0 + y * p1 + (1.f - x - y) * p2;
	const ei::Vec3 normal = x * n0 + y * n1 + (1.f - x - y) * n2;
	const ei::Vec2 uv = x * util::pun<ei::Vec2>(m_mesh->texcoord2D(vertices[0u]))
		+ y * util::pun<ei::Vec2>(m_mesh->texcoord2D(vertices[1u]))
		+ (1.f - x - y) * util::pun<ei::Vec2>(m_mesh->texcoord2D(vertices[2u]));

	// Use phong tessellation to have the tessellation do something useful in absence
	// of displacement mapping
	if(m_usePhongTessellation) {
		const ei::Vec3 plane0 = pos - ei::dot(pos - p0, n0) * n0;
		const ei::Vec3 plane1 = pos - ei::dot(pos - p1, n1) * n1;
		const ei::Vec3 plane2 = pos - ei::dot(pos - p2, n2) * n2;
		const ei::Vec3 phongPos = x * plane0 + y * plane1 + (1.f - x - y) * plane2;
		m_mesh->set_point(vertex, util::pun<OpenMesh::Vec3f>(phongPos));
	} else {
		m_mesh->set_point(vertex, util::pun<OpenMesh::Vec3f>(pos));
	}

	m_mesh->set_normal(vertex, util::pun<OpenMesh::Vec3f>(normal));
	m_mesh->set_texcoord2D(vertex, util::pun<OpenMesh::Vec2f>(uv));
}

void Tessellater::set_quad_face_inner(const OpenMesh::FaceHandle original,
											  const OpenMesh::FaceHandle newInner) {
	m_mesh->copy_all_properties(original, newInner);
}

void Tessellater::set_quad_face_outer(const OpenMesh::FaceHandle original,
											  const OpenMesh::FaceHandle newOuter) {
	m_mesh->copy_all_properties(original, newOuter);
}

void Tessellater::set_triangle_face_inner(const OpenMesh::FaceHandle original,
												  const OpenMesh::FaceHandle newInner) {
	m_mesh->copy_all_properties(original, newInner);
}

void Tessellater::set_triangle_face_outer(const OpenMesh::FaceHandle original,
												  const OpenMesh::FaceHandle newOuter) {
	m_mesh->copy_all_properties(original, newOuter);
}

// Perfoms tessellation for quad inner level
void Tessellater::tessellate_inner_quads(const u32 innerLevel, const OpenMesh::FaceHandle original) {
	for(u32 y = 0u; y < innerLevel - 1u; ++y) {
		for(u32 x = 0u; x < innerLevel - 1u; ++x) {
			const OpenMesh::VertexHandle& v0 = m_innerVertices[y * innerLevel + x];
			const OpenMesh::VertexHandle& v1 = m_innerVertices[y * innerLevel + x + 1u];
			const OpenMesh::VertexHandle& v2 = m_innerVertices[(y + 1u) * innerLevel + x + 1u];
			const OpenMesh::VertexHandle& v3 = m_innerVertices[(y + 1u) * innerLevel + x];

			const ei::Vec3 p0 = util::pun<ei::Vec3>(m_mesh->point(v0));
			const ei::Vec3 p1 = util::pun<ei::Vec3>(m_mesh->point(v1));
			const ei::Vec3 p2 = util::pun<ei::Vec3>(m_mesh->point(v2));
			const ei::Vec3 p3 = util::pun<ei::Vec3>(m_mesh->point(v3));

			const OpenMesh::FaceHandle newFace = m_mesh->add_face(v0, v1, v2, v3);
			this->set_quad_face_inner(original, newFace);
		}
	}
}

void Tessellater::tessellate_inner_triangles(const u32 innerLevel, const OpenMesh::FaceHandle original) {
	u32 offset = 0u;
	for(u32 y = 0u; y < innerLevel - 1u; ++y) {
		const u32 verticesInRow = innerLevel - y;
		// "Upward-facing" triangles
		for(u32 x = 0u; x < verticesInRow - 1u; ++x) {
			const OpenMesh::FaceHandle newFace = m_mesh->add_face(
				m_innerVertices[offset + x], m_innerVertices[offset + x + 1u],
				m_innerVertices[offset + verticesInRow + x]
			);
			if(!newFace.is_valid())
				throw std::runtime_error("Triangle: failed to add inner tessellated face (upwards)");
			this->set_triangle_face_inner(original, newFace);
		}
		// "Downward-facing" triangles
		for(u32 x = 1u; x < verticesInRow - 1u; ++x) {
			const OpenMesh::FaceHandle newFace = m_mesh->add_face(
				m_innerVertices[offset + x], m_innerVertices[offset + verticesInRow + x],
				m_innerVertices[offset + verticesInRow + x - 1u]
			);
			if(!newFace.is_valid())
				throw std::runtime_error("Triangle: failed to add inner tessellated face (downwards)");
			this->set_triangle_face_inner(original, newFace);
		}
		offset += verticesInRow;
	}
}

void Tessellater::triangulate(const OpenMesh::FaceHandle face) {
	// TODO: implement Delaunay
	m_mesh->triangulate(face);
	// TODO: post-process!
}

void Tessellater::spawn_inner_quad_vertices(const u32 innerLevel,
													const OpenMesh::FaceHandle face,
													const OpenMesh::VertexHandle (&vertices)[4u]) {
	// Spawn the inner vertices
	for(u32 y = 0u; y < innerLevel; ++y) {
		for(u32 x = 0u; x < innerLevel; ++x) {
			const OpenMesh::VertexHandle newVertex = m_mesh->add_vertex(OpenMesh::Vec3f{});
			if(!newVertex.is_valid())
				throw std::runtime_error("Quad: failed to add inner vertex");
			m_innerVertices.push_back(newVertex);
			this->set_quad_inner_vertex(static_cast<float>(x + 1u) / static_cast<float>(innerLevel + 1u),
										static_cast<float>(y + 1u) / static_cast<float>(innerLevel + 1u),
										m_innerVertices.back(), face, vertices);
		}
	}
}

void Tessellater::spawn_inner_triangle_vertices(const u32 innerLevel,
														const OpenMesh::FaceHandle face,
														const OpenMesh::VertexHandle(&vertices)[4u]) {
	for(u32 y = 0u; y < innerLevel; ++y) {
		const float edgeBaryZ = (y + 1u) / static_cast<float>(innerLevel + 1u);
		const float edgeBaryX = 1.f - edgeBaryZ;
		const float bZ = (y + 1u) / static_cast<float>(innerLevel + 2u);
		for(u32 x = 0u; x < innerLevel - y; ++x) {
			const OpenMesh::VertexHandle newVertex = m_mesh->add_vertex(OpenMesh::Vec3f{});
			if(!newVertex.is_valid())
				throw std::runtime_error("Triangle: failed to add inner vertex");
			m_innerVertices.push_back(newVertex);
			const float bX = ((innerLevel - y) - x) / static_cast<float>(innerLevel + 2u);

			const float baryX = edgeBaryX - edgeBaryX * (x + 1u) / static_cast<float>(innerLevel - y + 1u);
			const float baryY = 1.f - edgeBaryZ - baryX;
			this->set_triangle_inner_vertex(bX, 1.f - bZ - bX, newVertex, face, vertices);
		}
	}
}

u32 Tessellater::spawn_outer_quads(const u32 innerLevel, const u32 outerLevel,
										   const u32 startInner, const u32 startOuter,
										   const u32 edgeVertexOffset, const u32 edgeIndex,
										   const bool swapEdgeVertices, const OpenMesh::FaceHandle face) {
	const u32 stripeQuadCount = std::min(innerLevel, outerLevel) - 1u;
	for(u32 i = 0u; i < stripeQuadCount; ++i) {
		// Compute what edge of the inner quad the vertices are on
		// Make sure that the vertices are counter-clockwise
		u32 i0, i1;
		if(swapEdgeVertices) {
			i0 = edgeVertexOffset + outerLevel - (startOuter + i + 2u);
			i1 = edgeVertexOffset + outerLevel - (startOuter + i + 1u);
		} else {
			i0 = edgeVertexOffset + startOuter + i + 1u;
			i1 = edgeVertexOffset + startOuter + i;
		}
		const u32 i2 = get_inner_vertex_index(edgeIndex, startInner + i + 1u, innerLevel);
		const u32 i3 = get_inner_vertex_index(edgeIndex, startInner + i, innerLevel);

		const OpenMesh::VertexHandle& v0 = m_edgeVertexHandles[i0];
		const OpenMesh::VertexHandle& v1 = m_edgeVertexHandles[i1];
		const OpenMesh::VertexHandle& v2 = m_innerVertices[i2];
		const OpenMesh::VertexHandle& v3 = m_innerVertices[i3];

		// Add the face and post-process
		const OpenMesh::FaceHandle newFace = m_mesh->add_face(v0, v1, v2, v3);
		if(!newFace.is_valid())
			throw std::runtime_error("Quad: Failed to add tessellated outer face (quads)");
		this->set_quad_face_outer(face, newFace);
	}
	return stripeQuadCount;
}

void Tessellater::spawn_outer_corner_triangles(const u32 innerLevel, const u32 startInner,
													   const u32 startOuter, const u32 outerQuadCount,
													   const u32 edgeIndex, const AddedVertices& outerVertices,
													   const OpenMesh::VertexHandle from,
													   const OpenMesh::VertexHandle to,
													   const OpenMesh::FaceHandle face) {
	// Left corner vertices (all that are not part of the quads)
	m_stripVertices.clear();
	m_stripVertices.push_back(from);
	if(from != outerVertices.from) {
		const u32 maxOffset = outerVertices.offset + outerVertices.count;
		for(u32 i = 0u; i < startOuter + 1u; ++i)
			m_stripVertices.push_back(m_edgeVertexHandles[maxOffset - i - 1u]);
	} else {
		for(u32 i = 0u; i < startOuter + 1u; ++i)
			m_stripVertices.push_back(m_edgeVertexHandles[outerVertices.offset + i]);
	}
	for(u32 i = 0u; i < startInner + 1u; ++i)
		m_stripVertices.push_back(m_innerVertices[get_inner_vertex_index(edgeIndex, startInner - i, innerLevel)]);
	const OpenMesh::FaceHandle leftFace = m_mesh->add_face(m_stripVertices);
	// Right corner vertices (all that are not part of the quads)
	m_stripVertices.clear();
	m_stripVertices.push_back(to);
	const u32 startIndexInner = startInner + outerQuadCount;
	for(u32 i = startIndexInner; i < innerLevel; ++i)
		m_stripVertices.push_back(m_innerVertices[get_inner_vertex_index(edgeIndex, innerLevel - i - 1u + startIndexInner, innerLevel)]);

	if(from != outerVertices.from) {
		const u32 maxOffset = outerVertices.offset + outerVertices.count;
		for(u32 i = startOuter + outerQuadCount; i < outerVertices.count; ++i)
			m_stripVertices.push_back(m_edgeVertexHandles[maxOffset - i - 1u]);
	} else {
		for(u32 i = startOuter + outerQuadCount; i < outerVertices.count; ++i)
			m_stripVertices.push_back(m_edgeVertexHandles[outerVertices.offset + i]);
	}
	const OpenMesh::FaceHandle rightFace = m_mesh->add_face(m_stripVertices);

	// Make sure we succeeded in adding the n-gons, then triangulate and post-process
	if(!leftFace.is_valid())
		throw std::runtime_error("Quad: failed to add tessellated outer face (left corner triangles)");
	if(!rightFace.is_valid())
		throw std::runtime_error("Quad: failed to add tessellated outer face (right corner triangles)");
	this->triangulate(leftFace);
	this->triangulate(rightFace);
	this->set_quad_face_outer(face, leftFace);
	this->set_quad_face_outer(face, rightFace);
}

} // namespace mufflon::scene::tessellation