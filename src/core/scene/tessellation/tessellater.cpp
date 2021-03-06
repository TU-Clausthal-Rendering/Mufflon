#include "tessellater.hpp"
#include "util/punning.hpp"
#include "util/log.hpp"
#include "profiler/cpu_profiler.hpp"

namespace mufflon::scene::tessellation {

namespace {

OpenMesh::FaceHandle create_dummy_face(geometry::PolygonMeshType& mesh) {
	const OpenMesh::VertexHandle tempV0 = mesh.add_vertex(OpenMesh::Vec3f{});
	const OpenMesh::VertexHandle tempV1 = mesh.add_vertex(OpenMesh::Vec3f{});
	const OpenMesh::VertexHandle tempV2 = mesh.add_vertex(OpenMesh::Vec3f{});
	if(!tempV0.is_valid() || !tempV1.is_valid() || !tempV2.is_valid())
		throw std::runtime_error("Failed to add temporary vertices");
	const OpenMesh::FaceHandle tempFace = mesh.add_face(tempV0, tempV1, tempV2);
	if(!tempFace.is_valid())
		throw std::runtime_error("Failed to add temporary face");
	return tempFace;
}

} // namespace

OpenMesh::FaceHandle Tessellater::tessellate(geometry::PolygonMeshType& mesh) {
	auto profilerTimer = Profiler::core().start<CpuProfileState>("Tessellater::tessellate", ProfileLevel::HIGH);
	bool releaseVertexStatus = false;
	bool releaseFaceStatus = false;
	if(!mesh.has_vertex_status()) {
		mesh.request_vertex_status();
		releaseVertexStatus = true;
	}
	if(!mesh.has_face_status()) {
		mesh.request_face_status();
		releaseFaceStatus = true;
	}

	// Setup
	std::vector<std::pair<u32, u32>> newFaceIndices;
	m_mesh = &mesh;
	m_tessLevelOracle.set_mesh(&mesh);
	m_tessLevelOracle.set_phong_tessellation(m_usePhongTessellation);
	m_edgeVertexHandles.clear();
	m_innerVertices.clear();
	this->pre_tessellate();
	// We need to be able to delete faces and vertices
	if(m_mesh->add_property(m_addedVertexProp); !m_addedVertexProp.is_valid())
		throw std::runtime_error("Failed to add edge vertex property to mesh");
	if(m_mesh->add_property(m_oldFace); !m_oldFace.is_valid())
		throw std::runtime_error("Failed to add old face property to mesh");
	for(const auto face : m_mesh->faces())
		m_mesh->property(m_oldFace, face) = OpenMesh::FaceHandle{};
	// TODO: reserve proper amount for new faces

	// Spawn edge (outer) vertices and store necessary information
	this->tessellate_edges();

	// We need a temporary face to copy over properties in case we have a tessellation level of zero
	// at some edge
	const OpenMesh::FaceHandle tempFace = create_dummy_face(*m_mesh);

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

		// This vector holds the vertex and edge information of the current face in CCW order
		// Note that the edge information may be flipped locally
		static std::vector<std::pair<OpenMesh::VertexHandle, AddedVertices>> vertices;
		vertices.clear();

		auto hehIter = m_mesh->cfh_ccwbegin(face);
		u32 maxOuterLevel = 0u;
		for(u32 i = 0u; hehIter.is_valid(); ++hehIter, ++i) {
			const OpenMesh::EdgeHandle eh = m_mesh->edge_handle(*hehIter);
			vertices.push_back({ m_mesh->from_vertex_handle(*hehIter), m_mesh->property(m_addedVertexProp, eh) });
			maxOuterLevel = std::max(maxOuterLevel, vertices.back().second.count);
		}

		// Check if we're in a triangle or quad
		mAssert(vertices.size() <= 4u);

		// Create the faces for the inner vertices
		if(vertices.size() == 3u) {
			u32 innerLevel = m_tessLevelOracle.get_triangle_inner_tessellation_level(face);
			// Check if we have outer tessellation level (and bump it if necessary)
			if(innerLevel == 0u) {
				if(maxOuterLevel == 0u)
					continue;
				else
					innerLevel = 1u;
			}
			// We gotta remove the face here already since non-tessellated edges make problems otherwise
			// Copy over the properties and remove the face
			m_mesh->copy_all_properties(face, tempFace, false);
			m_mesh->property(m_oldFace, tempFace) = face;
			m_mesh->delete_face(face, false);

			this->tessellate_triangle(vertices, tempFace, innerLevel);
		} else {
			// Find out the inner tessellation level
			auto[innerLevelX, innerLevelY] = m_tessLevelOracle.get_quad_inner_tessellation_level(face);
			// Ensure we don't tessellate needlessly
			if(innerLevelX == 0 || innerLevelY == 0) {
				if(maxOuterLevel == 0u) {
					continue;
				} else {
					// If there's even one outer level we must tessellate
					innerLevelX = std::max(1u, innerLevelX);
					innerLevelY = std::max(1u, innerLevelY);
				}
			}
			// We gotta remove the face here already since non-tessellated edges make problems otherwise
			// Copy over the properties and remove the face
			m_mesh->copy_all_properties(face, tempFace, false);
			m_mesh->property(m_oldFace, tempFace) = face;
			m_mesh->delete_face(face, false);

			this->tessellate_quad(vertices, tempFace, innerLevelX, innerLevelY);
		}
	}

	// Clean up
	m_mesh->remove_property(m_addedVertexProp);
	m_mesh->delete_face(tempFace, true);
	if(releaseVertexStatus)
		m_mesh->release_vertex_status();
	if(releaseFaceStatus)
		m_mesh->release_face_status();

	this->post_tessellate();
	return tempFace;
}

void Tessellater::tessellate_edges() {
	for(auto edge : m_mesh->edges()) {
		const OpenMesh::VertexHandle from = m_mesh->from_vertex_handle(m_mesh->halfedge_handle(edge, 0));

		// Ask how many new vertices this edge should get
		const u32 newEdgeVertices = m_tessLevelOracle.get_edge_tessellation_level(edge);
		m_mesh->property(m_addedVertexProp, edge) = AddedVertices{ from, static_cast<u32>(m_edgeVertexHandles.size()), newEdgeVertices };
		// For each new vertex, ask to set the position, normal etc. (whatever they want)
		for(u32 i = 0u; i < newEdgeVertices; ++i) {
			const OpenMesh::VertexHandle newVertex = m_mesh->add_vertex(OpenMesh::Vec3f{});
			if(!newVertex.is_valid())
				throw std::runtime_error("Failed to add edge vertex");
			m_edgeVertexHandles.push_back(newVertex);
			this->set_edge_vertex(static_cast<float>(i + 1u) / static_cast<float>(newEdgeVertices + 1u),
								  edge, m_edgeVertexHandles.back());
		}
		// Also set the new position for the from-vertex of every edge in case of displacement mapping
		this->set_edge_vertex(0, edge, from);
	}
}

void Tessellater::tessellate_triangle(const std::vector<std::pair<OpenMesh::VertexHandle, AddedVertices>>& vertices,
									  const OpenMesh::FaceHandle face, const u32 innerLevel) {
	// Spawn the inner vertices
	m_innerVertices.clear();
	this->spawn_inner_triangle_vertices(innerLevel, face, vertices);
	this->tessellate_inner_triangles(innerLevel, face);

	// Bridge the gap between outer and inner tessellation
	m_stripVertices.clear();
	for(u32 edgeIndex = 0u; edgeIndex < 3u; ++edgeIndex) {
		m_stripVertices.clear();
		const AddedVertices& outerVertices = vertices[edgeIndex].second;
		const OpenMesh::VertexHandle from = vertices[edgeIndex].first;
		const OpenMesh::VertexHandle to = vertices[(edgeIndex + 1u < 3u) ? edgeIndex + 1u : 0u].first;
		const bool edgeNeedsFlip = from != outerVertices.from;

		// First all the edge vertices
		// Take care that we stay counter-clockwise
		m_stripVertices.push_back(from);
		if(edgeNeedsFlip) {
			const u32 upperOffset = outerVertices.offset + outerVertices.count;
			for(u32 i = 0u; i < outerVertices.count; ++i)
				m_stripVertices.push_back(m_edgeVertexHandles[upperOffset - i - 1u]);
		} else {
			for(u32 i = 0u; i < outerVertices.count; ++i)
				m_stripVertices.push_back(m_edgeVertexHandles[outerVertices.offset + i]);
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

		this->triangulate_strip(outerVertices.count, innerLevel, face);
	}
}

void Tessellater::tessellate_quad(const std::vector<std::pair<OpenMesh::VertexHandle, AddedVertices>>& vertices,
								  const OpenMesh::FaceHandle face, const u32 innerLevelX, const u32 innerLevelY) {
	// Spawn the inner vertices and create the quads
	m_innerVertices.clear();
	this->spawn_inner_quad_vertices(innerLevelX, innerLevelY, face, vertices);
	this->tessellate_inner_quads(innerLevelX, innerLevelY, face);

	// Create the faces between outer and inner tessellation
	for(u32 edgeIndex = 0u; edgeIndex < 4u; ++edgeIndex) {
		const AddedVertices& outerVertices = vertices[edgeIndex].second;
		const OpenMesh::VertexHandle from = vertices[edgeIndex].first;
		const OpenMesh::VertexHandle to = vertices[(edgeIndex + 1u < 4u) ? edgeIndex + 1u : 0u].first;
		const bool edgeNeedsFlip = from != outerVertices.from;

		// Keep track of the current edge tessellation level
		const u32 currInnerLevel = edgeIndex % 2 == 0 ? innerLevelX : innerLevelY;
		const u32 otherInnerLevel = edgeIndex % 2 != 0 ? innerLevelX : innerLevelY;

		// Catch edge cases
		if(outerVertices.count == 0u) {
			if(currInnerLevel == 2u) {
				// Create one big quad
				const OpenMesh::VertexHandle innerLeft = get_inner_vertex_quad(edgeIndex, 0u, innerLevelX, innerLevelY);
				const OpenMesh::VertexHandle innerRight = get_inner_vertex_quad(edgeIndex, 1u, innerLevelX, innerLevelY);
				const OpenMesh::FaceHandle newFace = m_mesh->add_face(from, to, innerRight, innerLeft);
				if(!newFace.is_valid())
					throw std::runtime_error("Tessellate: failed to add single big edge quad");
				this->set_quad_face_outer(face, newFace);
			} else {
				// Just triangulate strip
				m_stripVertices.clear();
				m_stripVertices.push_back(from);
				m_stripVertices.push_back(to);
				for(u32 i = 0u; i < currInnerLevel; ++i)
					m_stripVertices.push_back(get_inner_vertex_quad(edgeIndex, currInnerLevel - i - 1u,
																	innerLevelX, innerLevelY));
				this->triangulate_strip(0u, currInnerLevel, face);
			}
		} else {
			// Check what the next face edge holds in terms of level
			const u32 nextEdgeIndex = edgeIndex + 1u < 4u ? (edgeIndex + 1u) : 0u;
			const AddedVertices& nextOuterVertices = vertices[nextEdgeIndex].second;

			// Make sure that connecting quads look good by keeping a minimum ratio between inner and outer level
			const float innerOuterRatio = static_cast<float>(outerVertices.count) / static_cast<float>(currInnerLevel);
			if(outerVertices.count % 2 == currInnerLevel % 2 && std::abs(innerOuterRatio - 1.f) < 0.075f) {
				// Determine what vertex the quads start at
				u32 startInner, startOuter;
				if(currInnerLevel <= outerVertices.count) {
					startInner = 0u;
					startOuter = (outerVertices.count - currInnerLevel) / 2u;
				} else {
					startInner = (currInnerLevel - outerVertices.count) / 2u;
					startOuter = 0u;
				}

				// Create as many good-looking quads as possible
				const u32 outerQuadCount = this->spawn_outer_quads(innerLevelX, innerLevelY, currInnerLevel,
																   outerVertices.count, startInner,
																   startOuter, outerVertices.offset, edgeIndex,
																   edgeNeedsFlip, face);

				// The corner needs to be either triangulated or filled with a quad
				const u32 prevEdgeIndex = edgeIndex > 0u ? (edgeIndex - 1u) : 3u;
				const AddedVertices& prevOuterVertices = vertices[prevEdgeIndex].second;
				// Check if the previous edge also has space for a quad
				if(currInnerLevel == outerVertices.count && otherInnerLevel == prevOuterVertices.count) {
					// Spawn single corner quad
					this->spawn_outer_quad_corner_region(innerLevelX, innerLevelY, currInnerLevel, otherInnerLevel,
													     startInner, startOuter, edgeNeedsFlip, edgeIndex, outerQuadCount,
													     from, to, vertices[prevEdgeIndex].first, face, outerVertices,
													     prevOuterVertices, nextOuterVertices);
				} else {
					// We can't put a quad with our previous edge
					this->spawn_outer_corner_triangles(innerLevelX, innerLevelY, currInnerLevel, startInner,
													   startOuter, outerQuadCount, edgeIndex, outerVertices,
													   from, to, face, true,  currInnerLevel != outerVertices.count
													   || otherInnerLevel != nextOuterVertices.count);

				}
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
				for(u32 i = 0u; i < currInnerLevel; ++i)
					m_stripVertices.push_back(get_inner_vertex_quad(edgeIndex, currInnerLevel - i - 1u, innerLevelX, innerLevelY));
				this->triangulate_strip(outerVertices.count, currInnerLevel, face);
			}
		}
	}
}

void Tessellater::spawn_outer_quad_corner_region(const u32 innerLevelX, const u32 innerLevelY, const u32 currInnerLevel,
												 const u32 otherInnerLevel, const u32 startInner, const u32 startOuter,
												 const bool edgeNeedsFlip, const u32 edgeIndex, const u32 outerQuadCount,
												 const OpenMesh::VertexHandle from, const OpenMesh::VertexHandle to,
												 const OpenMesh::VertexHandle prevFrom,
												 const OpenMesh::FaceHandle face, const AddedVertices& outerVertices,
												 const AddedVertices& prevOuterVertices, const AddedVertices& nextOuterVertices) {
		// Spawn single corner quad
		const OpenMesh::VertexHandle v0 = from;
		OpenMesh::VertexHandle v1;
		if(edgeNeedsFlip)
			v1 = m_edgeVertexHandles[outerVertices.offset + outerVertices.count - 1u];
		else
			v1 = m_edgeVertexHandles[outerVertices.offset];
		const OpenMesh::VertexHandle v2 = get_inner_vertex_quad(edgeIndex, startInner, innerLevelX, innerLevelY);
		OpenMesh::VertexHandle v3;
		// Check for edge flip on previous edge
		if(prevFrom == prevOuterVertices.from)
			v3 = m_edgeVertexHandles[prevOuterVertices.offset + prevOuterVertices.count - 1u];
		else
			v3 = m_edgeVertexHandles[prevOuterVertices.offset];

		const OpenMesh::FaceHandle newFace = m_mesh->add_face(v0, v1, v2, v3);
		if(!newFace.is_valid())
			throw std::runtime_error("Any: failed to add tessellated outer face (corner quad)");
		this->set_quad_face_outer(face, newFace);

		// Since we're responsible for the right corner too we need to check if the next edge
		// may place a quad
		if(otherInnerLevel != nextOuterVertices.count)
			this->spawn_outer_corner_triangles(innerLevelX, innerLevelY, currInnerLevel, startInner,
											   startOuter, outerQuadCount, edgeIndex, outerVertices,
											   from, to, face, false, true);
}

void Tessellater::set_edge_vertex(const float x, const OpenMesh::EdgeHandle edge,
								  const OpenMesh::VertexHandle vertex) {
	if(x == 0)
		return;
	mAssert(x >= 0.f && x <= 1.f);
	const OpenMesh::VertexHandle from = m_mesh->from_vertex_handle(m_mesh->halfedge_handle(edge, 0u));
	const OpenMesh::VertexHandle to = m_mesh->to_vertex_handle(m_mesh->halfedge_handle(edge, 0u));
	const ei::Vec3 p0 = util::pun<ei::Vec3>(m_mesh->point(from));
	const ei::Vec3 p1 = util::pun<ei::Vec3>(m_mesh->point(to));
	const ei::Vec3 n0 = util::pun<ei::Vec3>(m_mesh->normal(from));
	const ei::Vec3 n1 = util::pun<ei::Vec3>(m_mesh->normal(to));
	// Use phong tessellation to have the tessellation do something useful in absence
	// of displacement mapping
	const ei::Vec3 pos = ei::lerp(p0, p1, x);
	const ei::Vec3 normal = normalize(ei::lerp(n0, n1, x));
	const ei::Vec2 uv = ei::lerp(util::pun<ei::Vec2>(m_mesh->texcoord2D(from)),
								 util::pun<ei::Vec2>(m_mesh->texcoord2D(to)), x);

	if(m_usePhongTessellation) {
		const ei::Vec3 plane0 = pos - ei::dot(pos - p0, n0) * n0;
		const ei::Vec3 plane1 = pos - ei::dot(pos - p1, n1) * n1;
		const ei::Vec3 phongPos = ei::lerp(plane0, plane1, x);
		const ei::Vec3 outPos = lerp(pos, phongPos, PHONGTESS_ALPHA);
		m_mesh->set_point(vertex, util::pun<OpenMesh::Vec3f>(outPos));
	} else {
		m_mesh->set_point(vertex, util::pun<OpenMesh::Vec3f>(pos));
	}
	m_mesh->set_normal(vertex, util::pun<OpenMesh::Vec3f>(normal));
	m_mesh->set_texcoord2D(vertex, util::pun<OpenMesh::Vec2f>(uv));
}

void Tessellater::set_quad_inner_vertex(const float x, const float y,
										const OpenMesh::VertexHandle vertex,
										const OpenMesh::FaceHandle /*face*/,
										const std::vector<std::pair<OpenMesh::VertexHandle, AddedVertices>>& vertices) {
	mAssert(x >= 0.f && x <= 1.f && y >= 0.f && y <= 1.f);
	const ei::Vec3 p0 = util::pun<ei::Vec3>(m_mesh->point(vertices[0u].first));
	const ei::Vec3 p1 = util::pun<ei::Vec3>(m_mesh->point(vertices[1u].first));
	const ei::Vec3 p2 = util::pun<ei::Vec3>(m_mesh->point(vertices[2u].first));
	const ei::Vec3 p3 = util::pun<ei::Vec3>(m_mesh->point(vertices[3u].first));
	const ei::Vec3 n0 = util::pun<ei::Vec3>(m_mesh->normal(vertices[0u].first));
	const ei::Vec3 n1 = util::pun<ei::Vec3>(m_mesh->normal(vertices[1u].first));
	const ei::Vec3 n2 = util::pun<ei::Vec3>(m_mesh->normal(vertices[2u].first));
	const ei::Vec3 n3 = util::pun<ei::Vec3>(m_mesh->normal(vertices[3u].first));

	// TODO: use shading normals to adjust point
	const ei::Vec3 normal = normalize(ei::bilerp(n0, n1, n3, n2, x, y));
	const ei::Vec3 pos = ei::bilerp(p0, p1, p3, p2, x, y);
	const ei::Vec2 uv = ei::bilerp(util::pun<ei::Vec2>(m_mesh->texcoord2D(vertices[0u].first)),
								   util::pun<ei::Vec2>(m_mesh->texcoord2D(vertices[1u].first)),
								   util::pun<ei::Vec2>(m_mesh->texcoord2D(vertices[3u].first)),
								   util::pun<ei::Vec2>(m_mesh->texcoord2D(vertices[2u].first)),
								   x, y);

	// Use phong tessellation to have the tessellation do something useful in absence
	// of displacement mapping

	if(m_usePhongTessellation) {
		const ei::Vec3 plane0 = pos - ei::dot(pos - p0, n0) * n0;
		const ei::Vec3 plane1 = pos - ei::dot(pos - p1, n1) * n1;
		const ei::Vec3 plane2 = pos - ei::dot(pos - p2, n2) * n2;
		const ei::Vec3 plane3 = pos - ei::dot(pos - p3, n3) * n3;
		const ei::Vec3 phongPos = ei::bilerp(plane0, plane1, plane3, plane2, x, y);
		const ei::Vec3 outPos = lerp(pos, phongPos, PHONGTESS_ALPHA);
		m_mesh->set_point(vertex, util::pun<OpenMesh::Vec3f>(outPos));
	} else {
		m_mesh->set_point(vertex, util::pun<OpenMesh::Vec3f>(pos));
	}

	m_mesh->set_normal(vertex, util::pun<OpenMesh::Vec3f>(normal));
	m_mesh->set_texcoord2D(vertex, util::pun<OpenMesh::Vec2f>(uv));
}

void Tessellater::set_triangle_inner_vertex(const float x, const float y,
											const OpenMesh::VertexHandle vertex,
											const OpenMesh::FaceHandle /*face*/,
											const std::vector<std::pair<OpenMesh::VertexHandle, AddedVertices>>& vertices) {
	mAssert(x >= 0.f && y >= 0.f && x + y <= 1.f);
	const ei::Vec3 p0 = util::pun<ei::Vec3>(m_mesh->point(vertices[0u].first));
	const ei::Vec3 p1 = util::pun<ei::Vec3>(m_mesh->point(vertices[1u].first));
	const ei::Vec3 p2 = util::pun<ei::Vec3>(m_mesh->point(vertices[2u].first));
	const ei::Vec3 n0 = util::pun<ei::Vec3>(m_mesh->normal(vertices[0u].first));
	const ei::Vec3 n1 = util::pun<ei::Vec3>(m_mesh->normal(vertices[1u].first));
	const ei::Vec3 n2 = util::pun<ei::Vec3>(m_mesh->normal(vertices[2u].first));

	const ei::Vec3 pos = x * p0 + y * p1 + (1.f - x - y) * p2;
	const ei::Vec3 normal = normalize(x * n0 + y * n1 + (1.f - x - y) * n2);
	const ei::Vec2 uv = x * util::pun<ei::Vec2>(m_mesh->texcoord2D(vertices[0u].first))
		+ y * util::pun<ei::Vec2>(m_mesh->texcoord2D(vertices[1u].first))
		+ (1.f - x - y) * util::pun<ei::Vec2>(m_mesh->texcoord2D(vertices[2u].first));

	// Use phong tessellation to have the tessellation do something useful in absence
	// of displacement mapping
	if(m_usePhongTessellation) {
		const ei::Vec3 plane0 = pos - ei::dot(pos - p0, n0) * n0;
		const ei::Vec3 plane1 = pos - ei::dot(pos - p1, n1) * n1;
		const ei::Vec3 plane2 = pos - ei::dot(pos - p2, n2) * n2;
		const ei::Vec3 phongPos = x * plane0 + y * plane1 + (1.f - x - y) * plane2;
		const ei::Vec3 outPos = lerp(pos, phongPos, PHONGTESS_ALPHA);
		m_mesh->set_point(vertex, util::pun<OpenMesh::Vec3f>(outPos));
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
void Tessellater::tessellate_inner_quads(const u32 innerLevelX, const u32 innerLevelY,
										 const OpenMesh::FaceHandle original) {
	// Simple pattern: go over a grid, grab the inner vertices in that grid, and create the new face
	for(u32 y = 0u; y < innerLevelY - 1u; ++y) {
		for(u32 x = 0u; x < innerLevelX - 1u; ++x) {
			const OpenMesh::VertexHandle& v0 = m_innerVertices[y * innerLevelX + x];
			const OpenMesh::VertexHandle& v1 = m_innerVertices[y * innerLevelX + x + 1u];
			const OpenMesh::VertexHandle& v2 = m_innerVertices[(y + 1u) * innerLevelX + x + 1u];
			const OpenMesh::VertexHandle& v3 = m_innerVertices[(y + 1u) * innerLevelX + x];

			const OpenMesh::FaceHandle newFace = m_mesh->add_face(v0, v1, v2, v3);
			this->set_quad_face_inner(original, newFace);
		}
	}
}

void Tessellater::tessellate_inner_triangles(const u32 innerLevel, const OpenMesh::FaceHandle original) {
	// Go over a grid, grab the inner vertices in that grid, and create the new face
	// This is slightly more tricky for triangles than for quads since every "row" has both
	// upwards and downwards facing triangles
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

void Tessellater::spawn_inner_quad_vertices(const u32 innerLevelX, const u32 innerLevelY,
											const OpenMesh::FaceHandle face,
											const std::vector<std::pair<OpenMesh::VertexHandle, AddedVertices>>& vertices) {
	// Spawn the inner vertices on the tessellation grid
	for(u32 y = 0u; y < innerLevelY; ++y) {
		for(u32 x = 0u; x < innerLevelX; ++x) {
			const OpenMesh::VertexHandle newVertex = m_mesh->add_vertex(OpenMesh::Vec3f{});
			if(!newVertex.is_valid())
				throw std::runtime_error("Quad: failed to add inner vertex");
			m_innerVertices.push_back(newVertex);
			this->set_quad_inner_vertex(static_cast<float>(x + 1u) / static_cast<float>(innerLevelX + 1u),
										static_cast<float>(y + 1u) / static_cast<float>(innerLevelY + 1u),
										m_innerVertices.back(), face, vertices);
		}
	}
}

void Tessellater::spawn_inner_triangle_vertices(const u32 innerLevel,
												const OpenMesh::FaceHandle face,
												const std::vector<std::pair<OpenMesh::VertexHandle, AddedVertices>>& vertices) {
	// Spawn the inner vertices on the tessellation grid
	for(u32 y = 0u; y < innerLevel; ++y) {
		const float edgeBaryZ = (y + 1u) / static_cast<float>(innerLevel + 1u);
		const float edgeBaryX = 1.f - edgeBaryZ;
		for(u32 x = 0u; x < innerLevel - y; ++x) {
			const OpenMesh::VertexHandle newVertex = m_mesh->add_vertex(OpenMesh::Vec3f{});
			if(!newVertex.is_valid())
				throw std::runtime_error("Triangle: failed to add inner vertex");
			m_innerVertices.push_back(newVertex);
			const float baryX = edgeBaryX - edgeBaryX * (x + 1u) / static_cast<float>(innerLevel - y + 1u);
			const float baryY = 1.f - edgeBaryZ - baryX;
			this->set_triangle_inner_vertex(baryX, baryY, newVertex, face, vertices);
		}
	}
}

u32 Tessellater::spawn_outer_quads(const u32 innerLevelX, const u32 innerLevelY,
								   const u32 currInnerLevel, const u32 outerLevel,
								   const u32 startInner, const u32 startOuter,
								   const u32 edgeVertexOffset, const u32 edgeIndex,
								   const bool swapEdgeVertices, const OpenMesh::FaceHandle face) {
	const u32 stripeQuadCount = std::min(currInnerLevel, outerLevel) - 1u;
	for(u32 i = 0u; i < stripeQuadCount; ++i) {
		// Compute what edge of the inner quad the vertices are on
		// Make sure that the vertices are counter-clockwise
		u32 i0, i1;
		if(swapEdgeVertices) {
			i0 = edgeVertexOffset + outerLevel - (startOuter + i + 1u);
			i1 = edgeVertexOffset + outerLevel - (startOuter + i + 2u);
		} else {
			i0 = edgeVertexOffset + startOuter + i;
			i1 = edgeVertexOffset + startOuter + i + 1u;
		}

		const OpenMesh::VertexHandle& v0 = m_edgeVertexHandles[i0];
		const OpenMesh::VertexHandle& v1 = m_edgeVertexHandles[i1];
		const OpenMesh::VertexHandle& v2 = get_inner_vertex_quad(edgeIndex, startInner + i + 1u, innerLevelX, innerLevelY);
		const OpenMesh::VertexHandle& v3 = get_inner_vertex_quad(edgeIndex, startInner + i, innerLevelX, innerLevelY);

		// Add the face and post-process
		const OpenMesh::FaceHandle newFace = m_mesh->add_face(v0, v1, v2, v3);
		if(!newFace.is_valid())
			throw std::runtime_error("Quad: Failed to add tessellated outer face (quads)");
		this->set_quad_face_outer(face, newFace);
	}
	return stripeQuadCount;
}

void Tessellater::spawn_outer_corner_triangles(const u32 innerLevelX, const u32 innerLevelY, const u32 currInnerLevel,
											   const u32 startInner, const u32 startOuter, const u32 outerQuadCount,
											   const u32 edgeIndex, const AddedVertices& outerVertices,
											   const OpenMesh::VertexHandle from,
											   const OpenMesh::VertexHandle to,
											   const OpenMesh::FaceHandle face,
											   bool doLeft, bool doRight) {
	// Left corner vertices (all that are not part of the quads)
	if(doLeft) {
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
			m_stripVertices.push_back(get_inner_vertex_quad(edgeIndex, startInner - i, innerLevelX, innerLevelY));
		this->triangulate_strip(startOuter, startInner + 1u, face);
	}
	if(doRight) {
		// Right corner vertices (all that are not part of the quads)
		m_stripVertices.clear();
		if(from != outerVertices.from) {
			const u32 maxOffset = outerVertices.offset + outerVertices.count;
			for(u32 i = startOuter + outerQuadCount; i < outerVertices.count; ++i)
				m_stripVertices.push_back(m_edgeVertexHandles[maxOffset - i - 1u]);
		} else {
			for(u32 i = startOuter + outerQuadCount; i < outerVertices.count; ++i)
				m_stripVertices.push_back(m_edgeVertexHandles[outerVertices.offset + i]);
		}
		m_stripVertices.push_back(to);
		const u32 startIndexInner = startInner + outerQuadCount;
		for(u32 i = startIndexInner; i < currInnerLevel; ++i)
			m_stripVertices.push_back(get_inner_vertex_quad(edgeIndex, currInnerLevel - i - 1u + startIndexInner,
															innerLevelX, innerLevelY));
		if(m_stripVertices.size() >= 3u)
			this->triangulate_strip(outerVertices.count - startOuter - outerQuadCount - 1u, currInnerLevel - startIndexInner, face);
	}
}

void Tessellater::triangulate_strip(const u32 outerLevel, const u32 innerLevel,
									const OpenMesh::FaceHandle original) {
	if(outerLevel == 0u && innerLevel == 0u)
		return;
	mAssertMsg(m_stripVertices.size() == static_cast<std::size_t>(outerLevel + innerLevel + 2u),
			   "Mismatch in vertex count");

	// Handle the various special cases upfront
	if(outerLevel == 0u) {
		// Only two vertices on the outside: left and right
		if(innerLevel == 1u) {
			// Only a single triangle
			const OpenMesh::FaceHandle newFace = m_mesh->add_face(m_stripVertices);
			if(!newFace.is_valid())
				throw std::runtime_error("Any: failed to add tessellated outer face (strip; outerLevel == 0u, innerLevel == 1)");
			this->set_triangle_face_outer(original, newFace);
		} else {
			// Connect until the distance to the other corner is smaller, then connect to other corner
			// Keep in mind that we have at least 2 inner vertices; we'll get innerLevel triangles
			// TODO: determine proper order: left to right or right to left? should depend on vertex proximity
			u32 i = 0u;
			while(i + 1u < innerLevel) {
				const float distSqLeft = ei::lensq(util::pun<ei::Vec3>(m_mesh->point(m_stripVertices[2u + i])) -
												   util::pun<ei::Vec3>(m_mesh->point(m_stripVertices[0u])));
				const float distSqRight = ei::lensq(util::pun<ei::Vec3>(m_mesh->point(m_stripVertices[2u + i + 1u])) -
													util::pun<ei::Vec3>(m_mesh->point(m_stripVertices[1u])));
				if(distSqLeft < distSqRight) {
					// Middle triangle connecting both outer vertices
					{
						const OpenMesh::FaceHandle newFace = m_mesh->add_face(m_stripVertices[0u], m_stripVertices[1u],
																			  m_stripVertices[2u + i]);
						if(!newFace.is_valid())
							throw std::runtime_error("Any: failed to add tessellated outer face (strip)");
						this->set_triangle_face_outer(original, newFace);
					}

					// Add the right-vertex triangles
					while(i + 1u < innerLevel) {
						const OpenMesh::FaceHandle newFace = m_mesh->add_face(m_stripVertices[2u + i + 1u], m_stripVertices[0u],
																			  m_stripVertices[2u + i]);
						if(!newFace.is_valid())
							throw std::runtime_error("Any: failed to add tessellated outer face (strip)");
						this->set_triangle_face_outer(original, newFace);
						++i;
					}
					return;
				}

				const OpenMesh::FaceHandle newFace = m_mesh->add_face(m_stripVertices[1u], m_stripVertices[2u + i],
																	  m_stripVertices[2u + i + 1u]);
				if(!newFace.is_valid())
					throw std::runtime_error("Any: failed to add tessellated outer face (strip; outerLevel == 0u, innerLevel != 1)");
				this->set_triangle_face_outer(original, newFace);
				++i;
			}

			// One last connection: all other inner vertices have been connected to the left outer vertex,
			// leaves the "middle" triangle
			const OpenMesh::FaceHandle newFace = m_mesh->add_face(m_stripVertices[0u], m_stripVertices[1u],
																  m_stripVertices[2u + i]);
			if(!newFace.is_valid())
				throw std::runtime_error("Any: failed to add tessellated outer face (strip)");
			this->set_triangle_face_outer(original, newFace);
		}
	} else if(innerLevel == 1u) {
		// Only one inner -> all edge points connect against it
		for(u32 i = 0u; i <= outerLevel; ++i) {
			const OpenMesh::FaceHandle newFace = m_mesh->add_face(m_stripVertices[i], m_stripVertices[i + 1u],
																  m_stripVertices.back());
			if(!newFace.is_valid())
				throw std::runtime_error("Any: failed to add tessellated outer face (strip; innerLevel == 1)");
			this->set_triangle_face_outer(original, newFace);
		}
	} else {
		// General case - need to determine lengths between multiple points
		u32 currInnerIndex = 0u;
		u32 currOuterIndex = 0u;
		while(currInnerIndex + 1u < innerLevel || currOuterIndex < outerLevel + 1u) {
			const OpenMesh::VertexHandle currInnerVertex = m_stripVertices[static_cast<u32>(m_stripVertices.size()) - currInnerIndex - 1u];
			const OpenMesh::VertexHandle currOuterVertex = m_stripVertices[currOuterIndex];
			OpenMesh::FaceHandle newFace;
			if(currInnerIndex + 1u >= innerLevel) {
				const OpenMesh::VertexHandle nextOuterVertex = m_stripVertices[currOuterIndex + 1u];
				newFace = m_mesh->add_face(currOuterVertex, nextOuterVertex, currInnerVertex);
				++currOuterIndex;
			} else if(currOuterIndex >= outerLevel + 1u) {
				const OpenMesh::VertexHandle nextInnerVertex = m_stripVertices[static_cast<u32>(m_stripVertices.size()) - currInnerIndex - 2u];
				newFace = m_mesh->add_face(currOuterVertex, nextInnerVertex, currInnerVertex);
				++currInnerIndex;
			} else {
				const OpenMesh::VertexHandle nextInnerVertex = m_stripVertices[static_cast<u32>(m_stripVertices.size()) - currInnerIndex - 2u];
				const OpenMesh::VertexHandle nextOuterVertex = m_stripVertices[currOuterIndex + 1u];
				const ei::Vec3 currInnerPos = util::pun<ei::Vec3>(m_mesh->point(currInnerVertex));
				const ei::Vec3 nextInnerPos = util::pun<ei::Vec3>(m_mesh->point(nextInnerVertex));
				const ei::Vec3 currOuterPos = util::pun<ei::Vec3>(m_mesh->point(currOuterVertex));
				const ei::Vec3 nextOuterPos = util::pun<ei::Vec3>(m_mesh->point(nextOuterVertex));
				// Check if we should finish the triangle by using the connection current inner to next outer
				// or vice versa
				float distSq0 = ei::lensq(currOuterPos - nextInnerPos);
				float distSq1 = ei::lensq(nextOuterPos - currInnerPos);

				if(distSq0 <= distSq1) {
					newFace = m_mesh->add_face(currOuterVertex, nextInnerVertex, currInnerVertex);
					++currInnerIndex;
				} else {
					newFace = m_mesh->add_face(currOuterVertex, nextOuterVertex, currInnerVertex);
					++currOuterIndex;
				}
			}

			if(!newFace.is_valid())
				throw std::runtime_error("Any: failed to add tessellated outer face (strip; general case)");
			this->set_triangle_face_outer(original, newFace);
		}
	}
}

OpenMesh::VertexHandle Tessellater::get_inner_vertex_triangle(const u32 edgeIndex, const u32 index,
															  const u32 innerLevel) const {
	// Returns the inner tessellated vertex with the given index at the given edge
	switch(edgeIndex) {
		case 0u: return m_innerVertices[index + innerLevel * 0u];
		case 1u: return m_innerVertices[(innerLevel - 1u) + innerLevel * index];
		case 2u: return m_innerVertices[(innerLevel - index - 1u) + innerLevel * (innerLevel - 1u)];
		default: mAssert(false); return OpenMesh::VertexHandle();
	}
}
OpenMesh::VertexHandle Tessellater::get_inner_vertex_quad(const u32 edgeIndex, const u32 index,
														  const u32 innerLevelX, const u32 innerLevelY) const {
	// Returns the inner tessellated vertex with the given index at the given edge
	switch(edgeIndex) {
		case 0u: return m_innerVertices[index + innerLevelX * 0u];
		case 1u: return m_innerVertices[(innerLevelX - 1u) + innerLevelX * index];
		case 2u: return m_innerVertices[(innerLevelX - index - 1u) + innerLevelX * (innerLevelY - 1u)];
		case 3u: return m_innerVertices[0u + (innerLevelY - index - 1u) * innerLevelX];
		default: mAssert(false); return OpenMesh::VertexHandle();
	}
}

} // namespace mufflon::scene::tessellation