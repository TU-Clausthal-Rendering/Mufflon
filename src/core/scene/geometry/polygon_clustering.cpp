#include "polygon.hpp"

namespace mufflon::scene::geometry {

std::tuple<unique_device_ptr<Device::CPU, char[]>, unique_device_ptr<Device::CPU, char[]>,
	unique_device_ptr<Device::CPU, u32[]>,
	unsigned, unsigned, unsigned>
	cluster_uniformly_cuda(unique_device_ptr<Device::CPU, char[]> vertexAttributes,
						   unique_device_ptr<Device::CPU, char[]> faceAttributes,
						   unique_device_ptr<Device::CPU, u32[]> indices, const unsigned vertexAttribSize,
						   const unsigned faceAttribSize, const unsigned vertexCount,
						   const unsigned triangleCount, const unsigned quadCount,
						   const ei::Box& aabb, const ei::UVec3& gridRes,
						   const unsigned* vertAttribElemSizes,
						   const unsigned* faceAttribElemSizes,
						   const std::size_t vertAttribCount,
						   const std::size_t faceAttribCount);

union EdgeOrGrading {
	struct Edge {
		u32 first;
		u32 second;
	} edge;
	struct VertexGrading {
		u32 vertexIndex;
		union GradeOrGridIndex {
			float grade;
			u32 gridIndex;
		} data;
	} grading;

	float get_grade() const noexcept { return grading.data.grade; }
	u32 get_grid_index() const noexcept { return grading.data.gridIndex; }
	void set_grid_index(const u32 gridIndex) noexcept { grading.data.gridIndex = gridIndex; }
};
struct Cluster {
	ei::Vec3 sum;
	u32 reprVertIdx;
	u32 count;
	u32 id;
};
struct Grid {
	std::optional<Cluster> cluster;
};

void get_sorted_edges_all(const u32* indices, const std::size_t triangleCount,
						  const std::size_t quadCount, EdgeOrGrading* edges) {
	auto* currEdge = edges;
	for(std::size_t i = 0u; i < 3u * triangleCount; i += 3u) {
		(currEdge++)->edge = { indices[i + 0u], indices[i + 1u] };
		(currEdge++)->edge = { indices[i + 1u], indices[i + 2u] };
		(currEdge++)->edge = { indices[i + 2u], indices[i + 0u] };
		(currEdge++)->edge = { indices[i + 0u], indices[i + 2u] };
		(currEdge++)->edge = { indices[i + 2u], indices[i + 1u] };
		(currEdge++)->edge = { indices[i + 1u], indices[i + 0u] };
	}
	for(std::size_t i = 0u; i < 4u * quadCount; i += 4u) {
		(currEdge++)->edge = { indices[i + 0u], indices[i + 1u] };
		(currEdge++)->edge = { indices[i + 1u], indices[i + 2u] };
		(currEdge++)->edge = { indices[i + 2u], indices[i + 3u] };
		(currEdge++)->edge = { indices[i + 3u], indices[i + 0u] };
		(currEdge++)->edge = { indices[i + 0u], indices[i + 3u] };
		(currEdge++)->edge = { indices[i + 3u], indices[i + 2u] };
		(currEdge++)->edge = { indices[i + 2u], indices[i + 1u] };
		(currEdge++)->edge = { indices[i + 1u], indices[i + 0u] };
	}

	// Sort them by vertex
	std::sort(edges, currEdge, [](const auto& left, const auto& right) { return left.edge.first < right.edge.first; });
}

void compute_vertex_grading(EdgeOrGrading* edges, const ei::Vec3* points, const std::size_t edgeCount,
							const std::size_t vertexCount) {
	std::size_t currIndex = 0u;
	while(currIndex < edgeCount) {
		auto currVertex = edges[currIndex].edge.first;
		float minCos = 1.f;

		// Find the end-index of edges
		std::size_t lastEdge;
		for(lastEdge = currIndex; lastEdge < edgeCount && edges[lastEdge].edge.first == currVertex; ++lastEdge) {}
		// Find the pairwise max. angle (== min. cosine) between the edges
		const auto v0 = points[currVertex];
		for(std::size_t e1 = currIndex; (e1 + 1u) < lastEdge; ++e1) {
			const auto v1 = points[edges[e1].edge.second];
			const auto v0v1 = ei::normalize(v1 - v0);
			for(std::size_t e2 = e1 + 1u; e2 < lastEdge; ++e2) {
				const auto v2 = points[edges[e2].edge.second];
				const auto v0v2 = ei::normalize(v2 - v0);
				const auto cosTheta = ei::dot(v0v1, v0v2);
				minCos = std::min(minCos, cosTheta);
			}
		}

		// Compute and store the grade
		// We simply start overwriting the buffer from the beginning since we're done with the edges
		edges[currVertex].grading = { currVertex, std::cos(std::acos(minCos) / 2.f) };
		currIndex = lastEdge;
	}
	// Sort the vertices by grade (descending)
	std::sort(edges, edges + vertexCount, [](const auto& lhs, const auto& rhs) { return lhs.get_grade() > rhs.get_grade(); });
}

void check_neighboring_grid_cells(const Grid* grids, const ei::Vec3* points, const u32 centreGridIdx,
								  const ei::UVec3& doubleResGridPos, const ei::UVec3& gridRes,
								  const ei::Vec3& currVertPos, std::pair<std::optional<u32>, float>& chosenGrid) {
	// The other grid indices we get by adding/subtracting one if the double grid position is odd/even
	const auto compareGrid = [&grids, &chosenGrid, points, centreGridIdx, currVertPos](const i32 indexOffset) {
		const auto currGridIndex = centreGridIdx + indexOffset;
		// Check if the grid cell even has a cluster associated with it
		if(grids[currGridIndex].cluster.has_value()) {
			// We compare the distance between vertex and cluster instead of using cluster weight
			const auto distance = ei::len(currVertPos - points[grids[centreGridIdx].cluster->reprVertIdx]);
			if(distance < chosenGrid.second)
				chosenGrid = { currGridIndex, distance };
		}
	};

	// Iterate all neighbor cells
	const ei::UVec3 neighborOffset{ 1u, gridRes.x, gridRes.x * gridRes.y };
	for(unsigned i = 1u; i < 8u; ++i) {
		// Loop over the three dimensions
		for(unsigned j = 0u; j < 3u; ++j) {
			// Interpret 'i' as the bitmask determining the dimension, so 1 == 001 == x only, 5 == 101 == x and z etc.
			if((i & (1u << j)) != 0u) {
				// Depending on which side of the cell we are we either check the "left" or "right" neighbor
				// Also make sure that we don't run out of the grid (if we do, there is no neighbor for this dimension)
				if((doubleResGridPos[j] & 1u) == 0u) {
					if(doubleResGridPos[j] > 0u)
						compareGrid(-static_cast<i32>(neighborOffset[j]));
				} else if(doubleResGridPos[j] < gridRes[j] * 2u - 1u) {
					compareGrid(static_cast<i32>(neighborOffset[j]));
				}
			}
		}
	}
}

u32 compute_grid_centres_and_vertex_association(EdgeOrGrading* vertexGrade, Grid* grids,
												const ei::Vec3* points, const ei::UVec3& gridRes,
												const ei::Box& aabb, const std::size_t vertexCount) {
	// Clustering step: first create the grid
	// For this we discretize the bounding box in equal parts
	u32 clusterCount = 0u;
	const auto aabbDiag = aabb.max - aabb.min;
	const auto get_grid_index = [gridRes](const ei::UVec3& gridPos) {
		return gridPos.x + gridPos.y * gridRes.x + gridPos.z * gridRes.y * gridRes.x;
	};
	for(std::size_t v = 0u; v < vertexCount; ++v) {
		auto& vertex = vertexGrade[v];
		const auto pos = points[vertex.grading.vertexIndex];
		// Check if the vertex is part of a cell already
		// For this, first find out the grid it belongs to and its neighbors
		std::pair<std::optional<u32>, float> chosenGrid{ std::nullopt, std::numeric_limits<float>::max() };

		// Get the normalized position [0, 1]^3
		const auto normPos = (pos - aabb.min) / aabbDiag;
		// Get the discretized grid position with twice the grid resolution
		const auto doubleResGridPos = ei::min(ei::UVec3{ normPos * ei::Vec3{ 2u * gridRes } }, 2u * gridRes - 1u);
		const auto centreGridIdx = get_grid_index(doubleResGridPos / 2u);
		if(grids[centreGridIdx].cluster.has_value())
			chosenGrid = { centreGridIdx, ei::len(pos - points[grids[centreGridIdx].cluster->reprVertIdx]) };

		// Check the closest surrounding neighbor cells whether their centre is closer to our vertex
		check_neighboring_grid_cells(grids, points, centreGridIdx, doubleResGridPos,
									 gridRes, pos, chosenGrid);

		// If vertex doesn't fit into any cell, create the cell with the vertex as centre
		if(!chosenGrid.first.has_value()) {
			grids[centreGridIdx].cluster = Cluster{ ei::Vec3{0.f}, vertex.grading.vertexIndex, 0u, clusterCount++ };
			chosenGrid.first = centreGridIdx;
		}
		const auto gridIdx = chosenGrid.first.value();
		grids[gridIdx].cluster->count += 1u;
		grids[gridIdx].cluster->sum += pos;

		// We set the grid index for the vertex - but after this loop, the array will be indexable by vertex index
		// again since we don't need the sorting by grade anymore!
		vertexGrade[vertex.grading.vertexIndex].set_grid_index(gridIdx);
	}

	return clusterCount;
}

std::size_t mark_degenerate_triangles(u32* indices, const EdgeOrGrading* vertexGrade,
									  const std::size_t triangleCount) {
	std::size_t triCount = triangleCount;
	for(auto* tri = indices; tri < indices + 3u * triangleCount; tri += 3u) {
		// Check if all three vertices are in different cells, otherwise mark the triangle as invalid
		if(vertexGrade[tri[0u]].get_grid_index() == vertexGrade[tri[1u]].get_grid_index()
		   || vertexGrade[tri[0u]].get_grid_index() == vertexGrade[tri[2u]].get_grid_index()
		   || vertexGrade[tri[1u]].get_grid_index() == vertexGrade[tri[2u]].get_grid_index()) {
			tri[2u] = std::numeric_limits<u32>::max();
			triCount -= 1u;
		}
	}

	return triCount;
}

std::pair<std::size_t, std::size_t> mark_degenerate_quads(u32* indices, const EdgeOrGrading* vertexGrade,
														  const std::size_t origQuadCount) {
	std::size_t quadCount = origQuadCount;
	std::size_t newTriCount = 0u;
	for(auto* quad = indices; quad < indices + 4u * origQuadCount; quad += 4u) {
		// Quads can degenerate into triangles
		ei::UVec4 quadVertIndices{
			vertexGrade[quad[0u]].get_grid_index(), vertexGrade[quad[1u]].get_grid_index(),
			vertexGrade[quad[2u]].get_grid_index(), vertexGrade[quad[3u]].get_grid_index()
		};
		// Check for tri
		// For triangles, we mark the outlying vertex, for edges/vertices we mark first and second
		if(quadVertIndices.x != quadVertIndices.y) {
			// At least edge
			if(quadVertIndices.x != quadVertIndices.z) {
				// At least triangle
				if(quadVertIndices.x == quadVertIndices.w) {
					// Mark as triangle
					quad[3u] = std::numeric_limits<u32>::max();
					quadCount -= 1u;
					newTriCount += 1u;
				}
			} else {
				// Pre-mark as triangle
				quadCount -= 1u;
				// Check if it is actually triangle, mark as removal otherwise
				if(quadVertIndices.x == quadVertIndices.w) {
					quad[0u] = std::numeric_limits<u32>::max();
					quad[1u] = std::numeric_limits<u32>::max();
				} else {
					quad[2u] = std::numeric_limits<u32>::max();
					newTriCount += 1u;
				}
			}
		} else {
			// No longer quad, but maybe triangle
			quadCount -= 1u;

			if(quadVertIndices.x != quadVertIndices.z) {
				quad[1u] = std::numeric_limits<u32>::max();
				if(quadVertIndices.x == quadVertIndices.w)
					quad[0u] = std::numeric_limits<u32>::max();
				else
					newTriCount += 1u;
			} else {
				quad[0u] = std::numeric_limits<u32>::max();
				quad[1u] = std::numeric_limits<u32>::max();
			}
		}
	}

	return std::make_pair(quadCount, newTriCount);
}

void remove_marked_triangles(u32* indices, Polygons::FaceAttributePoolType& faceAttribs,
							 const std::size_t triangleCount, const std::size_t remTriCount) {
	const auto skip_marked_from_back = [indices](std::size_t currEndTri) -> std::size_t {
		while(currEndTri > 0u && indices[3u * currEndTri - 1u] == std::numeric_limits<u32>::max()) { currEndTri -= 1u; }
		return currEndTri;
	};

	auto currEndTri = skip_marked_from_back(triangleCount);
	for(std::size_t i = 0u; i < remTriCount; ++i) {
		if(indices[3u * i + 2u] == std::numeric_limits<u32>::max()) {
			// Swap last triangle
			currEndTri -= 1u;
			std::memcpy(indices + 3u * i, indices + 3u * currEndTri, 3u * sizeof(u32));
			faceAttribs.copy(currEndTri, i);
			currEndTri = skip_marked_from_back(currEndTri);
		}
	}
}

void remove_or_transform_marked_quads(u32* indices, Polygons::FaceAttributePoolType& faceAttribs,
									  const std::size_t origTriCount, const std::size_t remTriCount,
									  const std::size_t newTriCount, const std::size_t quadCount,
									  const std::size_t remQuadCount) {
#if 0
	auto* quadIndices = indices + 3u * origTriCount;
	auto currTri = remTriCount;

	// Removing the quads is a lot more involved than triangles, because quads can turn into triangles
	// and then need to be swapped to the beginning of quads/end of triangles. This may lead
	// to non-continuity in the index buffer, because triangles need one less index.

	// Track if there are emtpy triangle slots left by removed triangles
	auto remEmptyTriSlots = origTriCount - remTriCount;

	// We split this process into two: first we swap out all deleted quads
	const auto skip_deleted_from_back = [quadIndices, &currTri](std::size_t currEndQuad) {
		while(currEndQuad > 0u && quadIndices[4u * currEndQuad] == std::numeric_limits<u32>::max()
			  && quadIndices[4u * currEndQuad + 1u] == std::numeric_limits<u32>::max()) {
			currEndQuad -= 1u;
		}
		return currEndQuad;
	};

	auto currEndQuad = skip_deleted_from_back(quadCount);
	auto* currQuad = quadIndices;
	for(std::size_t i = 0u; i < remQuadCount + newTriCount; ++i, currQuad += 4u) {
		if(currQuad[0u] == std::numeric_limits<u32>::max() && currQuad[1u] == std::numeric_limits<u32>::max()) {
			// Deleted quad - fill in quad from the end
			currEndQuad -= 1u;
			std::memcpy(currQuad, quadIndices + 4u * currEndQuad, 4u * sizeof(u32));
			faceAttribs.copy(currEndQuad + origTriCount, i + origTriCount);
			currEndQuad = skip_deleted_from_back(currEndQuad);
		}
	}

	// Now we only have quads-turned-triangles left
	// For these we can first use the space left by removed triangles
	currQuad = quadIndices;
	for(std::size_t i = 0u; i < remQuadCount + newTriCount; ++i, currQuad += 4u) {
		// Check if any of the indices is marked
		for(std::size_t j = 0u; j < 4u; ++j) {
			if(currQuad[j] == std::numeric_limits<u32>::max()) {
				// Check if there are empty tri-slots
				if(remEmptyTriSlots > 0u) {

				}
			}
		}
	}


	const auto skip_marked_from_back = [indices, quadIndices, origTriCount, &currTri, &faceAttribs](std::size_t currEndQuad) {
		while(currEndQuad > 0u) {
			if(quadIndices[4u * currEndQuad] == std::numeric_limits<u32>::max()) {
				if(quadIndices[4u * currEndQuad + 1u] != std::numeric_limits<u32>::max()) {
					// Degenerated - copy to triangle list
					std::memcpy(indices + 3u * currTri, quadIndices + 4u * currEndQuad + 1u, 3u * sizeof(u32));
					faceAttribs.copy(currEndQuad + origTriCount, currTri);
					currTri += 1u;
				}
				currEndQuad -= 1u;
			} else {
				// Check other indices for degeneration
				bool degenerated = false;
				for(std::size_t j = 1u; j < 4u && !degenerated; ++j) {
					if(quadIndices[4u * currEndQuad + j] == std::numeric_limits<u32>::max()) {
						// Degenerate quad - put at last triangle spot
						auto* currTriIndex = indices + 3u * currTri;
						for(std::size_t k = 0u; k < 4u; ++k) {
							if(k != j)
								*(currTriIndex++) = quadIndices[4u * currEndQuad + k];
						}
						faceAttribs.copy(currEndQuad + origTriCount, currTri);
						currTri += 1u;
						degenerated = true;
					}
				}
				if(degenerated)
					currEndQuad -= 1u;
				else
					break;
			}
		}
		return currEndQuad;
	};

	auto currEndQuad = skip_marked_from_back(quadCount);
	auto* currQuad = quadIndices;
	for(std::size_t i = 0u; i < remQuadCount; ++i, currQuad += 4u) {
		if(currQuad[0u] == std::numeric_limits<u32>::max()) {
			if(currQuad[1u] != std::numeric_limits<u32>::max()) {
				// Degenerate quad - put at last triangle spot
				std::memcpy(indices + 3u * currTri, currQuad + 1u, 3u * sizeof(u32));
				faceAttribs.copy(i + origTriCount, currTri);
				currTri += 1u;
			}
			// Either deleted or degenerate quad - fill in quad from the end
			currEndQuad -= 1u;
			std::memcpy(currQuad, quadIndices + 4u * currEndQuad, 4u * sizeof(u32));
			faceAttribs.copy(currEndQuad + origTriCount, i + origTriCount);
			currEndQuad = skip_marked_from_back(currEndQuad);
		} else {
			// No deleted quad possible anymore, only degenerate
			for(std::size_t j = 1u; j < 4u; ++j) {
				if(currQuad[j] == std::numeric_limits<u32>::max()) {
					// Degenerate quad - put at last triangle spot
					auto* currTriIndex = indices + 3u * currTri;
					for(std::size_t k = 0u; k < 4u; ++k) {
						if(k != j)
							*(currTriIndex++) = currQuad[k];
					}
					faceAttribs.copy(i + origTriCount, currTri);
					currTri += 1u;
					// Fill in hole
					currEndQuad -= 1u;
					std::memcpy(currQuad, quadIndices + 4u * currEndQuad, 4u * sizeof(u32));
					faceAttribs.copy(currEndQuad + origTriCount, i + origTriCount);
					currEndQuad = skip_marked_from_back(currEndQuad);

					// No other indices matter now
					break;
				}
			}
		}
	}
#endif // 0
}

void Polygons::cluster_uniformly(const ei::UVec3& gridRes) {
	using namespace std::chrono;
	const auto t0 = high_resolution_clock::now();
	this->template unload_index_buffer<Device::CUDA>();
	this->template unload_index_buffer<Device::OPENGL>();
	auto indices = std::move(m_indexBuffer.template get<IndexBuffer<Device::CPU>>().indices);
	auto vertData = m_vertexAttributes.release_pool_memory<Device::CPU>();
	auto faceData = m_faceAttributes.release_pool_memory<Device::CPU>();
	const auto vertBytes = m_vertexAttributes.get_pool_size();
	const auto faceBytes = m_faceAttributes.get_pool_size();
	const auto vertices = static_cast<unsigned>(this->get_vertex_count());
	const auto triangles = static_cast<unsigned>(this->get_triangle_count());
	const auto quads = static_cast<unsigned>(this->get_quad_count());

	const auto vertAttribSizes = m_vertexAttributes.get_attribute_element_sizes();
	const auto faceAttribSizes = m_faceAttributes.get_attribute_element_sizes();

	auto [newVertAttribs, newFaceAttribs, newIndices, remVerts, remTris, remQuads] =
		cluster_uniformly_cuda(std::move(vertData), std::move(faceData), std::move(indices),
							   static_cast<unsigned>(vertBytes / vertices),
							   static_cast<unsigned>(faceBytes / (triangles + quads)), vertices,
							   triangles, quads, m_boundingBox, gridRes,
							   vertAttribSizes.data(), faceAttribSizes.data(),
							   vertAttribSizes.size(), faceAttribSizes.size());


	m_triangles = remTris;
	m_quads = remQuads;
	m_vertexAttributes.replace_pool_memory<Device::CPU>(std::move(newVertAttribs), remVerts);
	m_faceAttributes.replace_pool_memory<Device::CPU>(std::move(newFaceAttribs), remTris + remQuads);
	m_indexBuffer.template get<IndexBuffer<Device::CPU>>().indices = std::move(newIndices);
	m_indexBuffer.template get<IndexBuffer<Device::CPU>>().reserved = 3u * remTris + 4u * remQuads;

	const auto t1 = high_resolution_clock::now();
	logWarning(duration_cast<milliseconds>(t1 - t0).count(), "ms");
}

#if 0
void Polygons::cluster_uniformly(const ei::UVec3& gridRes) {
	// Idea taken from here: https://www.comp.nus.edu.sg/~tants/Paper/simplify.pdf
	this->template synchronize<Device::CPU>();
	const auto* points = this->template acquire_const<Device::CPU, ei::Vec3>(this->get_points_hdl());
	auto* indices = m_indexBuffer.template get<IndexBuffer<Device::CPU>>().indices.get();

	const auto triangleCount = this->get_triangle_count();
	const auto quadCount = this->get_quad_count();
	const auto vertexCount = this->get_vertex_count();
	const auto edgeCount = triangleCount * 6u + quadCount * 8u;

	std::vector<Grid> grids(ei::prod(gridRes));
	// Pre-allocate buffer: memory bound is the edges, so 6 times triangle count
	std::vector<EdgeOrGrading> buffer(edgeCount);

	// Get all edges of each vertex
	get_sorted_edges_all(indices, triangleCount, quadCount, buffer.data());
	// Now we can compute the "grading" of each vertex
	compute_vertex_grading(buffer.data(), points, edgeCount, vertexCount);

	// Get the grids and vertex cluster association
	const auto clusterCount = compute_grid_centres_and_vertex_association(buffer.data(), grids.data(), points,
																		  gridRes, m_boundingBox, vertexCount);

	// Now we have to delete triangles which have degenerated into points/lines
	// Quads can also be transformed into triangles
	const auto remTriCount = mark_degenerate_triangles(indices, buffer.data(), triangleCount);
	const auto [remQuadCount, newTriCount] = mark_degenerate_quads(indices + 3u * triangleCount, buffer.data(), quadCount);
	// Now we have to iterate the index buffer to swap/remove all remaining faces
	remove_marked_triangles(indices, m_faceAttributes, triangleCount, remTriCount);
	remove_or_transform_marked_quads(indices, m_faceAttributes, triangleCount, remTriCount, quadCount, remQuadCount);

	// Set the new face counts
	m_triangles = remTriCount + newTriCount;
	m_quads = remQuadCount;

	// Set the new vertex indices for the triangles
	for(std::size_t i = 0u; i < 3u * m_triangles + 4u * m_quads; ++i)
		indices[i] = grids[buffer[indices[i]].get_grid_index()].cluster->id;

	// Now create and copy over the vertex attributes to the new vertices
	auto newVertAttribs = m_vertexAttributes.create_with_attributes(clusterCount);
	auto* newPoints = newVertAttribs.template acquire<Device::CPU, ei::Vec3>(this->get_points_hdl());
	for(const auto& grid : grids) {
		if(grid.cluster.has_value()) {
			const auto newIndex = grid.cluster->id;
			// Copy over the attributes
			newVertAttribs.copy(m_vertexAttributes, grid.cluster->reprVertIdx, newIndex);
			// Compute new position as average
			newPoints[newIndex] =  grid.cluster->sum / static_cast<float>(grid.cluster->count);
		}
	}
	m_vertexAttributes = std::move(newVertAttribs);
	m_faceAttributes.resize(m_triangles + m_quads);
	m_faceAttributes.shrink_to_fit();
	
	// Resize the buffer
	auto& indexBuffer = m_indexBuffer.template get<IndexBuffer<Device::CPU>>();
	indexBuffer.indices.reset(Allocator<Device::CPU>::realloc(indexBuffer.indices.release(), indexBuffer.reserved,
															  3u * m_triangles + 4u * m_quads));

	this->mark_changed(Device::CPU);
}
#endif // 0

} // namespace mufflon::scene::geometry