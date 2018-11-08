How many basic shapes should be supported? Should there more than only triangles?
The advantage of a renderer with pure triangle support is that there are no branches and a low thread divergence.
However, if one takes the step to different supported primitives the overhead for each additional one shrinks.

Decision
-
We support multiple types, because they all have different small advantages. A single one would rise the question if we realy want this feature, but in sum it seams to be a good idea which worth the overhead.

* Spheres: required by Feng, qualitative area lights (better NEE sampling)
* Triangles: just everything. Their vertices are in counter-clockwise order, ie. cross(v2-v0, v1-v0) gives you the surface normal.
* Quads: better interpolation of surface properties, better subdivision, less memory. Their vertices are to be in the following order:
	* First triangle: counter-clockwise, vertices 0, 1, and 2
	* Second triangle: counter-clockwise, vertices 0, 2, and 3. This implies that the ridge of a quad has to be the edge between vertex 0 and 2

Not supported:

* NURBS: +smooth surfaces, -very complex
* Implicit displacement mapped: +high detail, -tracer needs very long for a single triangle (iterative search). We will use (adaptive) tessellation from OpenMesh for displacement maps.

Consequences
-
1. Need branching in the BVH traversal
2. Need high level scene interface for multiple geometries.
   OpenMesh will be used for triangles and quads. Spheres must be held dedicated.
3. Loader must tessellate higher order polygons into tringles