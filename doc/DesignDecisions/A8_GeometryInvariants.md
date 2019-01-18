A8 Geometry Invariants
=

There are several confusing data structures or implicit assumptions used over multiple files which are all collected here.

Primitives are always sorted as: Triangles -> Quads -> Spheres
---

Primitives are indexed globally. An index >= #Tri means we have a quad or a sphere and index >= #Tri+#Quad must be a sphere.

The index buffer of triangles and quads has the same sorting (excluding the spheres which are not contained).
The main reason here is to be able to access any face given its index. Since Triangles and Quads have different data sizes (3*int and 4*int) the cannot be addressed in a single array, if they are arbitrary interleaved.

Object/Instance indices are different in the world and descriptors
---

There is masking.
If any instance (and its associated object) is masked it is not loaded in the descriptors. Since indices in the descriptors must be consecutive to avoid access violations or expensive mappings the indices are necessarily different.

Scene::m_objects is a map to resolve this problem on descriptor creation time. The order of object indices is defined by the iterator of this map. The instance indices are defined by iterating over the map and then iterating over the instances.
This leads to an additional invariant: instances of the same object are always in a single consecutive range.
The same order is and must be used by the world::load_lights method, because the created primitiveIds depend on this indexing.