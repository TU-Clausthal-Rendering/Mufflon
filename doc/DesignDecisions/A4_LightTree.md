There are two ways of storing light sources: either tightly packed or as a union with equal size each.
The former offers less memory consumption, but sacrifices random accessibility.
Additionally, our light tree has internal nodes, which may differ in size. The question here is the same: share a size with the light sources or not?
A third question regards the contents of lights: do they contain indices to vertices/triangles, or directly the positions?
Directly storing simplifies the interface and offers better cache performance, but sacrifices memory.

Decision
-
Light sources do not necessarily share sizes and are stored as distinct types. The same goes for light tree nodes.
Lights store their positions directly; depending on the underlying primitive for area lights as an array of points.
This grants us the lowest memory overhead for storing lights in a tree, while giving us better performance for light sampling.

Consequences
-
1. Light tree operates on a raw byte array (placement-new, reinterpret_cast)
2. Higher memory consumption for (tessellated) area lights