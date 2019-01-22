A11 Adaptive Tessellation
=


To avoid needlessly detailed meshes in unimportant areas, we need an adaptive tessellation scheme following user guidance.
Problem: OpenMesh's subdividers do not take callbacks to support adaptiveness (or even proper UV/normal recalculation) and
partially do not even work for quads.

Decision
-
We will implement our own tessellator. To avoid holes, it will ask via callbacks for each edge how many new vertices it should generate.
The inner tessellation of each face is then implicitly determined and will equal the original face's type, i.e. either quad or triangle.
There should also be both options of automatically generating the new vertex positions and asking via a callback; these should
also be used to determine interpolated normals/UVs/other attributes.