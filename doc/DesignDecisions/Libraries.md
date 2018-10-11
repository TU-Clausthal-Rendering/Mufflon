Math: Epsilon Intersection
-
Epsilon is designed for the specific application domain. It contains
Vector, Matrix, Quaternion, Intersection methods, (small) LU/Cholesky solvers.

RNG: Chi-Noise
-
Based on epsilon and designed for procedural content and sampling. Contains several RNGs and orthogonal samplers.
Also provides some fractal noise functions.

Scene: OpenMesh
-
OpenMesh has several useful mesh modification algorithms like edge-collapse and adaptive tessellation which can be used for:
adaptive out-of-core reductions and additions, displacement maps, textured area lights.
The drawback might be a lower tracing performance due to less cache friendly memory layout.
However, writing the collapse and tesselation algorithms on our own would also require a half-edge data structure and thus would suffer from the same problem.