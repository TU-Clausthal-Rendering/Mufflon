Math: Epsilon Intersection
-
Epsilon is designed for the specific application domain. It contains
Vector, Matrix, Quaternion, Intersection methods, (small) LU/Cholesky solvers.

RNG: Chi-Noise
-
Based on epsilon and designed for procedural content and sampling. Contains several RNGs and orthogonal samplers.
Also provides some fractal noise functions.

TODO: http://www.pcg-random.org/ and CuRand checken.

Scene: OpenMesh
-
OpenMesh has several useful mesh modification algorithms like edge-collapse and adaptive tessellation which can be used for:
adaptive out-of-core reductions and additions, displacement maps, textured area lights.
The drawback might be a lower tracing performance due to less cache friendly memory layout.
However, writing the collapse and tesselation algorithms on our own would also require a half-edge data structure and thus would suffer from the same problem.

Graphics: CUDA, OpenGL
-
CUDA has numerous performance advantages (async upload) and code sharing host/device are conformant with our design goals.
OpenGL is still inevitable for the realtime debugging renderers and the previews.

Parsergenerators: ?
-
For the loaders.

JSON: ?
-
For the loaders

Image loading: STBI, GLI
-

GUI
-
Wenzel Jakobs library?