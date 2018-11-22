A7 PathVertex Layout
=

The PathVertex is a wrapper class to unify the view to different interaction events of a path (camera, light, surface, ...).
This abstraction is complicated for two reason:

  1. Different renderer require different additional information
  2. Each interaction requires different information

For the functional aspect, polymorphic classes would be nice.
However, the vertex type also has to work on GPU (no virtual functions) and we want to be able to store the vertices as local as possible (registers, shared mem).
This means, given some generic memory, we want to store vertices of a path densely packed into this memory.

Dense Packing
-

We favor dense packing of vertices, because for recursive materials a single vertex can become large. Using the maximum size of a vertex to pack vertices into an array would cause a lot of wasted memory.

Using less memory bears the danger of out-of-memory for paths with many very complex interactions.
The consequence is that paths with complex interactions might be shorter than the expected maximum, which an error we accept.

**How to iterate through a path?**

The VertexFactory helps to determine the size of a given vertex.
In forward direction a path can be iterated by adding the vertex size to the current head pointer.
To iterate paths backwards (which is more important for MIS computations), vertices store an offset **relative to the beginning of the path**.
This can be used to to compute the address of the previous vertex by computing the pathMemory+offset pointer.

Read Only
-

By design, vertices are read only.
In some renderers (e.g. VCM) the vertices are accessed in parallel and in others (BPT) the are accessed multiple times sequentially.
Anyway, it is convenient to rely on the untouched state of vertices.

Per Renderer Abstraction
-

To solve the problem of per renderer additional content, a simple template member is sufficient.
The maintenance is up to the renderer and read/write access is explicitly allowed.
Although the vertex itself is read-only a renderer might want to use changeable attributes (e.g. usage statistics of a vertex).
Further, write access is probably necessary for the maintenance on creation.

Per Interaction Abstraction
-

The interaction type changes on runtime for each vertex of a path.
This causes an unknown dynamic size of vertices.
To be able to still use vertices in user defined memory we need to use pointer magic which is done by the vertex internals and a vertex factory.

This pointer magic causes a problem: the vertices are incompatible with register memory.
It is possible to design multiple vertex types to circumvent this problem as long as the type is known.
However, the entire idea of vertices is to NOT know the type (it shall be a unified view to an arbitrary interaction).
Also, the dense packing and the iteration through paths require pointers.
Therefore, vertices are **not storable in register memory**.

If possible try to hold the pure information (Photon, EvalValue, ...) on stack/register as long as possible.
Only convert information to PathVertices for reasons of storing and unified access.
If there is a decision between *maximum unity* and *locality of special data*, prefer the special date for reasons of performance.
Sometimes a more unified view would clean up the code, but the cost can be high.