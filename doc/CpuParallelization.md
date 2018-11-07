How do we parallelize CPU sided renderers?
=

There are three options: using a thread-pool task library, implementing our own thread-pool or using OpenMP

Decision
-

We will use OpenMP for its simplicity.
An advantage of a self-made pool would be more control over the execution order of tasks.
However, it is also possible to rely on a linear sequence in parallel for loops. It is even possible
to remap the pixel coordinate (e.g. Hilbert curve) to control execution order.

Unknown issues
-

Will OpenMP(+Windows) be able to use the ThreadRipper CPU?