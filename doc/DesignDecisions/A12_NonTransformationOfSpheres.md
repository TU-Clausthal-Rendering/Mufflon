A12 (Non) Transformation of spheres
=

Spheres may only be translated and uniformly scaled.

* They should **not** be scaled non-uniformly.
* They should **not** be rotated.

Reason
-

Non-uniform scaling breaks the NEE sampler for spheres. It samples the visible part of a sphere and
not that of an ellipsoids.

Rotations cause several problems with texturing. A non-textured sphere does not need to be rotated (invariant anyway).
However, if the sphere is textured+rotated, it is necessary that area lights, alpha testing and texture mapping know about this rotation.
Since there can be multiple spheres in one object, an additional rotation matrix per sphere would be necessary for correct support.
This is considered to be too expensive and to seldomly used.

Consequences
-

Handling of sphere primitives is simplified and can be made faster.
Therefore, certain scene configurations become invalid.

Alternatives
-

Mesh your sphere! If you need a textured + rotated sphere or an ellipsoid use a mesh instead.