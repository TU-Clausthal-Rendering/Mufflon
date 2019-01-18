A9 Constant Materials
=

Materials cannot be changed or deleted.
This includes:

  * Any material properties (type, name, albedo, ...)
  * The assignment of materials per face

It is valid to change the material assignment (BinaryMat from faces to some other material)

Reason
-
This is a research renderer not a 3D content creation tool. Allowing runtime changes requires a lot of logic, state tracking, interfaces and expensive updates.
Instead we want to focus on algorithms without the burden of constantly changing content.