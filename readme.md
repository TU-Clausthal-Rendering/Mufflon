Muflon\
A scientific renderer
---

Muflon is a research renderer of the computer graphics group of the Technical University Clausthal.
Its main purpose is to allow large scene GPU rendering.
Other research renderers like Mitsuba or PBRT lack the GPU support.
Frameworks like OptiX or Embree lack the freedom to bend the BVH/acceleration structure for our specific needs.
While they might be battle proven, the point with reasreach is to do new things.

General Design Goals
-
* Maintainability: a modularized architecture and certain guild lines (see doc/ folder) shall provide a high maintainability for a long lifetime
* Performance: we target performance over artist freedom. There is no need for a 20 layer skin model or other highly customisable effects. Therefore optimization may partially done on the cost of flexibility.
* CPU-GPU hybrid: Algorithms on both sides shall create the same results. If possible code should be shared directly.
* Debugging: Profiling and debugging options for the analyze of new algortihms