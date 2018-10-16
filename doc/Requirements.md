
* CPU/GPU
  - CPU multithreaded (with switch for deterministic behavior)
  - shared code with CPU if possible
  - If existent, the same algorithm CPU/GPU should produce the same result
  - CPU execution without context
* BVH
  - build on load
    - LLBVH, SBVH?
    - store as additional file for debugging/cached load
  - optional load from file: no
  - triangles, quads and spheres
  - [Opt] extensions: NURBS or similar? No work for now. Potentielly breaking the scene interfaces - but we take that.
* Renderes
  - PT
  - BPT
  - VCM
  - SPPM mit MIS
  - Realtime preview
  - BVH-debug
  - [Opt] Volumetric transport
* Random Number Generation
  - fixed number per event: 2dir, 1layer
  - option to fix the seed (for debugging), fix sequences for any work package
  - thread dispatch with deterministic assignments (same results for same seed)? - Not necessary
* Scene
  - Instancing
  - forced de-instancing
  - LOD or partial loading
  - object masking (loader, [Opt] at runtime)
  - animated meshes?
    - Load all instances of an animation, react with special intersection cernels to handle these events. Otherwise use masking to avoid the multiple objects.
  - Per triangle tangent spaces
  - [Opt] displacement mapping, use tesselation
  - (adaptive) subdivision (using OpenMesh)
  - spheres as extra geometry
  - dynamic vertex attributes
* GUI
  - [Opt] Performance and debugging number output
  - Script command line
  - Per-pixel information
  - Moveable camera
    - path recording for animation rendering
* Command-Script
  - sets all options and runs repeatable experiments
* File formats
  - own
    - readable materials
    - readable lights
    - readable camera
    - binary meshes
  - others?
  - simulation output (particle mixtures)
*  Image formats
  - Textures: hdr, pfm, ktx, png, jpg, tga
  - Output: hdr, pfm
  - multitexture files
* Post processing
  - [Opt] denoise
  - tonemapping (at least +- brightness)
  - arbitrary shaders
* Materials
  - Lambert
  - Torrance-Sparrow
  - Walter
  - Emissive
  - Mixed
  - [Opt] Oren-Nayar
* Lights
  - Point
  - Directional
  - Envmap
  - Area (material basis -> sphere, triangle, quads, [Opt] textured)
  - Spot (PBRT type)
  - [Opt] Textured point light
  - Hierarchy for sampling and realtime renderer
* Cameras
  - Pinhole
  - Focus
* [Opt] Realtime shader editing (does not apply to most renderers because of CUDA)