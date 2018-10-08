
* CPU/GPU
  - CPU multithreaded (with switch)
  - shared code with CPU if possible
  - If existent, the same algorithm CPU/GPU should produce the same result
  - CPU execution without context?
* BVH
  - build on load?
  - load from file?
  - triangles and spheres?
* Renderes
  - PT
  - BPT
  - VCM
  - Realtime preview
  - [Opt] BVH-debug
  - [Opt] Volumetric transport
* Random Number Generation
  - fixed number per event
  - option to fix the seed (for debugging) - only in non-threaded execution?
* Scene
  - Instancing
  - forced de-instancing?
  - LOD or partial loading
  - object masking
  - animated meshes? (skinning, rigid body sim)
  - Per triangle tangent spaces?
  - displacement mapping?
  - (adaptive) subdivision?
* GUI
  - [Opt] Performance and debugging number output
  - Script command line
  - [Opt] Per-pixel information
  - Moveable camera
    - path recording for animation rendering
* Command-Script
  - sets all options and runs repeatable experiments
* File formats
  - own
    - readable materials
    - readable lights
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
  - Area (triangle, sphere?)
  - Spot?
  - Hierarchy for sampling and realtime renderer
* Cameras
  - Pinhole
  - [Opt] Focus
* Realtime shader editing