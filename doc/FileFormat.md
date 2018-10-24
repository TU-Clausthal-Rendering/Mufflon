Mufflon File Format
=

Scene descriptions in Mufflon consist of two files: a json file with editable properties and a binary file for the mesh data.

---

The JSON properties file
-

The order of the blocks in the following template can be freely changed.
However, all information are mandatory if not flagged as optional.
For optional attributes, the default value is given next to the flag.
Text which is set like `"<stuff>"` can be chosen by the user (i.e. suff can be any name of the user choice).
Curly brackets are used to declare a choice of different possible strings (e.g. `"{pinhole, focus}"` must be either "pinhole" or "focus").
In the case of multiple type choices, details on further mandatory properties will be given below.
The syntax `"[stuff:name]"` is used to make a link to the binary file information.
The names must match the names stored in the binary itself.

    {
        "version": "1.0",
        "binary": "<file name relative to this json>"
        "cameras": {
            "<name1>": {
                "type": "{pinhole, focus, ortho}",
                "path": [[x1,y1,z1], ...],    // List of vec3 (at least 1)
                "viewDir": [[x1,y1,z1], ...], // List of vec3 with the same length as "path", not necessarily normalized
                "up": [[x1,y1,z1], ...],      // OPTIONAL [0,1,0], must have the same length as "path" if defined, not necessarily normalized
                ...
            },
            "<name2>": {
                ...
            } ...
        },
        "lights": {
            "<name1>": {
                "type": "{point, directional, spot, envmap, goniometric}",
                ...
            } ...
        },
        "materials": {
            "<name1>": {
                "type": "{lambert, torrance, walter, emissive, orennayar, blend, fresnel, glass, opaque}",
                ...
            } ...
        },
        "scenarios": {
            "<name1>: {
                "camera": "<name of a camera>",
                "lights": ["light:name1", ...]  // List of light sources
                "lod": int,             // Global level of detail number [0,...] where 0 has the highest resolution, OPTIONAL 0
                "[mat:name1]": "<name of a material>",
                "[mat:name2]": "<name of a material>",
                ....                    // Each material in the binary must be mapped to one of the above materials.
                                        // A material can be used by multiple binray-materials.
                "[obj:name]": {         // OPTIONAL per object properties
                    "mask",             // Do not render this object (blacklisted)
                    "lod": int,         // Use a specific LOD different/independent from global LOD
                    // More meta information
                }
            }
            ...
        }
    }

Cameras
--

Lights
--

This section lists the required attributes for the different types of light sources

`"type": "point"`

    "position": [x,y,z],            // vec3 vec3 world space position
    "flux" or "intensity: [a,b,c],  // Exclusive (either flux [W] or intensity [W/sr] must be specified as vec3)
    "scale": float,                 // Multiplier for "flux"/"intensity"

`"type": "directional"`

    "direction": [x,y,z],           // Direction in which the light travels (incident direction)
    "radiance": [a,b,c],            // Radiance [W/m²sr]
    "scale": float,                 // Multiplier for "radiance"

`"type": "spot"`\
PBRT type of spot light: "intensity" * clamp((cosθ - "cosWidth") / ("falloffStart" - "cosWidth"), 0, 1) ^ "exponent".

    "intensity": [a,b,c],           // Peak intensity [W/sr]
    "scale": float,                 // Multiplier for "intensity"
    "exponent": float,
    "cosWidth" or "width": float,   // An angle "width" in radiant for the half-opening angle or the
                                    // cosine of this angle
    "cosFalloffStart" or "falloffStart": float  // An angle "falloffStart" in radiant for the angle up to
                                                //which the peak intensity is used or the cosine of this angle

`"type": "envmap"`

    "map": "<texture name>",        // A 360° texture (polar-mapped, cubemap), relative to this file, interpreted as radiance [W/m²sr]
    "scale": float,                 // An energy scaling factor for the environment map

`"type": "goniometric"`\
A measured light source. Similar to a point light

    "position": [x,y,z],            // vec3 world space position
    "map": "<texture name>"         // A 360° texture (polar-mapped, cubemap), relative to this file, interpreted as intensity [W/sr],
    "scale":, float,                // Multiplier for the "map"

Materials
--

Alias types:
* "glass" = "fresnel"["torrance", "walter"]

  prefer for optimal sampling
* "opaque" = "fresnel"["torrance", "lambert"]



---

The binary file (*.mff)
-

The binary file contains all the geometric and instancing data of a scene with a strict ordering.
It does not contain textures or ascii headers.

The high level structure is

    <MATERIALS_HEADER>
    <OBJECTS>
      <OBJECTS_HEADER>
      <OBJECTS>
    <INSTANCES>

The <MATERIALS_HEADER> maps integer indices to the "binary-names" which are used as a link in the JSON file:

    <MATERIALS_HEADER> = u32 'Mats' // Type check for this section
                         u64        // absolute start position of next section (<OBJECTS>)
                         u32        // Number of materials M
                         M*<STRING> // M strings, the index of the string in this array is the <MATID> (0-based)

    <STRING> = u32      // Number of bytes L in this string
               L*u8     // UTF-8 string with length L

The <OBJECTS_HEADER> provides a fast way to find a single object for partial and masked loading processes.
Since the same pattern is used for LOD inside objects, their is a generic specification for the <JUMP_TABLE>.

    <OBJECTS_HEADER> = u32 'Objs'   // Type check for this section
                       u64          // Absolute start position of next section (<INSTANCES>)
                       <JUMP_TABLE>

    <JUMP_TABLE> = u32          // Number of entries in the table N
                   N*u64        // Absolute start position of the object/LOD,
                                // the index in the array is the <OBJID>/lod level (0-based)

    <OBJECTS> = u32 <FLAGS>
                N*<OBJECT>      // List of objects with the number specified by N from the jump table

    <FLAGS> = DEFLATE 1                 // All <VERTEXDATA>, <SPHERES>, and index data (<TRIANGLES>, <QUADS>
                                        // indies, material indices) and <ATTRIBUTES> are
                                        // deflate compressed. All optional compressed blocks are marked
                                        // with ' (each ' is one independent DEFLATE stream).
            | COMPRESSED_NORMALS 2      // Normals can be compressed into 32bit with a custom compression

    <OBJECT> = u32 'Obj_'       // Type check for this section
               <STRING>         // The name, used as [obj:name] in the above JSON specification
               u32              // Keyframe of the object if animated or 0xffffffff
               u32              // <OBJID> of the previous object in an animation sequence or 0xffffffff
               <JUMP_TABLE>     // Jump table over LODs (number = D)
               D*<LOD>

LODs contain the real geometry. It must have sorted geometry, because attributes (like count of indices) differ per geometry.

    <LOD> = u32 'LOD_'          // Type check for this section
            u32                 // Number of triangles T
            u32                 // Number of quads Q
            u32                 // Number of spheres S
            u32                 // Number of vertices V
            u32                 // Number of vertex attributes
            u32                 // Number of face attributes (same for triangles and quads)
            u32                 // Number of sphere attributes
            <VERTEXDATA>'
            <ATTRIBUTES>'       // Optional Vertex attributes
            <TRIANGLES>'
            <QUADS>'
            <SPHERES>'
    <VERTEXDATA> = V*3*f32      // Positions (vec3)
                   V*3*f32 | V*u32  // Normals (vec3, normalized)
                                // OR compressed normals (Oct-mapping see below)
                   V*2*f32      // UV coordinates (vec2)

    <TRIANGLES> = T*3*u32       // Indices of the vertices (0-based, per LOD)
                  T*u16         // Material indices (<MATID> from <MATERIALS_HEADER>)
                  <ATTRIBUTES>  // Optional list of face attributes
    <QUADS> = Q*4*u32           // Indices of vertices (0-based, per LOD)
              Q*u16             // Material indices (<MATID> from <MATERIALS_HEADER>)
              <ATTRIBUTES>      // Optional list of face attributes, must have the same entries as for triangles
    <SPHERES> = S*4*f32         // Position (3 f32) and radius (1 f32) interleaved
                S*u16           // Material indices (<MATID> from <MATERIALS_HEADER>)
                <ATTRIBUTES>

The custom attributes can add additional information to triangles, quads and spheres.
There might be attributes outside this specification. Therefore the syntax is very general.
However, there is a specification for a few predefined possible attributes (with [] below).

    <ATTRIBUTES> = u32 'Attr'   // Type check for this section
                   <STRING>     // Name (Custom)
                   <STRING>     // Meta information (Custom)
                   u32          // Meta information (Custom flags)
                   u64          // Size in bytes
                   <BYTES>      // Size many bytes

    [AdditionalUV2D] = "AdditionalUV2D"
                       "{Light, Displacement}"
                       0
                       #Bytes to be interpreted as 2*f32 per vertex/face
    [Color8] = "Color8"
               "{RGBA, sRGB_A}"
               0
               #Bytes to be interpreted as 8-bit RGBA normalized uint tuples (in total 32 bit per element)
    [Color32] = "Color32"
                "{RGBA, sRGB_A}"
                0
                #Bytes to be interpreted as 32-bit RGBA float tuples (in total 128 bit per element)

Finally, there are the instances.
Objects which are not instanced explicitly will have one instance with the identity transformation.

    <INSTANCES> = u32 'Inst'    // Type check for this section
                  u32           // Number of instances I
                  I*<INSTANCE>
    <INSTANCE> = u32            // <OBJID> (0-based index of the object in the <OBJECTS> section)
                 u32            // Keyframe of the instance if animated or 0xffffffff
                 u32            // <InstID> of the previous object in an animation sequence or 0xffffffff
                 12*f32         // 3x4 transformation matrix (rotation, scaling, translation)