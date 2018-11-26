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
        "binary": "<file name relative to this json>",
		"defaultScenario": "<name of a scenario>"  // OPTIONAL the scenario to load on startup.
		                                           // If none is given, the chosen scenario is unspecified
        "cameras": {
            "<name1>": {
                "type": "{pinhole, focus, ortho}",
                "path": [[x1,y1,z1], ...],    // List of vec3 (at least 1)
                "viewDir": [[x1,y1,z1], ...], // List of vec3 with the same length as "path", not necessarily normalized
                "up": [[x1,y1,z1], ...],      // OPTIONAL [0,1,0], must have the same length as "path" if defined, not necessarily normalized
                "near": float,                // OPTIONAL >0 near clipping plane, Default: 1/10000 * scene size (diagonal of BB)
                "far": float,                 // OPTIONAL >"near" far clipping plane, Default: 2 * scene size (diagonal of BB)
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
                "outerMedium": {      // OPTIONAL, the inner medium is always specified by the other material parameters
                    "refractionIndex": float | [n,k],   // The real part of the refraction index (for dielectric)
                                                        // OR complex number (vec2, for conductor)
                    "absorption": [r,g,b]               // Absorption λ per meter (transmission = exp(-λ*d)) [0,inf]^3
                },
                ...
            } ...
        },
        "scenarios": {
            "<name1>: {
                "camera": "<name of a camera>",
                "resolution": [int,int],        // Target image resolution
                "lights": ["<name of a light>", ...]  // List of light sources
                "lod": int,             // Global level of detail number [0,...] where 0 has the highest resolution, OPTIONAL 0
				"materialAssignments": {
					"[mat:name1]": "<name of a material>",
					"[mat:name2]": "<name of a material>",
					....                    // Each material in the binary must be mapped to one of the above materials.
											// A material can be used by multiple binray-materials.
				},
				"objectProperties": {
					"[obj:name]": {         // OPTIONAL per object properties
						"mask",             // Do not render this object (blacklisted)
						"lod": int,         // Use a specific LOD different/independent from global LOD
						// More meta information
					}
				}
            }
            ...
        }
    }

Cameras
--

`"type": "pinhole"`\
Infinitely sharp camera model.

    "fov": float     // vertical field of view in degree [°], DEFAULT 25°

`"type": "focus"`\
Realistic camera model.

    "focalLength": float,           // FocalLength in [mm] (typical 10-400mm), DEFAULT 35mm
    "chipHeight": float,            // Chip heigh in [mm] (typical <24mm), DEFAULT 24mm
    "focusDistance": float,         // sharp distance in meters [m],
    "aperture": float               // Aperture in f-stops (typical 1.0, 1.4, 1.8, ...), DEFAULT 1.0

`"type": "ortho"`\
The orthographic camera spans a box volume which is projected to a plane with position and orientation specified by "path", "viewDir" and "up".
It has a near clipping at this plane (lower half space is not projected), but extends to infinity in positive direction.
The box is spanned symmetrical around the position ("width"/2 in each horizontal direction, ...).

    "width": float,                 // Width of the box volume in meter [m]
    "height": float,                // Height of the box volume in meter [m]

Lights
--

This section lists the required attributes for the different types of light sources

`"type": "point"`

    "position": [x,y,z],            // vec3 vec3 world space position in [m]
    "flux" or "intensity: [a,b,c],  // Exclusive (either flux [W] or intensity [W/sr] must be specified as vec3)
    "scale": float,                 // Multiplier for "flux"/"intensity"

`"type": "directional"`

    "direction": [x,y,z],           // Direction in which the light travels (incident direction), not necessarily normalized
    "radiance": [a,b,c],            // Radiance [W/m²sr]
    "scale": float,                 // Multiplier for "radiance"

`"type": "spot"`\
PBRT type of spot light: "intensity" * clamp((cosθ - "cosWidth") / ("falloffStart" - "cosWidth"), 0, 1) ^ "exponent".

    "position": [x,y,z],            // vec3 vec3 world space position in [m]
    "direction": [x,y,z],           // Direction in which the light travels (incident direction), not necessarily normalized
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

    "position": [x,y,z],            // vec3 world space position in [m]
    "map": "<texture name>"         // A 360° texture (polar-mapped, cubemap), relative to this file, interpreted as intensity [W/sr],
    "scale":, float,                // Multiplier for the "map"

Materials
--

`"type": "lambert"`

    "albedo": [r,g,b] | <texture>   // vec3 [0,1]^3 for the color OR an RGB texture (relative path)
                                    // DEFAULT: [0.5, 0.5, 0.5]

`"type": "torrance"`

    "roughness": float | [α_x,α_y,r]    // isotropic roughness value [0,1] (except Beckmann [0,inf])
                | <texture>,            // OR anisotropic roughness and angle in radiant [0,1]^2 x [0,π]
                                        // OR a texture with one or three channels (relative path)
                                        // DEFAULT: 0.5
	"ndf": "{BS,GGX,Cos}",				// Name of the normal distribution function
    "albedo": [r,g,b] | <texture>       // vec3 [0,1]^3 for the color OR an RGB texture (relative path)
                                        // DEFAULT: [0.5, 0.5, 0.5]

`"type": "walter"`

    "roughness": float | [α_x,α_y,r]    // isotropic roughness value [0,1] (except Beckmann [0,inf])
                 | <texture>,           // OR anisotropic roughness and angle in radiant [0,1]^2 x [0,π]
                                        // OR a texture with one or three channels (relative path)
                                        // DEFAULT: 0.5
	"ndf": "{BS,GGX,Cos}",				// Name of the normal distribution function
    "absorption": [r,g,b]               // Absorption λ per meter (transmission = exp(-λ*d)) [0,inf]^3

`"type": "emissive"`

    "radiance": [r,g,b] | <texture>,    // Surface radiance in [W/m²sr]
    "scale: float                       // Multiplier for radiance

`"type": "orennayar"`

    "albedo": [r,g,b] | <texture>,  // vec3 [0,1]^3 for the color OR an RGB texture (relative path)
                                    // DEFAULT: [0.5, 0.5, 0.5]
    "roughness": float              // [0,π/2], with 0 this resembles to "lambert"
                                    // DEFAULT: 1.0

`"type": "blend"`\
Additive blending. Usually the factors for the two layers should be positive and add to one for physical plausible results.
It is allowed to use all kinds of factors if desired.
E.g. "factorA" = "factorB" = 1 makes sense for an "emissive", "lambert" mixed material.

    "layerA": {
        <recursive material>        // A different material beginning with "type"...
    },
    "layerB": {
        <recursive material>        // A different material beginning with "type"...
    },
    "factorA": float,               // Factor which is multiplied with the reflectance of layer A
    "factorB": float,               // Factor which is multiplied with the reflectance of layer B

`"type": "fresnel"`\
Angular dependent blending of two layers (dielectric-dielectric DD or dielectric-conductor DC fresnel).

    "refractionIndex": float | [n,k], // The real part of the refraction index (for DD) OR complex number (vec2, for DC)
    "layerReflection": {
        <recursive material>        // A different material beginning with "type"...
    },
    "layerRefraction": {
        <recursive material>        // A different material beginning with "type"...
    }

Alias types:
* "glass" = "fresnel"["torrance", "walter"]

  prefer for optimal sampling



---

The binary file (*.mff)
-

The binary file contains all the geometric and instancing data of a scene with a strict ordering.
It does not contain textures or ascii headers.
Binary data is stored in little endian (native on x86_64).

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
               3*f32            // Bounding box min for the object (in object space)
               3*f32            // Bounding box max for the object (in object space)
               <JUMP_TABLE>     // Jump table over LODs (number = D)
               D*<LOD>          // LODs sorted after detail (0 has the highest detail)

LODs contain the real geometry. It must have sorted geometry, because attributes (like count of indices) differ per geometry.

    <LOD> = u32 'LOD_'          // Type check for this section
            u32                 // Number of triangles T
            u32                 // Number of quads Q
            u32                 // Number of spheres S
            u32                 // Number of vertices V
            u32                 // Number of edges E
            u32                 // Number of vertex attributes
            u32                 // Number of face attributes (same for triangles and quads)
            u32                 // Number of sphere attributes
            <VERTEXDATA>'
            <ATTRIBUTES>'       // Optional Vertex attributes
            <TRIANGLES>'
            <QUADS>'
			(T+Q)*u16'           // Material indices (<MATID> from <MATERIALS_HEADER>, in order: first triangles, then quads)
            <ATTRIBUTES>'        // Optional list of face attributes (in order: first triangles, then quads)
            <SPHERES>'
    <VERTEXDATA> = V*3*f32      // Positions (vec3, in meter [m])
                   V*3*f32 | V*u32  // Normals (vec3, normalized)
                                // OR compressed normals (Oct-mapping see below)
                   V*2*f32      // UV coordinates (vec2)

    <TRIANGLES> = T*3*u32       // Indices of the vertices (0-based, per LOD)
    <QUADS> = Q*4*u32           // Indices of vertices (0-based, per LOD)
    <SPHERES> = S*4*f32         // Position (3 f32) and radius (1 f32) interleaved in meter [m]
                S*u16           // Material indices (<MATID> from <MATERIALS_HEADER>)
                <ATTRIBUTES>

The custom attributes can add additional information to triangles, quads and spheres.
There might be attributes outside this specification. Therefore the syntax is very general.
However, there is a specification for a few predefined possible attributes (with [] below).

    <ATTRIBUTES> = u32 'Attr'   // Type check for this section
                   <STRING>     // Name (Custom)
                   <STRING>     // Meta information (Custom)
                   u32          // Meta information (Custom flags)
				   u32 <TYPE>   // Type information
                   u64          // Size in bytes
                   <BYTES>      // Size many bytes
	<TYPE> = i8 0
	       | u8 1
	       | i16 2
	       | u16 3
	       | i32 4
	       | u32 5
	       | i64 6
	       | u64 7
	       | f32 8
	       | f64 9
	       | 2*u8 10
	       | 3*u8 11
	       | 4*u8 12
	       | 2*i32 13
	       | 3*i32 14
	       | 4*i32 15
	       | 2*f32 16
	       | 3*f32 17
	       | 4*f32 18

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


Normal Compression
--
Additional to the global DEFLATE compression normals have a custom discretization if `COMPRESSED_NORMALS` is set.
For compression/decompression the following codes is used:

    # map a direction from the sphere to u,v
    packNormal32(vec3 dir) -> (u32 uv)
        l1norm = abs(dir.x) + abs(dir.y) + abs(dir.z)
        if dir.z >= 0:
            u = dir.x / l1norm
            v = dir.y / l1norm
        else # warp lower hemisphere
            u = (1 - dir.y / l1norm) * (dir.x >= 0 ? 1 : -1)
            v = (1 - dir.x / l1norm) * (dir.y >= 0 ? 1 : -1)
        end
        u = floor((u / 2 + 0.5) * 65535.49 + 0.5)   # from [-1,1] to [0,2^16-1]
        v = floor((v / 2 + 0.5) * 65535.49 + 0.5)   # from [-1,1] to [0,2^16-1]
        return u | (v << 16)
    end

    # map a 32 bit uint back to a normal vector
    unpackNormal32(u32 uv) -> (vec3 dir)
        u = (uv & 0xff) / float(2^16-1)
        v = (uv >> 16) / float(2^16-1)
        u = u * 2 - 1
        v = v * 2 - 1
        z = 1 - abs(u) - abs(v)
        if z >= 0:
            x = u
            y = v
        else
            x = (1 - abs(v)) * (u >= 0 ? 1 : -1)
            y = (1 - abs(u)) * (v >= 0 ? 1 : -1)
        end
        return normalize([x,y,z])
    end