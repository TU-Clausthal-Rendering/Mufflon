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
            "<name1>" {
                "type": "{pinhole, focus}",
                "path": [[x1,y1,z1], ...],    // List of vec3 (at least 1)
                "viewDir": [[x1,y1,z1], ...], // List of vec3 with the same length as "path", not necessarily normalized
                "up": [[x1,y1,z1], ...],      // OPTIONAL [0,1,0], must have the same length as "path" if defined, not necessarily normalized
                ...
            },
            "<name2>" {
                ...
            } ...
        },
        "lights" {
            "<name1>": {
                "type": "{point, directitonal, spot, envmap, goniometric}",
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

Materials
--

Alias types:
* "glass" = "fresnel"["torrance", "walter"]

  prefer for optimal sampling
* "opaque" = "fresnel"["torrance", "lambert"]



---

The binary file
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
                   N*u64        // Absolute start position of the object/LOD, the index in the array is the <OBJID>/lod level (0-based)

    <OBJECTS> = N*<OBJECT>      // List of objects with the number specified by N from the jump table

    <OBJECT> = <STRING>         // The name, used as [obj:name] in the above JSON specification
               <JUMP_TABLE>     // Jump table over LODs (number = D)
               D*<LOD>

LODs contain the real geometry. It must have sorted geometry, because attributes (like count of indices) differ per geometry.

    <LOD> = <TRIANGLES>
            <QUADS>
            <SPHERES>
    <TRIANGLES> = TODO

            TODO:
            Vertex <POSITIONS>, <NORMALS>, <UV>, CUSTOM
            Tri, indices, material
            Sphere <POSITION>, <RADIUS>, <COLOR>/CUSTOM

Finally, there are the instances.
Objects which are not instanced explicitly will have one instance with the identity transformation.

    <INSTANCES>