A6 LODs in Blender
=

While not part of the project itself, we have a custom format exporter from Blender.
Unfortunately, Blender doesn't have a convenient LOD system and requires a custom setup.

Intrinsic System (do not use)
-

If the renderer is set to "Blender Game" an LOD property panel appears in the object properties.
Here some other object can be linked.
However, the two objects remain independent objects which are both rendered/shown and are not transformed together.
Also, it is unclear for the linked object how it knows to be part of a LOD of some bigger object.

Our Workaround
-

Parenting allows to transform multiple objects as if they are one.
Since parenting is a useful tool, we need to distinguish between standard parents and LOD-objects (which are the parent of all LODs).
An LOD-object must have the custom property "LODGroup".
The children do not need any additional information and each of them is seen as one level of detail.
The order of LOD levels must be induced from the name.
Therefore, each children (LOD) must have an name "<bla>X" where <bla> can be freely chosen and X is a signed integer number (arbitrary number of digits).
The integer must be extracted and the LODs be sorted by the integers ascending.
I.e. the smallest integer is assumed to present the highest detail LOD.
The integers are not necessarily consecutive.

**Example**:\
[Blender:]

    Statue       // Empty object with custom property "LODGroup"
     |-- StatueL-1
     |-- StatueL3
     |-- StatueL0

The above must be exported in the order StatueL-1, StatueL0, StatueL3 as three LODs of the same object with name "Statue" (from the root object).

**How to instance an LOD-object?**

Instancing the top level object unfortunately only instances the parent element.
To be able to see the instance the sub-content must be duplicated explicitly.
This is possible by *Shift* selecting all of the objects. Important: to duplicate all LODs, they must all be visible.

When exporting the exporter has to ignore all children of the instance object.
Independent on which children where duplicated, this is still an instance of the entire object with all its LODs.

    Statue <--------------- Statue2 (Instance of Statue)
     |-- StatueL-1           |-- Statue2L0 (Instance of StatueL0)
     |-- StatueL3
     |-- StatueL0

In the example there is on Object (Statue) with three LODs from which two instances exist.