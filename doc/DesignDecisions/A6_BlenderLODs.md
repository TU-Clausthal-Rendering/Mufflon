A6 LODs in Blender
=

While not part of the project itself, we have a custom format exporter from Blender.
Unfortunately, Blenders LOD system has some drawbacks and requires a custom setup.

Intrinsic System (in use)
-

If the renderer is set to "Blender Game" an LOD property panel appears in the object properties.
Here some other object can be linked.
However, the two objects remain independent objects which are both rendered/shown and are not transformed together.
Also, it is unclear for the linked object how it knows to be part of a LOD of some bigger object.

To solve this problems we define:

1. LODs are linked in a **ring buffer**
2. The **least** detailed LOD must have the **highest "Distance"** in the LOD property
3. There is a proxy **without geometry** linking one or multiple LODs.\
  The proxy is the object which gets instanced

Optionally, the LODs are in a separate layer and only proxies appear in the scene.

Consequence for an exporter: all LODs of one object can be found by following the LOD ring buffer chain. Non-ring buffer LODs are ignored and instanced as usual objects. The highest "Distance" allows to find the least detailed LOD (whose next LOD is the highest detailed one). Instances can be detected by: *they point to a LOD ring buffer, but are not part of the ring* AND *they do not have own geometry*. It should be sufficient to check the second property which must be ensured by a correct workflow.

**Blender Workflow**

* Set renderer to "Blender Game"
* Create LODs on a separate layer. The objects do not need to have the same translation matrix... (Object space is used from Blender and Exporter).
* Link all LODs in a big cycle.
* Adding or removing new LODs is simple by adding/removing an element in the ring.
* Make sure the least detailed LOD has the highest "Distance" in the LOD property.
* Create a proxy: e.g. a box, delete its geometry in edit mode, link one or multiple LODs for the blender scene rendering
* Create instances of the proxy.

It may happen that different instances link different LODs. This is fine. The exporter does not use the information of the proxy to detect the LODs (it uses the ring-buffer).

[DEPRECATED] Former Decision: our Workaround
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