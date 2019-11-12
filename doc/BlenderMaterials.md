Blender Materials Mapping
=

From blender 2.8 and onward we only support the nodes system. Since our renderer has only a limited fixed material set, only a few node configurations are valid.

Diffuse BSDF
-

Maps to `lambert` if `Roughness==0`, otherwise maps to `orenNayar`

Glass BSDF
-

Maps to `microfacet`.
Must be used stand alone (not in mixed materials)

Uses 1-`Color` as absorption coefficients.

Only `Beckmann`, `GGX` and `SHARP` are valid choices of NDFs.

The `Roughness` value is squared on export (matches cycles interpretation)

Refraction BSDF
-

Same parameter mapping as in Glass BSDF, but type is `walter`. This means, the renderer will not have a Fresnel reflective component.

Glossy BSDF
-

Maps to `torrance`.

Only `Beckmann`, `GGX` and `SHARP` are valid choices of NDFs.

The `Roughness` value is squared on export (matches cycles interpretation)

Anisotropic BSDF
-

Maps to `torrance`

Only `Beckmann` and `GGX` are valid choices of NDFs.

The `Roughness` value is squared on export (matches cycles interpretation), α = R^2

The `Anisotropy` value is mapped according to the cycles interpretation:

    anisotropy = clamp(anisotropy, -0.99, 0.99)
    if anisotropy < 0:
        [α / (1 + anisotropy), α * (1 + anisotropy)]
    else:
        [α * (1 - anisotropy), α / (1 - anisotropy)]

Emissivion
-

Maps to `emissive`

Mix Shader
-
Can be used to blend several materials with constant or Fresnel factors. If `Fac` is a number a `blend` material will be created. If it is a `Fresnel` shader node, a Fresnel blend material is created. Textured blending is not supported.

Valid combinations are (any order of the two):

* Lambert + Emissive (Value blend only)
* Lambert + Torrance
* Torrance + Walter

Custom Properties
=

There is currently one custom property to specify the outer medium at an interface. This parametrization is unique to our renderer and has no mapping in blender.

`outerMedium` must be a custom property of the material with 4 float values: [ior, r, g, b]. All values must match the values of a glass or refractive material around the object. The values r,g,b are the absorptions coefficients λ (used as exp(-d λ)). Note that the Glass BSDF uses 1-r,g,b.