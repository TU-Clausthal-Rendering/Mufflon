Often, preventing code reduncy is a good thing to limit the number of changes and resulting bugs.
It increases the testability, reliability and understanding.
However, in research-code the number of variants and experiments leads to many branches (including precompiler) which make code reduncy desirable.

Rules
-

1. Copy renderers whenever any change/experiment is performed.

   Reason: Stability. It is necessary to rely on tested renderes like a BPT. Any change which might break the outcome (bias, NaNs, ...) is bad. Bug fixes and tested performance enhancements should of course be applied to old renderers.

2. Prevent reduncy within one code unit

3. Share unchanging algorithms

   Example: BRDF or light source evaluation code.

   If a BRDF/light/... is changed make a copy of it. This is the same as for renderers. Rendering an old scene should result in the same outcome.


---------------------

Break rules if reasonable. The rules above are based on experience and no laws of nature. If new experiences require reasonable changes document them.
Mark an outdate rule with [Outdated]. And new overwriting rules with [Replaces X.]