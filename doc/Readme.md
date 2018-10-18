How to contribute
=

A documentation is only useful if it is up to date. Therefore the provided information in this directory are to be held light-weight.
Most documentation should be made at the declaration of interfaces and other points in the source code.
Rather, the documentation provided here should give a hint to higher level constructs and intents.

There is a concept called [Architecture Decision Records (ADR)](http://thinkrelevance.com/blog/2011/11/15/documenting-architecture-decisions) with exactly the intend to have a maintainable and useful documentation.
An ADR is created whenever a high level decision for the project is made.
It explains why something is as it is.
An ARD can be revisited and changed later, but should always include its own history of previous decisions.
The gain is that new contributors get a fast introduction to the project.
Also, there will be less arbitrary repeated changes.
Anyone who thinks "this could be done better" should first look if and why it was not done better.
If the new idea is indeed better, new ADRs documenting the idea should be created and outdate ones annotated.
It is important to keep each one in the history to make sure bad ideas are not repeated.

The structure of an ARD is:

1. A **numbered** **title**
2. A problem description (**context**)
3. Description of the **decision** with appropriate arguments
4. **Consequences** predicted or real effects based on the decision.


Everything should be written in full sentences.
Using only short phrases and key words might seem sufficient in the moment of writing,
however it often turns out later that nobody remembers the intention of some of the notes.
Writing sentences reduces this problem and increases the usefulness of the ADR.


Current ADRs
-

* A0: Naming Conventions
* A1: Code Redundancy
* A2: Libraries
* A3: Geometry Support