build/
-

This directory contains the build files from all configurations and projects, including .exe, .dll, and .pdb. The subdirectory 'temp' contains temporary files created during compilation.

doc/
-

Contains the documentation of the project. Here go readme, explanatory documents, requirement and issue reports.


external/
-

This directory contains the project's precompiled dependencies which do not need to be installed. Each dependency gets its own subdirectory, with its version as part of the folder's name. It is required of all dependencies included this way to come with a debug and release version, which are to be put into separate subdirectories.

project_files/
-

Here go any project files created by Visual Studio bar the solution file.

src/
-

This directory contains all source and header files mixed together. Folder names should indicate what module the files belong to.