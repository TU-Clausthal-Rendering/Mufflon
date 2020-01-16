cmake_minimum_required(VERSION 3.12)
project("mufflon" VERSION 0.0.1 LANGUAGES C CXX CUDA)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)
# Force 'Release' as build type if no other has been specified
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to 'Release' as none was specified.")
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release"
    "MinSizeRel" "RelWithDebInfo")
endif()

# Discourage in-source builds
if("${PROJECT_SOURCE_DIR}" STREQUAL "${PROJECT_BINARY_DIR}")
   message(WARNING "In-source building is discouraged!")
endif("${PROJECT_SOURCE_DIR}" STREQUAL "${PROJECT_BINARY_DIR}")

# We need CUDA, OpenMP is optional; other dependencies are packaged as sub-repos or in binary form
find_package(OpenMP)
find_package(CUDA REQUIRED)

if(NOT OpenMP_CXX_FOUND)
	message(WARNING "OpenMP is strongly recommended, as all CPU-side parallel code relies on it and will run serially without it")
endif()

option(BUILD_SHARED_LIBS "Build shared library" ON)
# Option to be more strict on compilation
option(MUFFLON_STRICT_COMPILATION "Enable W4/Wextra and WX/Werror" OFF)
option(MUFFLON_ENABLE_OPEN_DENOISE "Enable OpenDenoise" ON)
set(MUFFLON_ENABLE_SANITIZER "OFF" CACHE STRING "Compile with sanitizer support")
set_property(CACHE MUFFLON_ENABLE_SANITIZER PROPERTY STRINGS OFF ADDRESS UB THREAD LEAK)
# CUDA arch
set(MUFFLON_CUDA_ARCH "Auto" CACHE STRING "CUDA target compute capability (Auto uses present GPUs")

# Select the compilation target for NVCC in terms of compute capability
CUDA_SELECT_NVCC_ARCH_FLAGS(ARCH_FLAGS ${MUFFLON_CUDA_ARCH})
if("${MUFFLON_CUDA_ARCH}" STREQUAL "Auto")
	STRING(REGEX REPLACE "sm_(.)(.)" "\\1.\\2" ARCH_CC ${ARCH_FLAGS_readable})
	set(MUFFLON_CUDA_ARCH "${ARCH_CC}" CACHE STRING "CUDA target compute capability (Auto uses present GPUs)" FORCE)
	message(STATUS "Detected CUDA arch: ${MUFFLON_CUDA_ARCH}")
endif()
# For whatever reason the arch flags may be in the wrong format, so we gotta fix that
string(REGEX REPLACE "gencode;" "gencode=" ARCH_FLAGS "${ARCH_FLAGS}")
string(REGEX REPLACE ";code" ",code" ARCH_FLAGS "${ARCH_FLAGS}")
string(REGEX REPLACE ";-gencode" " -gencode" ARCH_FLAGS "${ARCH_FLAGS}")
# Since CMake's target_link_options does not pass options to nvlink yet, we have to set them globally to the NVCC flags...
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${ARCH_FLAGS}")

# Only MSVC generators have C# support
if(MSVC)
	enable_language(CSharp)
	# Option to not build GUI
	option(MUFFLON_BUILD_GUI "Build the GUI or not" ON)
	# C# version
	set(MUFFLON_DotNet_VERSION "4.6.1" CACHE STRING ".NET version to use")
endif()

# Ensure that our binaries get put in the proper directory
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/build/$<CONFIG>")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/build/$<CONFIG>")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/build/$<CONFIG>")

# For Linux and MacOS systems we have to set the rpath of the resulting shared libs
set(CMAKE_MACOSX_RPATH 1)
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}")

# Store our source files in a variable to auto-gen source groups
file(GLOB_RECURSE MUFFLON_CORE_SOURCE_LIST CONFIGURE_DEPENDS
	"${CMAKE_CURRENT_SOURCE_DIR}/src/core/*"
	"${CMAKE_CURRENT_SOURCE_DIR}/src/plugin/*"
	"${CMAKE_CURRENT_SOURCE_DIR}/src/profiler/*"
	"${CMAKE_CURRENT_SOURCE_DIR}/src/util/*"
)
file(GLOB_RECURSE MUFFLON_SHADER_SOURCE_LIST CONFIGURE_DEPENDS
	"${CMAKE_CURRENT_SOURCE_DIR}/src/*.glsl"
)
file(GLOB_RECURSE MUFFLON_GUI_SOURCE_LIST CONFIGURE_DEPENDS
	"${CMAKE_CURRENT_SOURCE_DIR}/project_files/gui/*.cs"
	"${CMAKE_CURRENT_SOURCE_DIR}/project_files/gui/*.xaml"
	"${CMAKE_CURRENT_SOURCE_DIR}/project_files/gui/*.settings"
	"${CMAKE_CURRENT_SOURCE_DIR}/project_files/gui/*.resx"
)
file(GLOB MUFFLON_GUI_ICONS CONFIGURE_DEPENDS
	"${CMAKE_CURRENT_SOURCE_DIR}/project_files/gui/Icons/*"
)
list(FILTER MUFFLON_GUI_ICONS EXCLUDE REGEX "gitignore")
list(APPEND MUFFLON_GUI_SOURCE_LIST ${MUFFLON_GUI_ICONS})

file(GLOB_RECURSE MUFFLON_GUI_CS_XAML_LIST CONFIGURE_DEPENDS
	"${CMAKE_CURRENT_SOURCE_DIR}/project_files/gui/*.xaml.cs"
)

file(GLOB_RECURSE MUFFLON_GLILOADER_SOURCE_LIST CONFIGURE_DEPENDS
	"${CMAKE_CURRENT_SOURCE_DIR}/src/gliloader/*"
	"${CMAKE_CURRENT_SOURCE_DIR}/src/plugin/*"
	"${CMAKE_CURRENT_SOURCE_DIR}/src/util/*"
)
file(GLOB_RECURSE MUFFLON_MFFLOADER_SOURCE_LIST CONFIGURE_DEPENDS
	"${CMAKE_CURRENT_SOURCE_DIR}/src/mffloader/*"
	"${CMAKE_CURRENT_SOURCE_DIR}/src/util/*"
	"${CMAKE_CURRENT_SOURCE_DIR}/src/profiler/cpu_*"
	"${CMAKE_CURRENT_SOURCE_DIR}/src/profiler/profiling*"
)
file(GLOB_RECURSE MUFFLON_PFMLOADER_SOURCE_LIST CONFIGURE_DEPENDS
	"${CMAKE_CURRENT_SOURCE_DIR}/src/pfmloader/*"
	"${CMAKE_CURRENT_SOURCE_DIR}/src/plugin/*"
	"${CMAKE_CURRENT_SOURCE_DIR}/src/util/*"
)
file(GLOB_RECURSE MUFFLON_STBILOADER_SOURCE_LIST CONFIGURE_DEPENDS
	"${CMAKE_CURRENT_SOURCE_DIR}/src/plugin/*"
	"${CMAKE_CURRENT_SOURCE_DIR}/src/stbiloader/*"
	"${CMAKE_CURRENT_SOURCE_DIR}/src/util/*"
)

# Dependency sources
file(GLOB_RECURSE CHINOISE_SOURCE_LIST CONFIGURE_DEPENDS
	"${CMAKE_CURRENT_SOURCE_DIR}/deps/chinoise/include/*.inl"
	"${CMAKE_CURRENT_SOURCE_DIR}/deps/chinoise/include/*.hpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/deps/chinoise/src/*.cpp"
)
file(GLOB_RECURSE CUB_SOURCE_LIST CONFIGURE_DEPENDS
	"${CMAKE_CURRENT_SOURCE_DIR}/deps/cub/cub/*.cuh"
)
file(GLOB_RECURSE EPSILON_SOURCE_LIST CONFIGURE_DEPENDS
	"${CMAKE_CURRENT_SOURCE_DIR}/deps/epsilon/include/ei/*.hpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/deps/epsilon/include/ei/*.inl"
	"${CMAKE_CURRENT_SOURCE_DIR}/deps/epsilon/include/ei/*.cpp"
)
file(GLOB_RECURSE GLI_SOURCE_LIST CONFIGURE_DEPENDS
	"${CMAKE_CURRENT_SOURCE_DIR}/deps/gli/gli/core/*.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/deps/gli/gli/core/*.inl"
	"${CMAKE_CURRENT_SOURCE_DIR}/deps/gli/gli/core/*.hpp"
)
set(GLAD_SOURCE_LIST
	"${CMAKE_CURRENT_SOURCE_DIR}/deps/glad/glad/glad.h"
	"${CMAKE_CURRENT_SOURCE_DIR}/deps/glad/KHR/khrplatform.h"
	"${CMAKE_CURRENT_SOURCE_DIR}/deps/glad/glad.c"
)
set(MINIZ_SOURCE_LIST
	"${CMAKE_CURRENT_SOURCE_DIR}/deps/miniz/miniz/miniz.c"
	"${CMAKE_CURRENT_SOURCE_DIR}/deps/miniz/miniz/miniz.h"
)
set(STBI_SOURCE_LIST
	"${CMAKE_CURRENT_SOURCE_DIR}/deps/stbi/stbi/stb_image.h"
	"${CMAKE_CURRENT_SOURCE_DIR}/deps/stbi/stbi/stb_image_write.h"
)
# Source groups for easier navigating
source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}/src"
	PREFIX "src"
	FILES ${MUFFLON_CORE_SOURCE_LIST}
)
source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}/src"
	PREFIX "src"
	FILES ${MUFFLON_GLILOADER_SOURCE_LIST}
)
source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}/src"
	PREFIX "src"
	FILES ${MUFFLON_MFFLOADER_SOURCE_LIST}
)
source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}/src"
	PREFIX "src"
	FILES ${MUFFLON_PFMLOADER_SOURCE_LIST}
)
source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}/src"
	PREFIX "src"
	FILES ${MUFFLON_STBILOADER_SOURCE_LIST}
)
source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}/src"
	PREFIX "src"
	FILES ${MUFFLON_SHADER_SOURCE_LIST}
)
source_group(deps\\chinoise FILES ${CHINOISE_SOURCE_LIST})
source_group(deps\\cub FILES ${CUB_SOURCE_LIST})
source_group(deps\\epsilon FILES ${EPSILON_SOURCE_LIST})
source_group(deps\\glad FILES ${GLAD_SOURCE_LIST})
source_group(deps\\gli FILES ${GLI_SOURCE_LIST})
source_group(deps\\miniz FILES ${MINIZ_SOURCE_LIST})
source_group(deps\\stbi FILES ${STBI_SOURCE_LIST})
source_group(bindings FILES "${CMAKE_CURRENT_SOURCE_DIR}/src/pythonbinding/bindings.py")

# Build artifacts (we include the deps sources to get them to show up in visual studio)
add_library(core ${MUFFLON_CORE_SOURCE_LIST} ${CHINOISE_SOURCE_LIST} ${EPSILON_SOURCE_LIST} ${GLAD_SOURCE_LIST} ${CUB_SOURCE_LIST})
add_library(gliloader MODULE ${MUFFLON_GLILOADER_SOURCE_LIST} ${EPSILON_SOURCE_LIST})
add_library(mffloader ${MUFFLON_MFFLOADER_SOURCE_LIST} ${CHINOISE_SOURCE_LIST} ${EPSILON_SOURCE_LIST} ${MINIZ_SOURCE_LIST})
add_library(pfmloader MODULE ${MUFFLON_PFMLOADER_SOURCE_LIST} ${EPSILON_SOURCE_LIST})
add_library(stbiloader MODULE ${MUFFLON_STBILOADER_SOURCE_LIST} ${EPSILON_SOURCE_LIST} ${STBI_SOURCE_LIST})
# Dependency library targets
add_library(openmesh_core SHARED IMPORTED)
add_library(openmesh_tools SHARED IMPORTED)
if(MUFFLON_ENABLE_OPEN_DENOISE)
	add_library(opendenoise SHARED IMPORTED)
	add_library(tbb SHARED IMPORTED)
	add_library(tbbmalloc SHARED IMPORTED)
endif()

# Check for link-time optimization support
include(CheckIPOSupported)
check_ipo_supported(RESULT LTO_SUPPORTED OUTPUT error)
if(LTO_SUPPORTED)
	message(STATUS "LTO supported and enabled")
	set_property(TARGET core PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
	set_property(TARGET mffloader PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
else()
	message(STATUS "LTO not supported")
endif()

# Set the libraries' headers for inclusion
set_target_properties(core PROPERTIES PUBLIC_HEADER
	"${CMAKE_CURRENT_SOURCE_DIR}/src/core/export/core_interface.h;${CMAKE_CURRENT_SOURCE_DIR}/src/core/export/core_api.h;${CMAKE_CURRENT_SOURCE_DIR}/src/core/export/texture_data.h")
set_target_properties(mffloader PROPERTIES PUBLIC_HEADER
	"${CMAKE_CURRENT_SOURCE_DIR}/src/mffloader/interface/mff_interface.h;${CMAKE_CURRENT_SOURCE_DIR}/src/mffloader/interface/mff_api.h")

# Set the include directories
target_include_directories(core
	PRIVATE
		${CMAKE_CURRENT_SOURCE_DIR}/src
		${CMAKE_CURRENT_SOURCE_DIR}/src/core/export
		${CMAKE_CURRENT_SOURCE_DIR}/deps
		${CMAKE_CURRENT_SOURCE_DIR}/deps/chinoise/include
		${CMAKE_CURRENT_SOURCE_DIR}/deps/cub
		${CMAKE_CURRENT_SOURCE_DIR}/deps/epsilon/include
		${CMAKE_CURRENT_SOURCE_DIR}/deps/glad
		${CMAKE_CURRENT_SOURCE_DIR}/external/OpenMesh-8.0/$<$<CONFIG:Debug>:Debug>$<$<NOT:$<CONFIG:Debug>>:Release>/include
		${CMAKE_CURRENT_SOURCE_DIR}/external/oidn-0.9/$<$<CONFIG:Debug>:Debug>$<$<NOT:$<CONFIG:Debug>>:Release>/include
)
target_include_directories(gliloader
	PRIVATE
		${CMAKE_CURRENT_SOURCE_DIR}/src
		${CMAKE_CURRENT_SOURCE_DIR}/deps
		${CMAKE_CURRENT_SOURCE_DIR}/deps/epsilon/include
		${CMAKE_CURRENT_SOURCE_DIR}/deps/gli
		${CMAKE_CURRENT_SOURCE_DIR}/deps/gli/external
)
target_include_directories(mffloader
	PRIVATE
		${CMAKE_CURRENT_SOURCE_DIR}/src
		${CMAKE_CURRENT_SOURCE_DIR}/src/core/export
		${CMAKE_CURRENT_SOURCE_DIR}/deps
		${CMAKE_CURRENT_SOURCE_DIR}/deps/chinoise/include
		${CMAKE_CURRENT_SOURCE_DIR}/deps/epsilon/include
		${CMAKE_CURRENT_SOURCE_DIR}/deps/miniz
		${CMAKE_CURRENT_SOURCE_DIR}/external/rapidjson-1.1.0/$<$<CONFIG:Debug>:Debug>$<$<NOT:$<CONFIG:Debug>>:Release>/include
		${CMAKE_CURRENT_SOURCE_DIR}/src/core/export
)
target_include_directories(pfmloader
	PRIVATE
		${CMAKE_CURRENT_SOURCE_DIR}/src
		${CMAKE_CURRENT_SOURCE_DIR}/deps
		${CMAKE_CURRENT_SOURCE_DIR}/deps/epsilon/include
)
target_include_directories(stbiloader
	PRIVATE
		${CMAKE_CURRENT_SOURCE_DIR}/src
		${CMAKE_CURRENT_SOURCE_DIR}/deps
		${CMAKE_CURRENT_SOURCE_DIR}/deps/stbi
		${CMAKE_CURRENT_SOURCE_DIR}/deps/epsilon/include
)

# Set the target definitions for DLL export
target_compile_definitions(core PRIVATE _USE_MATH_DEFINES CORE_EXPORTS _CRT_SECURE_NO_WARNINGS)
target_compile_definitions(mffloader PRIVATE LOADER_EXPORTS _CRT_SECURE_NO_WARNINGS)
target_compile_definitions(stbiloader PRIVATE _CRT_SECURE_NO_WARNINGS)
# Set the compile definition for min. CC requirement
target_compile_definitions(core PRIVATE MUFFLON_CUDA_ARCHES="${MUFFLON_CUDA_ARCH}")

set(CXX_TARGETS core gliloader mffloader pfmloader stbiloader)

# Set C++ standard and CUDA flags
set_target_properties(core PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(core PROPERTIES CUDA_STANDARD 14)
set_target_properties(core PROPERTIES CUDA_STANDARD_REQUIRED ON)
target_compile_options(core PRIVATE
	$<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

set_target_properties(${CXX_TARGETS}
	PROPERTIES
		CXX_STANDARD 17
		CXX_STANDARD_REQUIRED ON
		CXX_EXTENSIONS OFF
		CUDA_SEPARABLE_COMPILATION ON
		CUDA_STANDARD 14
		CUDA_STANDARD_REQUIRED ON
)

# Set compiler-specific flags
if (MSVC)
	# First remove the default warning level for MSVC (I really hope CMake introduces a better abstraction one day...)
	if(CMAKE_CXX_FLAGS MATCHES "/W[0-4]")
		string(REGEX REPLACE "/W[0-4]" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
	endif()
	
	foreach(TAR ${CXX_TARGETS})
		# Enable warnings and treat them as errors
		# Enable parallel build
		# Enable unicode, remove windows header minmax etc.
		if(MUFFLON_STRICT_COMPILATION)
			target_compile_options(${TAR} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/W4> $<$<COMPILE_LANGUAGE:CXX>:/WX>)
		else(MUFFLON_STRICT_COMPILATION)
			target_compile_options(${TAR} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/W3>)
		endif(MUFFLON_STRICT_COMPILATION)
		target_compile_options(${TAR} PUBLIC $<$<COMPILE_LANGUAGE:CXX>:/MP>)
		target_compile_definitions(${TAR} PUBLIC NOMINMAX UNICODE STRICT)
	endforeach(TAR)
else()
	foreach(TAR ${CXX_TARGETS})
		# Enable warnings and treat them as errors
		set(MUFFLON_WARNING_FLAGS -Wall -Wextra -Wno-comment -Wno-unknown-pragmas -Wno-class-memaccess -Wno-unused-parameter
		-Wno-unused-variable CACHE STRING "Standard warning flags")
		if(MUFFLON_STRICT_COMPILATION)
			target_compile_options(${TAR} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${MUFFLON_WARNING_FLAGS} -Werror>)
		else(MUFFLON_STRICT_COMPILATION)
			target_compile_options(${TAR} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${MUFFLON_WARNING_FLAGS}>)
		endif(MUFFLON_STRICT_COMPILATION)
	endforeach(TAR)
	# On older GCC versions, we need to link in a library to get filesystem support (also to avoid getting NASTY runtime bugs due to incompatibilities between libstdc++ and gcc!)
	set(CXX_FS_LIBRARY "stdc++fs")
	if("${MUFFLON_ENABLE_SANITIZER}" STREQUAL "ADDRESS")
		target_compile_options(core PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-fsanitize=address>)
	elseif("${MUFFLON_ENABLE_SANITIZER}" STREQUAL "LEAK")
	target_compile_options(core PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-fsanitize=leak>)
	elseif("${MUFFLON_ENABLE_SANITIZER}" STREQUAL "THREAD")
	target_compile_options(core PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-fsanitize=thread>)
	elseif("${MUFFLON_ENABLE_SANITIZER}" STREQUAL "UB")
	target_compile_options(core PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-fsanitize=undefined>)
	endif()
endif()
# Since GLI has some warnings themselves that we can't (or won't) fix, we just ignore them
if(NOT MSVC)
	target_compile_options(gliloader PRIVATE
		-Wno-ignored-qualifiers -Wno-unused-parameter -Wno-empty-body
		-Wno-unused-result -Wno-unused-variable -Wno-type-limits
	)
endif()

# Dependency library targets
set(MUFFLON_OPENMESH_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/external/OpenMesh-8.0")
set(MUFFLON_OPENDENOISE_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/external/oidn-0.9")
set(MUFFLON_TBB_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/external/tbb-2019.9")
if(MSVC)
	set_target_properties(openmesh_core PROPERTIES IMPORTED_LOCATION_DEBUG "${MUFFLON_OPENMESH_DIRECTORY}/Debug/bin/OpenMeshCored.dll")
	set_target_properties(openmesh_core PROPERTIES IMPORTED_LOCATION_RELEASE "${MUFFLON_OPENMESH_DIRECTORY}/Release/bin/OpenMeshCore.dll")
	set_target_properties(openmesh_core PROPERTIES IMPORTED_LOCATION_RELWITHDEBINFO "${MUFFLON_OPENMESH_DIRECTORY}/Release/bin/OpenMeshCore.dll")
	set_target_properties(openmesh_core PROPERTIES IMPORTED_LOCATION_MINSIZEREL "${MUFFLON_OPENMESH_DIRECTORY}/Release/bin/OpenMeshCore.dll")
	set_target_properties(openmesh_core PROPERTIES IMPORTED_IMPLIB_DEBUG "${MUFFLON_OPENMESH_DIRECTORY}/Debug/lib/OpenMeshCored.lib")
	set_target_properties(openmesh_core PROPERTIES IMPORTED_IMPLIB_RELEASE "${MUFFLON_OPENMESH_DIRECTORY}/Release/lib/OpenMeshCore.lib")
	set_target_properties(openmesh_core PROPERTIES IMPORTED_IMPLIB_RELWITHDEBINFO "${MUFFLON_OPENMESH_DIRECTORY}/Release/lib/OpenMeshCore.lib")
	set_target_properties(openmesh_core PROPERTIES IMPORTED_IMPLIB_MINSIZEREL "${MUFFLON_OPENMESH_DIRECTORY}/Release/lib/OpenMeshCore.lib")
	set_target_properties(openmesh_tools PROPERTIES IMPORTED_LOCATION_DEBUG "${MUFFLON_OPENMESH_DIRECTORY}/Debug/bin/OpenMeshToolsd.dll")
	set_target_properties(openmesh_tools PROPERTIES IMPORTED_LOCATION_RELEASE "${MUFFLON_OPENMESH_DIRECTORY}/Release/bin/OpenMeshTools.dll")
	set_target_properties(openmesh_tools PROPERTIES IMPORTED_LOCATION_RELWITHDEBINFO "${MUFFLON_OPENMESH_DIRECTORY}/Release/bin/OpenMeshTools.dll")
	set_target_properties(openmesh_tools PROPERTIES IMPORTED_LOCATION_MINSIZEREL "${MUFFLON_OPENMESH_DIRECTORY}/Release/bin/OpenMeshTools.dll")
	set_target_properties(openmesh_tools PROPERTIES IMPORTED_IMPLIB_DEBUG "${MUFFLON_OPENMESH_DIRECTORY}/Debug/lib/OpenMeshToolsd.lib")
	set_target_properties(openmesh_tools PROPERTIES IMPORTED_IMPLIB_RELEASE "${MUFFLON_OPENMESH_DIRECTORY}/Release/lib/OpenMeshTools.lib")
	set_target_properties(openmesh_tools PROPERTIES IMPORTED_IMPLIB_RELWITHDEBINFO "${MUFFLON_OPENMESH_DIRECTORY}/Release/lib/OpenMeshTools.lib")
	set_target_properties(openmesh_tools PROPERTIES IMPORTED_IMPLIB_MINSIZEREL "${MUFFLON_OPENMESH_DIRECTORY}/Release/lib/OpenMeshTools.lib")
	install(FILES
			"${MUFFLON_OPENMESH_DIRECTORY}/$<$<CONFIG:Debug>:Debug>$<$<NOT:$<CONFIG:Debug>>:Release>/bin/OpenMeshCore$<$<CONFIG:Debug>:d>.dll"
			"${MUFFLON_OPENMESH_DIRECTORY}/$<$<CONFIG:Debug>:Debug>$<$<NOT:$<CONFIG:Debug>>:Release>/bin/OpenMeshTools$<$<CONFIG:Debug>:d>.dll"
		DESTINATION "."
	)
	if(MUFFLON_ENABLE_OPEN_DENOISE)
		set_target_properties(opendenoise PROPERTIES IMPORTED_IMPLIB_DEBUG "${MUFFLON_OPENDENOISE_DIRECTORY}/Debug/lib/OpenImageDenoise.lib")
		set_target_properties(opendenoise PROPERTIES IMPORTED_IMPLIB_RELEASE "${MUFFLON_OPENDENOISE_DIRECTORY}/Release/lib/OpenImageDenoise.lib")
		set_target_properties(opendenoise PROPERTIES IMPORTED_IMPLIB_RELWITHDEBINFO "${MUFFLON_OPENDENOISE_DIRECTORY}/Release/lib/OpenImageDenoise.lib")
		set_target_properties(opendenoise PROPERTIES IMPORTED_IMPLIB_MINSIZEREL "${MUFFLON_OPENDENOISE_DIRECTORY}/Release/lib/OpenImageDenoise.lib")
		set_target_properties(opendenoise PROPERTIES IMPORTED_LOCATION_DEBUG "${MUFFLON_OPENDENOISE_DIRECTORY}/Debug/bin/OpenImageDenoise.dll")
		set_target_properties(opendenoise PROPERTIES IMPORTED_LOCATION_RELEASE "${MUFFLON_OPENDENOISE_DIRECTORY}/Release/bin/OpenImageDenoise.dll")
		set_target_properties(opendenoise PROPERTIES IMPORTED_LOCATION_RELWITHDEBINFO "${MUFFLON_OPENDENOISE_DIRECTORY}/Release/bin/OpenImageDenoise.dll")
		set_target_properties(opendenoise PROPERTIES IMPORTED_LOCATION_MINSIZEREL "${MUFFLON_OPENDENOISE_DIRECTORY}/Release/bin/OpenImageDenoise.dll")
		set_target_properties(tbb PROPERTIES IMPORTED_IMPLIB_DEBUG "${MUFFLON_TBB_DIRECTORY}/Debug/lib/tbb.lib")
		set_target_properties(tbb PROPERTIES IMPORTED_IMPLIB_RELEASE "${MUFFLON_TBB_DIRECTORY}/Release/lib/tbb.lib")
		set_target_properties(tbb PROPERTIES IMPORTED_IMPLIB_RELWITHDEBINFO "${MUFFLON_TBB_DIRECTORY}/Release/lib/tbb.lib")
		set_target_properties(tbb PROPERTIES IMPORTED_IMPLIB_MINSIZEREL "${MUFFLON_TBB_DIRECTORY}/Release/lib/tbb.lib")
		set_target_properties(tbb PROPERTIES IMPORTED_LOCATION_DEBUG "${MUFFLON_TBB_DIRECTORY}/Debug/bin/tbb.dll")
		set_target_properties(tbb PROPERTIES IMPORTED_LOCATION_RELEASE "${MUFFLON_TBB_DIRECTORY}/Release/bin/tbb.dll")
		set_target_properties(tbb PROPERTIES IMPORTED_LOCATION_RELWITHDEBINFO "${MUFFLON_TBB_DIRECTORY}/Release/bin/tbb.dll")
		set_target_properties(tbb PROPERTIES IMPORTED_LOCATION_MINSIZEREL "${MUFFLON_TBB_DIRECTORY}/Release/bin/tbb.dll")
		set_target_properties(tbbmalloc PROPERTIES IMPORTED_IMPLIB_DEBUG "${MUFFLON_TBB_DIRECTORY}/Debug/lib/tbbmalloc.lib")
		set_target_properties(tbbmalloc PROPERTIES IMPORTED_IMPLIB_RELEASE "${MUFFLON_TBB_DIRECTORY}/Release/lib/tbbmalloc.lib")
		set_target_properties(tbbmalloc PROPERTIES IMPORTED_IMPLIB_RELWITHDEBINFO "${MUFFLON_TBB_DIRECTORY}/Release/lib/tbbmalloc.lib")
		set_target_properties(tbbmalloc PROPERTIES IMPORTED_IMPLIB_MINSIZEREL "${MUFFLON_TBB_DIRECTORY}/Release/lib/tbbmalloc.lib")
		set_target_properties(tbbmalloc PROPERTIES IMPORTED_LOCATION_DEBUG "${MUFFLON_TBB_DIRECTORY}/Debug/bin/tbbmalloc.dll")
		set_target_properties(tbbmalloc PROPERTIES IMPORTED_LOCATION_RELEASE "${MUFFLON_TBB_DIRECTORY}/Release/bin/tbbmalloc.dll")
		set_target_properties(tbbmalloc PROPERTIES IMPORTED_LOCATION_RELWITHDEBINFO "${MUFFLON_TBB_DIRECTORY}/Release/bin/tbbmalloc.dll")
		set_target_properties(tbbmalloc PROPERTIES IMPORTED_LOCATION_MINSIZEREL "${MUFFLON_TBB_DIRECTORY}/Release/bin/tbbmalloc.dll")
		install(FILES
				"${MUFFLON_OPENDENOISE_DIRECTORY}/$<$<CONFIG:Debug>:Debug>$<$<NOT:$<CONFIG:Debug>>:Release>/bin/OpenImageDenoise.dll"
				"${MUFFLON_TBB_DIRECTORY}/$<$<CONFIG:Debug>:Debug>$<$<NOT:$<CONFIG:Debug>>:Release>/bin/tbb.dll"
				"${MUFFLON_TBB_DIRECTORY}/$<$<CONFIG:Debug>:Debug>$<$<NOT:$<CONFIG:Debug>>:Release>/bin/tbbmalloc.dll"
			DESTINATION "."
		)
	endif()
else(MSVC)
	set_target_properties(openmesh_core PROPERTIES IMPORTED_LOCATION_DEBUG "${MUFFLON_OPENMESH_DIRECTORY}/Debug/lib/libOpenMeshCored.so.8.0")
	set_target_properties(openmesh_core PROPERTIES IMPORTED_LOCATION_RELEASE "${MUFFLON_OPENMESH_DIRECTORY}/Release/lib/libOpenMeshCore.so.8.0")
	set_target_properties(openmesh_core PROPERTIES IMPORTED_LOCATION_RELWITHDEBINFO "${MUFFLON_OPENMESH_DIRECTORY}/Release/lib/libOpenMeshCore.so.8.0")
	set_target_properties(openmesh_core PROPERTIES IMPORTED_LOCATION_MINSIZEREL "${MUFFLON_OPENMESH_DIRECTORY}/Release/lib/libOpenMeshCore.so.8.0")
	set_target_properties(openmesh_tools PROPERTIES IMPORTED_LOCATION_DEBUG "${MUFFLON_OPENMESH_DIRECTORY}/Debug/lib/libOpenMeshToolsd.so.8.0")
	set_target_properties(openmesh_tools PROPERTIES IMPORTED_LOCATION_RELEASE "${MUFFLON_OPENMESH_DIRECTORY}/Release/lib/libOpenMeshTools.so.8.0")
	set_target_properties(openmesh_tools PROPERTIES IMPORTED_LOCATION_RELWITHDEBINFO "${MUFFLON_OPENMESH_DIRECTORY}/Release/lib/libOpenMeshTools.so.8.0")
	set_target_properties(openmesh_tools PROPERTIES IMPORTED_LOCATION_MINSIZEREL "${MUFFLON_OPENMESH_DIRECTORY}/Release/lib/libOpenMeshTools.so.8.0")
	install(FILES
			"${MUFFLON_OPENMESH_DIRECTORY}/$<$<CONFIG:Debug>:Debug>$<$<NOT:$<CONFIG:Debug>>:Release>/lib/libOpenMeshCore$<$<CONFIG:Debug>:d>.so.8.0"
			"${MUFFLON_OPENMESH_DIRECTORY}/$<$<CONFIG:Debug>:Debug>$<$<NOT:$<CONFIG:Debug>>:Release>/lib/libOpenMeshTools$<$<CONFIG:Debug>:d>.so.8.0"
		DESTINATION "."
	)
	if(MUFFLON_ENABLE_OPEN_DENOISE)
		set_target_properties(opendenoise PROPERTIES IMPORTED_LOCATION_DEBUG "${MUFFLON_OPENDENOISE_DIRECTORY}/Debug/lib/libOpenImageDenoise.so.0")
		set_target_properties(opendenoise PROPERTIES IMPORTED_LOCATION_RELEASE "${MUFFLON_OPENDENOISE_DIRECTORY}/Release/lib/libOpenImageDenoise.so.0")
		set_target_properties(opendenoise PROPERTIES IMPORTED_LOCATION_RELWITHDEBINFO "${MUFFLON_OPENDENOISE_DIRECTORY}/Release/lib/libOpenImageDenoise.so.0")
		set_target_properties(opendenoise PROPERTIES IMPORTED_LOCATION_MINSIZEREL "${MUFFLON_OPENDENOISE_DIRECTORY}/Release/lib/libOpenImageDenoise.so.0")
		set_target_properties(tbb PROPERTIES IMPORTED_LOCATION_DEBUG "${MUFFLON_TBB_DIRECTORY}/Debug/lib/libtbb.so.2")
		set_target_properties(tbb PROPERTIES IMPORTED_LOCATION_RELEASE "${MUFFLON_TBB_DIRECTORY}/Debug/Release/libtbb.so.2")
		set_target_properties(tbb PROPERTIES IMPORTED_LOCATION_RELWITHDEBINFO "${MUFFLON_TBB_DIRECTORY}/Debug/Release/libtbb.so.2")
		set_target_properties(tbb PROPERTIES IMPORTED_LOCATION_MINSIZEREL "${MUFFLON_TBB_DIRECTORY}/Debug/Release/libtbb.so.2")
		set_target_properties(tbbmalloc PROPERTIES IMPORTED_LOCATION_DEBUG "${MUFFLON_TBB_DIRECTORY}/Debug/lib/libtbbmalloc.so.2")
		set_target_properties(tbbmalloc PROPERTIES IMPORTED_LOCATION_RELEASE "${MUFFLON_TBB_DIRECTORY}/Release/lib/libtbbmalloc.so.2")
		set_target_properties(tbbmalloc PROPERTIES IMPORTED_LOCATION_RELWITHDEBINFO "${MUFFLON_TBB_DIRECTORY}/Release/lib/libtbbmalloc.so.2")
		set_target_properties(tbbmalloc PROPERTIES IMPORTED_LOCATION_MINSIZEREL "${MUFFLON_TBB_DIRECTORY}/Release/lib/libtbbmalloc.so.2")
		install(FILES
				"${MUFFLON_OPENDENOISE_DIRECTORY}/$<$<CONFIG:Debug>:Debug>$<$<NOT:$<CONFIG:Debug>>:Release>/lib/libOpenImageDenoise.so.0"
				"${MUFFLON_TBB_DIRECTORY}/$<$<CONFIG:Debug>:Debug>$<$<NOT:$<CONFIG:Debug>>:Release>/lib/libtbb.so.2"
				"${MUFFLON_TBB_DIRECTORY}/$<$<CONFIG:Debug>:Debug>$<$<NOT:$<CONFIG:Debug>>:Release>/lib/libtbbmalloc.so.2"
			DESTINATION "."
		)
	endif()
endif()

# GUI building
if(MUFFLON_BUILD_GUI)
	include(CSharpUtilities)
	
	add_executable(gui ${MUFFLON_GUI_SOURCE_LIST})

	unset(MUFFLON_GUI_CS_LIST)
	foreach(CS_XAML_FILE ${MUFFLON_GUI_CS_XAML_LIST})
		string(REGEX REPLACE "\\.cs$" "" XAML_FILE ${CS_XAML_FILE})
		list(APPEND MUFFLON_GUI_CS_LIST "${XAML_FILE}")
		list(APPEND MUFFLON_GUI_CS_LIST "${CS_XAML_FILE}")
	endforeach(CS_XAML_FILE)

	file(GLOB_RECURSE GUI_PROPERTY_FILES "${CMAKE_CURRENT_SOURCE_DIR}/project_files/gui/Properties/*")
	csharp_set_xaml_cs_properties(${MUFFLON_GUI_CS_LIST})
	csharp_set_designer_cs_properties(${GUI_PROPERTY_FILES})
	
	# Icons
	set_source_files_properties(${MUFFLON_GUI_ICONS}
		PROPERTIES VS_TOOL_OVERRIDE "Resource")


	set_property(SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/project_files/gui/App.xaml" PROPERTY VS_XAML_TYPE "ApplicationDefinition")
	set_property(TARGET gui PROPERTY VS_DOTNET_TARGET_FRAMEWORK_VERSION "v${MUFFLON_DotNet_VERSION}")
	set_property(TARGET gui PROPERTY WIN32_EXECUTABLE TRUE)
	set_property(TARGET gui PROPERTY VS_DOTNET_REFERENCES
		"Microsoft.CSharp"
		"PresentationCore"
		"PresentationFramework"
		"System"
		"System.Core"
		"System.Data"
		"System.Data.DataSetExtensions"
		"System.Drawing"
		"System.Net.Http"
		"System.Windows.Forms"
		"System.Xaml"
		"System.Xml"
		"System.Xml.Linq"
		"WindowsBase"
	)
	set_property(TARGET gui PROPERTY VS_GLOBAL_ApplicationIcon "${CMAKE_CURRENT_SOURCE_DIR}/project_files/gui/Icons/logo1.ico")
	set_target_properties(gui PROPERTIES OUTPUT_NAME "MufflonGUI")

	# Set the language version, which for whatever reason does not have its own property
	# Regex matches for either a version number, latest, or default
	# Unfortunately we can't properly set the language version, so we rely on the default being high enough
	string(REGEX REPLACE "/langversion:.*$" "" CMAKE_CSharp_FLAGS "${CMAKE_CSharp_FLAGS}")
	string(REGEX REPLACE "/langversion:.* " "" CMAKE_CSharp_FLAGS "${CMAKE_CSharp_FLAGS}")
	
	# This solution is taken from https://stackoverflow.com/questions/18132987/cmake-and-msvs-nuget
	find_program(MUFFLON_NUGET nuget)
	if(NOT MUFFLON_NUGET)
		message(WARNING "CMake could not find the nuget command line tool. Please manually specify its location in MUFFLON_NUGET, or you'll have to manually isntall/restore NuGet packages!")
	else()
		# Copy the Nuget config file from source location to the CMake build directory.
		configure_file("${CMAKE_CURRENT_SOURCE_DIR}/project_files/gui/packages.config" packages.config COPYONLY)
		# Run Nuget using the .config file to install any missing dependencies to the build directory.
		execute_process(COMMAND 
			${MUFFLON_NUGET} restore packages.config -SolutionDirectory ${CMAKE_BINARY_DIR}
			WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
		)
	endif()
	# Provide the path to the Nuget-installed references, in case this is a fresh project build.
	set(MUFFLON_JSON_DLL "${CMAKE_BINARY_DIR}/packages/Newtonsoft.Json.12.0.2/lib/net45/Newtonsoft.Json.dll")
	set_property(TARGET gui PROPERTY 
		VS_DOTNET_REFERENCE_Newtonsoft.Json "${MUFFLON_JSON_DLL}"
	)

	target_link_libraries(gui
		PUBLIC
			core
			mffloader
	)
	install(TARGETS gui
		ARCHIVE DESTINATION "."
		RUNTIME DESTINATION "."
		LIBRARY DESTINATION "."
	)
	install(FILES "${MUFFLON_JSON_DLL}" DESTINATION ".")
endif(MUFFLON_BUILD_GUI)

# Inter-target dependencies
target_link_libraries(mffloader PRIVATE ${CXX_FS_LIBRARY} core)
target_link_libraries(core PRIVATE ${CXX_FS_LIBRARY} openmesh_core openmesh_tools)
if(MUFFLON_ENABLE_OPEN_DENOISE)
	target_link_libraries(core PRIVATE opendenoise tbb)
	set_property(SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/src/core/export/interface.cpp"
		APPEND
		PROPERTY COMPILE_DEFINITIONS
		MUFFLON_ENABLE_OPEN_DENOISE
	)
endif()
if(OpenMP_CXX_FOUND)
	target_link_libraries(core PRIVATE OpenMP::OpenMP_CXX)
	target_link_libraries(gliloader PRIVATE OpenMP::OpenMP_CXX)
	target_link_libraries(stbiloader PRIVATE OpenMP::OpenMP_CXX)
	target_link_libraries(pfmloader PRIVATE OpenMP::OpenMP_CXX)
endif()

# Set output targets for plugins
set_target_properties(gliloader pfmloader stbiloader PROPERTIES
	ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/build/$<CONFIG>/plugins"
	LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/build/$<CONFIG>/plugins"
	RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/build/$<CONFIG>/plugins"
)

# Add python bindings as custom target and install thempython bindings
add_custom_target(pythonbindings
	SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/pythonbinding/bindings.py"
	DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/pythonbinding/bindings.py"
)
add_custom_command(TARGET pythonbindings PRE_BUILD
	COMMAND ${CMAKE_COMMAND} -E copy_if_different
		"${CMAKE_CURRENT_SOURCE_DIR}/src/pythonbinding/bindings.py"
		"${CMAKE_BINARY_DIR}/build/$<CONFIG>"
)
add_dependencies(core pythonbindings)

# Copy the shaders and dependencies into the build folder
# We could copy the shaders in one go, but on Windows there is a limit for the number of characters
# a single command may have, so a loop is the safer call
foreach(shader ${MUFFLON_SHADER_SOURCE_LIST})
	get_filename_component(shader_name ${shader} NAME)
	add_custom_command(TARGET core PRE_BUILD
		COMMAND ${CMAKE_COMMAND} -E copy_if_different
			"${shader}"
			"${CMAKE_BINARY_DIR}/build/$<CONFIG>/shader/${shader_name}"
	)
endforeach()

if(WIN32)
	add_custom_command(TARGET core PRE_BUILD
		COMMAND ${CMAKE_COMMAND} -E copy_if_different
			"${MUFFLON_OPENMESH_DIRECTORY}/$<$<CONFIG:Debug>:Debug>$<$<NOT:$<CONFIG:Debug>>:Release>/bin/OpenMeshCore$<$<CONFIG:Debug>:d>.dll"
			"${MUFFLON_OPENMESH_DIRECTORY}/$<$<CONFIG:Debug>:Debug>$<$<NOT:$<CONFIG:Debug>>:Release>/bin/OpenMeshTools$<$<CONFIG:Debug>:d>.dll"
			"${MUFFLON_OPENDENOISE_DIRECTORY}/$<$<CONFIG:Debug>:Debug>$<$<NOT:$<CONFIG:Debug>>:Release>/bin/OpenImageDenoise.dll"
			"${CMAKE_BINARY_DIR}/build/$<CONFIG>"
	)
	if(MUFFLON_ENABLE_OPEN_DENOISE)
		add_custom_command(TARGET core PRE_BUILD
			COMMAND ${CMAKE_COMMAND} -E copy_if_different
				"${MUFFLON_OPENDENOISE_DIRECTORY}/$<$<CONFIG:Debug>:Debug>$<$<NOT:$<CONFIG:Debug>>:Release>/bin/OpenImageDenoise.dll"
				"${MUFFLON_TBB_DIRECTORY}/$<$<CONFIG:Debug>:Debug>$<$<NOT:$<CONFIG:Debug>>:Release>/bin/tbb.dll"
				"${MUFFLON_TBB_DIRECTORY}/$<$<CONFIG:Debug>:Debug>$<$<NOT:$<CONFIG:Debug>>:Release>/bin/tbbmalloc.dll"
				"${CMAKE_BINARY_DIR}/build/$<CONFIG>"
		)
	endif()
else()
	add_custom_command(TARGET core PRE_BUILD
		COMMAND ${CMAKE_COMMAND} -E copy_if_different
			"${MUFFLON_OPENMESH_DIRECTORY}/$<$<CONFIG:Debug>:Debug>$<$<NOT:$<CONFIG:Debug>>:Release>/lib/libOpenMeshCore$<$<CONFIG:Debug>:d>.so.8.0"
			"${MUFFLON_OPENMESH_DIRECTORY}/$<$<CONFIG:Debug>:Debug>$<$<NOT:$<CONFIG:Debug>>:Release>/lib/libOpenMeshTools$<$<CONFIG:Debug>:d>.so.8.0"
			"${CMAKE_BINARY_DIR}/build/$<CONFIG>"
	)
	if(MUFFLON_ENABLE_OPEN_DENOISE)
		add_custom_command(TARGET core PRE_BUILD
			COMMAND ${CMAKE_COMMAND} -E copy_if_different
				"${MUFFLON_OPENDENOISE_DIRECTORY}/$<CONFIG>/lib/libOpenImageDenoise.so.0"
				"${MUFFLON_TBB_DIRECTORY}/$<$<CONFIG:Debug>:Debug>$<$<NOT:$<CONFIG:Debug>>:Release>/lib/libtbb.so.2"
				"${MUFFLON_TBB_DIRECTORY}/$<$<CONFIG:Debug>:Debug>$<$<NOT:$<CONFIG:Debug>>:Release>/lib/libtbbmalloc.so.2"
				"${CMAKE_BINARY_DIR}/build/$<CONFIG>"
		)
	endif()
endif()

# Install shared libraries, dependencies, and shaders
install(TARGETS core mffloader
	ARCHIVE DESTINATION "."
	RUNTIME DESTINATION "."
	LIBRARY DESTINATION "."
	PUBLIC_HEADER DESTINATION "include"
)
install(TARGETS stbiloader pfmloader gliloader
	ARCHIVE DESTINATION "."
	RUNTIME DESTINATION "."
	LIBRARY DESTINATION "plugins"
)
install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/src/pythonbinding/bindings.py"
	DESTINATION ".")
install(FILES ${MUFFLON_SHADER_SOURCE_LIST}
	DESTINATION "shader")
