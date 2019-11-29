cmake_minimum_required(VERSION 3.12)
project("mufflon" VERSION 0.0.1 LANGUAGES C CXX CUDA)

# Find OpenMP for linking
find_package(OpenMP)
find_package(CUDA REQUIRED)

# Force 'Release' as build type if no other has been specified
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to 'Release' as none was specified.")
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release"
    "MinSizeRel" "RelWithDebInfo")
endif()

# Give user the option to disable folders for MSVC solutions
option(MUFFLON_USE_FOLDERS "Enable solution folders in Visual Studio" ON)
if(MUFFLON_USE_FOLDERS)
	set_property(GLOBAL PROPERTY USE_FOLDERS ON)
endif()

# Option to be more strict on compilation
option(MUFFLON_STRICT_COMPILATION "Enable W4/Wextra and WX/Werror" OFF)
# CUDA arch
set(ARCH "Auto" CACHE STRING "CUDA target compute capability (Auto uses present GPUs")

# Select the compilation target for NVCC in terms of compute capability
CUDA_SELECT_NVCC_ARCH_FLAGS(ARCH_FLAGS ${ARCH})
if("${ARCH}" STREQUAL "Auto")
	STRING(REGEX REPLACE "sm_(.)(.)" "\\1.\\2" ARCH_CC ${ARCH_FLAGS_readable})
	set(ARCH "${ARCH_CC}" CACHE STRING "CUDA target compute capability (Auto uses present GPUs)" FORCE)
	message(STATUS "Detected CUDA arch: ${ARCH}")
endif()

if(MSVC)
	enable_language(CSharp)
	# Option to not build GUI
	option(MUFFLON_BUILD_GUI "Build the GUI or not" ON)
	# C# version
	set(MUFFLON_DotNet_VERSION "4.6.1" CACHE STRING ".NET version to use")
	endif()

# Discourage in-source builds
if("${PROJECT_SOURCE_DIR}" STREQUAL "${PROJECT_BINARY_DIR}")
   message(WARNING "In-source building is discouraged!")
endif("${PROJECT_SOURCE_DIR}" STREQUAL "${PROJECT_BINARY_DIR}")

# Ensure that our binaries get put in the proper directory
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/build")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_DEBUG "${CMAKE_BINARY_DIR}/build/Debug")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}/build/Release")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/build")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_DEBUG "${CMAKE_BINARY_DIR}/build/Debug")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}/build/Release")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/build")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_BINARY_DIR}/build/Debug")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}/build/Release")

# Store our source files in a variable to auto-gen source groups
file(GLOB_RECURSE MUFFLON_CORE_SOURCE_LIST CONFIGURE_DEPENDS
	"${CMAKE_CURRENT_SOURCE_DIR}/src/core/*"
	"${CMAKE_CURRENT_SOURCE_DIR}/src/plugin/*"
	"${CMAKE_CURRENT_SOURCE_DIR}/src/profiler/*"
	"${CMAKE_CURRENT_SOURCE_DIR}/src/util/*"
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
source_group(deps\\chinoise FILES ${CHINOISE_SOURCE_LIST})
source_group(deps\\cub FILES ${CUB_SOURCE_LIST})
source_group(deps\\epsilon FILES ${EPSILON_SOURCE_LIST})
source_group(deps\\glad FILES ${GLAD_SOURCE_LIST})
source_group(deps\\gli FILES ${GLI_SOURCE_LIST})
source_group(deps\\miniz FILES ${MINIZ_SOURCE_LIST})
source_group(deps\\stbi FILES ${STBI_SOURCE_LIST})
source_group(bindings FILES "${CMAKE_CURRENT_SOURCE_DIR}/src/pythonbinding/bindings.py")

# Build artifacts (we include the deps sources to get them to show up in visual studio)
add_library(core SHARED ${MUFFLON_CORE_SOURCE_LIST} ${CHINOISE_SOURCE_LIST} ${EPSILON_SOURCE_LIST} ${GLAD_SOURCE_LIST} ${CUB_SOURCE_LIST})
add_library(gliloader MODULE ${MUFFLON_GLILOADER_SOURCE_LIST} ${EPSILON_SOURCE_LIST})
add_library(mffloader SHARED ${MUFFLON_MFFLOADER_SOURCE_LIST} ${CHINOISE_SOURCE_LIST} ${EPSILON_SOURCE_LIST} ${MINIZ_SOURCE_LIST})
add_library(pfmloader MODULE ${MUFFLON_PFMLOADER_SOURCE_LIST} ${EPSILON_SOURCE_LIST})
add_library(stbiloader MODULE ${MUFFLON_STBILOADER_SOURCE_LIST} ${EPSILON_SOURCE_LIST} ${STBI_SOURCE_LIST})
# TODO: GUI and python script

# Set the include directories
target_include_directories(core
	PRIVATE
		${CMAKE_CURRENT_SOURCE_DIR}/src
		${CMAKE_CURRENT_SOURCE_DIR}/deps
		${CMAKE_CURRENT_SOURCE_DIR}/deps/chinoise/include
		${CMAKE_CURRENT_SOURCE_DIR}/deps/cub
		${CMAKE_CURRENT_SOURCE_DIR}/deps/epsilon/include
		${CMAKE_CURRENT_SOURCE_DIR}/deps/glad
		${CMAKE_CURRENT_SOURCE_DIR}/external/OpenMesh-8.0/$<CONFIG>/include
		${CMAKE_CURRENT_SOURCE_DIR}/external/oidn-0.9/$<CONFIG>/include
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
		${CMAKE_CURRENT_SOURCE_DIR}/deps
		${CMAKE_CURRENT_SOURCE_DIR}/deps/chinoise/include
		${CMAKE_CURRENT_SOURCE_DIR}/deps/epsilon/include
		${CMAKE_CURRENT_SOURCE_DIR}/deps/miniz
		${CMAKE_CURRENT_SOURCE_DIR}/external/rapidjson-1.1.0/$<CONFIG>/include
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

set(CXX_TARGETS core gliloader mffloader pfmloader stbiloader)

# Set CUDA flags
set_target_properties(core PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(core PROPERTIES CUDA_STANDARD 14)
set_target_properties(core PROPERTIES CUDA_STANDARD_REQUIRED ON)
target_compile_options(core PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
target_compile_options(core PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:${ARCH_FLAGS}>)

set_target_properties(${CXX_TARGETS} PROPERTIES
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
		set(MUFFLON_WARNING_FLAGS -Wall -Wextra -Wno-comment -Wno-unknown-pragmas -Wno-class-memaccess)
		if(MUFFLON_STRICT_COMPILATION)
			target_compile_options(${TAR} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${MUFFLON_WARNING_FLAGS} -Werror>)
		else(MUFFLON_STRICT_COMPILATION)
			target_compile_options(${TAR} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${MUFFLON_WARNING_FLAGS}>)
		endif(MUFFLON_STRICT_COMPILATION)
	endforeach(TAR)
	# On GCC, we need to link in a library to get filesystem support
	set(CXX_FS_LIBRARY "stdc++fs")
endif()
# Since GLI has some warnings themselves that we can't (or won't) fix, we just ignore them
if(NOT MSVC)
	target_compile_options(gliloader PRIVATE
		-Wno-ignored-qualifiers -Wno-unused-parameter -Wno-empty-body -Wno-unused-result -Wno-unused-variable -Wno-type-limits
	)
endif()

# Dependency library targets
add_library(openmesh_core SHARED IMPORTED)
add_library(openmesh_tools SHARED IMPORTED)
add_library(opendenoise SHARED IMPORTED)
set(MUFFLON_OPENMESH_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/external/OpenMesh-8.0")
set(MUFFLON_OPENDENOISE_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/external/oidn-0.9")
if(MSVC)
	set_target_properties(openmesh_core PROPERTIES IMPORTED_LOCATION_DEBUG "${MUFFLON_OPENMESH_DIRECTORY}/Debug/bin/OpenMeshCored.dll")
	set_target_properties(openmesh_core PROPERTIES IMPORTED_LOCATION_RELEASE "${MUFFLON_OPENMESH_DIRECTORY}/Release/bin/OpenMeshCore.dll")
	set_target_properties(openmesh_core PROPERTIES IMPORTED_IMPLIB_DEBUG "${MUFFLON_OPENMESH_DIRECTORY}/Debug/lib/OpenMeshCored.lib")
	set_target_properties(openmesh_core PROPERTIES IMPORTED_IMPLIB_RELEASE "${MUFFLON_OPENMESH_DIRECTORY}/Release/lib/OpenMeshCore.lib")
	set_target_properties(openmesh_tools PROPERTIES IMPORTED_LOCATION_DEBUG "${MUFFLON_OPENMESH_DIRECTORY}/Debug/bin/OpenMeshToolsd.dll")
	set_target_properties(openmesh_tools PROPERTIES IMPORTED_LOCATION_RELEASE "${MUFFLON_OPENMESH_DIRECTORY}/Release/bin/OpenMeshTools.dll")
	set_target_properties(openmesh_tools PROPERTIES IMPORTED_IMPLIB_DEBUG "${MUFFLON_OPENMESH_DIRECTORY}/Debug/lib/OpenMeshToolsd.lib")
	set_target_properties(openmesh_tools PROPERTIES IMPORTED_IMPLIB_RELEASE "${MUFFLON_OPENMESH_DIRECTORY}/Release/lib/OpenMeshTools.lib")
	set_target_properties(opendenoise PROPERTIES IMPORTED_IMPLIB_DEBUG "${MUFFLON_OPENDENOISE_DIRECTORY}/Debug/lib/OpenImageDenoise.lib")
	set_target_properties(opendenoise PROPERTIES IMPORTED_IMPLIB_DEBUG "${MUFFLON_OPENDENOISE_DIRECTORY}/Release/lib/OpenImageDenoise.lib")
	set_target_properties(opendenoise PROPERTIES IMPORTED_LOCATION_DEBUG "${MUFFLON_OPENDENOISE_DIRECTORY}/Debug/bin/OpenImageDenoise.dll")
	set_target_properties(opendenoise PROPERTIES IMPORTED_LOCATION_RELEASE "${MUFFLON_OPENDENOISE_DIRECTORY}/Release/bin/OpenImageDenoise.dll")
	install(FILES
			"${MUFFLON_OPENMESH_DIRECTORY}/Debug/bin/OpenMeshCored.dll"
			"${MUFFLON_OPENMESH_DIRECTORY}/Debug/bin/OpenMeshToolsd.dll"
			"${MUFFLON_OPENDENOISE_DIRECTORY}/Debug/bin/OpenImageDenoise.dll"
		DESTINATION "."
		CONFIGURATIONS Debug
	)
	install(FILES
			"${MUFFLON_OPENMESH_DIRECTORY}/Release/bin/OpenMeshCore.dll"
			"${MUFFLON_OPENMESH_DIRECTORY}/Release/bin/OpenMeshTools.dll"
			"${MUFFLON_OPENDENOISE_DIRECTORY}/Release/bin/OpenImageDenoise.dll"
		DESTINATION "."
		CONFIGURATIONS Release
	)
else(MSVC)
	set_target_properties(openmesh_core PROPERTIES IMPORTED_LOCATION_DEBUG "${MUFFLON_OPENMESH_DIRECTORY}/Debug/lib/libOpenMeshCored.so")
	set_target_properties(openmesh_core PROPERTIES IMPORTED_LOCATION_RELEASE "${MUFFLON_OPENMESH_DIRECTORY}/Release/lib/libOpenMeshCore.so")
	set_target_properties(openmesh_tools PROPERTIES IMPORTED_LOCATION_DEBUG "${MUFFLON_OPENMESH_DIRECTORY}/Debug/lib/libOpenMeshToolsd.so")
	set_target_properties(openmesh_tools PROPERTIES IMPORTED_LOCATION_RELEASE "${MUFFLON_OPENMESH_DIRECTORY}/Release/lib/libOpenMeshTools.so")
	set_target_properties(opendenoise PROPERTIES IMPORTED_LOCATION_DEBUG "${MUFFLON_OPENDENOISE_DIRECTORY}/Debug/lib/libOpenImageDenoise.so")
	set_target_properties(opendenoise PROPERTIES IMPORTED_LOCATION_RELEASE "${MUFFLON_OPENDENOISE_DIRECTORY}/Release/lib/libOpenImageDenoise.so")
	install(FILES
			"${MUFFLON_OPENMESH_DIRECTORY}/Debug/lib/libOpenMeshCored.so"
			"${MUFFLON_OPENMESH_DIRECTORY}/Debug/lib/libOpenMeshToolsd.so"
			"${MUFFLON_OPENDENOISE_DIRECTORY}/Debug/lib/libOpenImageDenoise.so"
		DESTINATION "."
		CONFIGURATIONS Debug
	)
	install(FILES
			"${MUFFLON_OPENMESH_DIRECTORY}/Release/lib/libOpenMeshCore.so"
			"${MUFFLON_OPENMESH_DIRECTORY}/Release/lib/libOpenMeshTools.so"
			"${MUFFLON_OPENDENOISE_DIRECTORY}/Release/lib/libOpenImageDenoise.so"
		DESTINATION "."
		CONFIGURATIONS Release
	)
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

	target_link_libraries(gui
		PUBLIC
			core
			mffloader
	)
endif(MUFFLON_BUILD_GUI)

# Inter-target dependencies
target_link_libraries(mffloader
	PUBLIC
		core
)
target_link_options(core PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:${ARCH_FLAGS}>)
target_link_libraries(core
	PUBLIC
		openmesh_core
		openmesh_tools
		opendenoise
)
if(OpenMP_CXX_FOUND)
	target_link_libraries(core
		PUBLIC
			OpenMP::OpenMP_CXX
	)
endif()

# Set output targets for plugins
set_target_properties(gliloader PROPERTIES
	ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/build/plugins"
	ARCHIVE_OUTPUT_DIRECTORY_DEBUG "${CMAKE_BINARY_DIR}/build/Debug/plugins"
	ARCHIVE_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}/build/Release/plugins"
	LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/build/plugins"
	LIBRARY_OUTPUT_DIRECTORY_DEBUG "${CMAKE_BINARY_DIR}/build/Debug/plugins"
	LIBRARY_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}/build/Release/plugins"
	RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/build/plugins"
	RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_BINARY_DIR}/build/Debug/plugins"
	RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}/build/Release/plugins"
)
set_target_properties(pfmloader PROPERTIES
	ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/build/plugins"
	ARCHIVE_OUTPUT_DIRECTORY_DEBUG "${CMAKE_BINARY_DIR}/build/Debug/plugins"
	ARCHIVE_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}/build/Release/plugins"
	LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/build/plugins"
	LIBRARY_OUTPUT_DIRECTORY_DEBUG "${CMAKE_BINARY_DIR}/build/Debug/plugins"
	LIBRARY_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}/build/Release/plugins"
	RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/build/plugins"
	RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_BINARY_DIR}/build/Debug/plugins"
	RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}/build/Release/plugins"
)
set_target_properties(stbiloader PROPERTIES
	ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/build/plugins"
	ARCHIVE_OUTPUT_DIRECTORY_DEBUG "${CMAKE_BINARY_DIR}/build/Debug/plugins"
	ARCHIVE_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}/build/Release/plugins"
	LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/build/plugins"
	LIBRARY_OUTPUT_DIRECTORY_DEBUG "${CMAKE_BINARY_DIR}/build/Debug/plugins"
	LIBRARY_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}/build/Release/plugins"
	RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/build/plugins"
	RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_BINARY_DIR}/build/Debug/plugins"
	RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}/build/Release/plugins"
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

# Install the dependencies (OpenMesh etc)
if(WIN32)
	add_custom_command(TARGET core PRE_BUILD
		COMMAND ${CMAKE_COMMAND} -E copy_if_different
			"${MUFFLON_OPENMESH_DIRECTORY}/$<CONFIG>/bin/OpenMeshCore$<$<CONFIG:Debug>:d>.dll"
			"${MUFFLON_OPENMESH_DIRECTORY}/$<CONFIG>/bin/OpenMeshTools$<$<CONFIG:Debug>:d>.dll"
			"${MUFFLON_OPENDENOISE_DIRECTORY}/$<CONFIG>/bin/OpenImageDenoise.dll"
			"${CMAKE_BINARY_DIR}/build/$<CONFIG>"
	)
else()
	add_custom_command(TARGET core PRE_BUILD
		COMMAND ${CMAKE_COMMAND} -E copy_if_different
			"${MUFFLON_OPENMESH_DIRECTORY}/$<CONFIG>/lib/libOpenMeshCore$<$<CONFIG:Debug>:d>.so"
			"${MUFFLON_OPENMESH_DIRECTORY}/$<CONFIG>/lib/libOpenMeshTools$<$<CONFIG:Debug>:d>.so"
			"${MUFFLON_OPENDENOISE_DIRECTORY}/$<CONFIG>/lib/libOpenImageDenoise.so"
			"${CMAKE_BINARY_DIR}/build/$<CONFIG>"
	)
endif()

# TODO: specify install targets
install(TARGETS core mffloader
	LIBRARY DESTINATION "."
)
install(TARGETS stbiloader pfmloader gliloader
	LIBRARY DESTINATION "plugins"
)
install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/src/pythonbinding/bindings.py"
	DESTINATION ".")