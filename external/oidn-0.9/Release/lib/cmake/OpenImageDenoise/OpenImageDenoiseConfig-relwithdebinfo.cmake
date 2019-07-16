#----------------------------------------------------------------
# Generated CMake target import file for configuration "RelWithDebInfo".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "OpenImageDenoise" for configuration "RelWithDebInfo"
set_property(TARGET OpenImageDenoise APPEND PROPERTY IMPORTED_CONFIGURATIONS RELWITHDEBINFO)
set_target_properties(OpenImageDenoise PROPERTIES
  IMPORTED_IMPLIB_RELWITHDEBINFO "${_IMPORT_PREFIX}/lib/OpenImageDenoise.lib"
  IMPORTED_LOCATION_RELWITHDEBINFO "${_IMPORT_PREFIX}/bin/OpenImageDenoise.dll"
  )

list(APPEND _IMPORT_CHECK_TARGETS OpenImageDenoise )
list(APPEND _IMPORT_CHECK_FILES_FOR_OpenImageDenoise "${_IMPORT_PREFIX}/lib/OpenImageDenoise.lib" "${_IMPORT_PREFIX}/bin/OpenImageDenoise.dll" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
