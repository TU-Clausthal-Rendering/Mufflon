#pragma once
#include <string>

#ifdef DLLFUNC
#error DLLFUNC redefinition
#endif
#define DLLFUNC(return_type) extern "C" __declspec(dllexport) return_type __cdecl

/// \brief initializes the opengl functions (glad)
/// \param logCallback callback function for the gui (the text will be displayed on the console)
/// \return true on success
DLLFUNC(bool) initialize(void(*logCallback)(const char*, int));

/// \brief performs one iteration of the active renderer
DLLFUNC(bool) iterate();

/// \brief indicates that the size of the client area has changed. 
///        Will be called after intitialize as well
/// \param width width of the viewport area
/// \param height height of the viewport area
/// \return true on success
DLLFUNC(bool) resize(int width, int height, int offsetX, int offsetY);

DLLFUNC(void) execute_command(const char* command);

/// \brief retrieves last error from set_error()
/// \param length of the error
/// \return last error
DLLFUNC(const char*) get_error(int& length);

/// \brief sets the get_error() return value
/// \param error
void set_error(std::string error);

#undef DLLFUNC