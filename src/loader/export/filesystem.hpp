#pragma once

// Make use of filepaths
#if !defined(__cpp_lib_filesystem)
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;
#else // !defined(__cpp_lib_filesystem)
#include <filesystem>
namespace fs = std::filesystem;
#endif // !defined(__cpp_lib_filesystem)