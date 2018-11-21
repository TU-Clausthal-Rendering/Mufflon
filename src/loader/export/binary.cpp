#include "binary.hpp"
#include "util/int_types.hpp"
#include <fstream>
#include <string>

namespace loader::binary {

namespace {

template < class T >
T read(std::ifstream& stream) {
	T val;
	stream >> val;
	return val;
}

} // namespace

using namespace mufflon;

inline constexpr u32 MATERIALS_HEADER_MAGIC = ('M' << 24u) | ('a' << 16u) | ('t' << 8u) | 's';

void parse_file(fs::path filePath) {
	if(!fs::exists(filePath))
		throw std::runtime_error("Binary file '" + filePath.string() + "' does not exist");

	// Open the binary file and enable exception management
	std::ifstream fileStream(filePath, std::ios::in | std::ios::binary);
	if(fileStream.bad())
		throw std::runtime_error("Failed to open binary file '" + filePath.string() + '\'');
	fileStream.exceptions(std::ifstream::failbit);

	// Read the materials header
	if(read<u32>(fileStream) != MATERIALS_HEADER_MAGIC)
		throw std::runtime_error("Invalid material header magic constant");
}

} // namespace loader::binary