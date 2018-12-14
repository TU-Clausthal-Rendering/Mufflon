#pragma once

//#include "core/scene/instance.hpp"
//#include "core/scene/handles.hpp"
//#include "core/memory/residency.hpp"
//#include <OpenMesh/Core/Mesh/PolyConnectivity.hh>
#include <cstdlib>

// Forward declarations
namespace ei {
struct Ray;
} // namespace ei

namespace mufflon { namespace scene {

namespace accel_struct {

// All supported types of acceleration structures
enum class AccelType {
	NONE,
	LBVH
};

constexpr std::size_t MAX_ACCEL_STRUCT_PARAMETER_SIZE = 24;

}}} // namespace mufflon::scene::accel_struct