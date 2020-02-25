#include "hashmap.hpp"
#include "core/concepts.hpp"

namespace mufflon {

template struct DeviceManagerConcept<HashMapManager<int, int>>;

} // namespace mufflon