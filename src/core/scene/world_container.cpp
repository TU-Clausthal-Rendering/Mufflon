#include "world_container.hpp"

namespace mufflon::scene {

material::MaterialHandle WorldContainer::add_material(std::unique_ptr<material::IMaterial> material) {
	m_materials.push_back(move(material));
	return m_materials.back().get();
}

} // namespace mufflon::scene