#include "object.hpp"
#include "core/scene/accel_structs/accel_struct.hpp"

namespace mufflon::scene {

Object::Object() {
	// Invalidate bounding box
	m_boundingBox.min = {
		std::numeric_limits<float>::max(),
		std::numeric_limits<float>::max(),
		std::numeric_limits<float>::max()
	};
	m_boundingBox.max = {
		std::numeric_limits<float>::min(),
		std::numeric_limits<float>::min(),
		std::numeric_limits<float>::min()
	};
}

Object::~Object() = default;

bool Object::is_accel_dirty(Device res) const noexcept {
	return m_accelDirty || m_accelStruct->is_dirty(res);
}

void Object::clear_accel_structure() {
	// Mark as dirty only if we change something
	m_accelDirty |= m_accelStruct != nullptr;
	m_accelStruct.reset();
}

void Object::build_accel_structure() {
	// We no longer need this indication - the structure itself will tell us
	// if and where we are dirty
	m_accelDirty = false;
	m_accelStruct->build(m_boundingBox,
						 m_geometryData.get<geometry::Polygons>().faces(),
						 m_geometryData.get<geometry::Spheres>().get_spheres(),
						 m_geometryData.get<geometry::Polygons>().get_triangle_count(),
						 m_geometryData.get<geometry::Polygons>().get_quad_count());
}

} // namespace mufflon::scene
