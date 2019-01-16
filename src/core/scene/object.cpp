#include "object.hpp"
#include "profiler/cpu_profiler.hpp"
#include "core/scene/scenario.hpp"
#include "core/scene/materials/material.hpp"


namespace mufflon::scene {

Object::Object() {
}

Object::Object(Object&& obj) :
	m_name(obj.m_name),
	m_geometryData(std::move(obj.m_geometryData)),
	m_accelStruct(),
	m_animationFrame(obj.m_animationFrame),
	m_lodLevel(obj.m_lodLevel),
	m_flags(obj.m_flags)
{

}

Object::~Object() {

}

bool Object::is_emissive(const Scenario& scenario) const noexcept {
	for(MaterialIndex m : m_geometryData.get<geometry::Polygons>().get_unique_materials())
		if(scenario.get_assigned_material(m)->get_properties().is_emissive()) return true;
	for(MaterialIndex m : m_geometryData.get<geometry::Spheres>().get_unique_materials())
		if(scenario.get_assigned_material(m)->get_properties().is_emissive()) return true;
	return false;
}

void Object::clear_accel_structure() {
	/*m_accelStruct[get_device_index<Device::CPU>()].type = accel_struct::AccelType::NONE;
	m_accelStruct[get_device_index<Device::CUDA>()].type = accel_struct::AccelType::NONE;*/
	// TODO memory
}

template < Device dev >
ObjectDescriptor<dev> Object::get_descriptor() {
	ObjectDescriptor<dev> desc{
		m_geometryData.get<geometry::Polygons>().get_descriptor<dev>(),
		m_geometryData.get<geometry::Spheres>().get_descriptor<dev>(),
		0, // Below (very lengthy otherwise)
		AccelDescriptor{}
	};
	desc.numPrimitives = desc.polygon.numTriangles + desc.polygon.numQuads + desc.spheres.numSpheres;
	// (Re)build acceleration structure if necessary
	if(m_accelStruct.needs_rebuild<dev>()) {
		auto timer = Profiler::instance().start<CpuProfileState>("[Object::get_descriptor] build object BVH.");
		m_accelStruct.build(desc, get_bounding_box());
	}
	desc.accelStruct = m_accelStruct.acquire_const<dev>();
	return desc;
}

template < Device dev >
void Object::update_attribute_descriptor(ObjectDescriptor<dev>& descriptor,
										 const std::vector<const char*>& vertexAttribs,
										 const std::vector<const char*>& faceAttribs,
										 const std::vector<const char*>& sphereAttribs) {
	m_geometryData.get<geometry::Polygons>().update_attribute_descriptor<dev>(descriptor.polygon, vertexAttribs, faceAttribs);
	m_geometryData.get<geometry::Spheres>().update_attribute_descriptor<dev>(descriptor.spheres, sphereAttribs);
}

template ObjectDescriptor<Device::CPU> Object::get_descriptor<Device::CPU>();
template ObjectDescriptor<Device::CUDA> Object::get_descriptor<Device::CUDA>();
template void Object::update_attribute_descriptor<Device::CPU>(ObjectDescriptor<Device::CPU>&,
												  const std::vector<const char*>&,
												  const std::vector<const char*>&,
												  const std::vector<const char*>&);
template void Object::update_attribute_descriptor<Device::CUDA>(ObjectDescriptor<Device::CUDA>&,
												  const std::vector<const char*>&,
												  const std::vector<const char*>&,
												  const std::vector<const char*>&);
/*template ObjectDescriptor<Device::OPENGL> Object::get_descriptor<Device::OPENGL>(const std::vector<const char*>&,
																				 const std::vector<const char*>&,
																				 const std::vector<const char*>&);*/

} // namespace mufflon::scene
