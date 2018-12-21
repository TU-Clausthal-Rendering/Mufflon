#include "object.hpp"
#include "profiler/cpu_profiler.hpp"

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

void Object::clear_accel_structure() {
	/*m_accelStruct[get_device_index<Device::CPU>()].type = accel_struct::AccelType::NONE;
	m_accelStruct[get_device_index<Device::CUDA>()].type = accel_struct::AccelType::NONE;*/
	// TODO memory
}

template < Device dev >
ObjectDescriptor<dev> Object::get_descriptor(const std::vector<const char*>& vertexAttribs,
											 const std::vector<const char*>& faceAttribs,
											 const std::vector<const char*>& sphereAttribs) {
	ObjectDescriptor<dev> desc{
		m_geometryData.get<geometry::Polygons>().get_descriptor<dev>(vertexAttribs, faceAttribs),
		m_geometryData.get<geometry::Spheres>().get_descriptor<dev>(sphereAttribs),
		0, // Below (very lengthy otherwise)
		AccelDescriptor{}
	};
	desc.numPrimitives = desc.polygon.numTriangles + desc.polygon.numQuads + desc.spheres.numSpheres;
	// (Re)build acceleration structure if necessary
	if (m_accelStruct.needs_rebuild<dev>()) {
		auto scope = Profiler::instance().start<CpuProfileState>("build_obj_bvh");
		m_accelStruct.build(desc, get_bounding_box());
	}
	desc.accelStruct = m_accelStruct.acquire_const<dev>();

	return desc;
}

template ObjectDescriptor<Device::CPU> Object::get_descriptor<Device::CPU>(const std::vector<const char*>&,
																		   const std::vector<const char*>&,
																		   const std::vector<const char*>&);
template ObjectDescriptor<Device::CUDA> Object::get_descriptor<Device::CUDA>(const std::vector<const char*>&,
																			 const std::vector<const char*>&,
																			 const std::vector<const char*>&);
/*template ObjectDescriptor<Device::OPENGL> Object::get_descriptor<Device::OPENGL>(const std::vector<const char*>&,
																				 const std::vector<const char*>&,
																				 const std::vector<const char*>&);*/

} // namespace mufflon::scene
