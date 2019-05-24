#include "lod.hpp"
#include "object.hpp"
#include "scenario.hpp"
#include "materials/material.hpp"
#include "profiler/cpu_profiler.hpp"
#include "core/scene/tessellation/tessellater.hpp"

namespace mufflon::scene {

bool Lod::is_emissive(const Scenario& scenario) const noexcept {
	for(MaterialIndex m : m_geometry.template get<geometry::Polygons>().get_unique_materials())
		if(scenario.get_assigned_material(m)->get_properties().is_emissive()) return true;
	for(MaterialIndex m : m_geometry.template get<geometry::Spheres>().get_unique_materials())
		if(scenario.get_assigned_material(m)->get_properties().is_emissive()) return true;
	return false;
}

void Lod::clear_accel_structure() {
	/*m_accelStruct[get_device_index<Device::CPU>()].type = accel_struct::AccelType::NONE;
	m_accelStruct[get_device_index<Device::CUDA>()].type = accel_struct::AccelType::NONE;*/
	// TODO memory
}

template < Device dev >
LodDescriptor<dev> Lod::get_descriptor() {
	// TODO: LOD that shit
	LodDescriptor<dev> desc{
		m_geometry.get<geometry::Polygons>().get_descriptor<dev>(),
		m_geometry.get<geometry::Spheres>().get_descriptor<dev>(),
		0, // Below (very lengthy otherwise)
		AccelDescriptor{}
	};
	desc.numPrimitives = desc.polygon.numTriangles + desc.polygon.numQuads + desc.spheres.numSpheres;
	// (Re)build acceleration structure if necessary
	if(m_accelStruct.needs_rebuild()) {
		logInfo("[Lod::get_descriptor] Building accelleration structure for object '", m_parent->get_name(),
				"' with ", desc.numPrimitives, " primitives (", desc.polygon.numTriangles, "T / ", desc.polygon.numQuads, "Q / ", desc.spheres.numSpheres, "S).");
		auto timer = Profiler::instance().start<CpuProfileState>("[Lod::get_descriptor] build object BVH.");
		m_accelStruct.build(desc, get_bounding_box());
	}
	desc.accelStruct = m_accelStruct.acquire_const<dev>();
	return desc;
}

void Lod::displace(tessellation::TessLevelOracle& tessellater, const Scenario& scenario) {
	m_geometry.for_each([&tessellater, &scenario](auto& elem) {
		elem.displace(tessellater, scenario);
	});
}

template < Device dev >
void Lod::update_attribute_descriptor(LodDescriptor<dev>& descriptor,
										 const std::vector<const char*>& vertexAttribs,
										 const std::vector<const char*>& faceAttribs,
										 const std::vector<const char*>& sphereAttribs) {
	m_geometry.get<geometry::Polygons>().update_attribute_descriptor<dev>(descriptor.polygon, vertexAttribs, faceAttribs);
	m_geometry.get<geometry::Spheres>().update_attribute_descriptor<dev>(descriptor.spheres, sphereAttribs);
}

template LodDescriptor<Device::CPU> Lod::get_descriptor<Device::CPU>();
template LodDescriptor<Device::CUDA> Lod::get_descriptor<Device::CUDA>();
template LodDescriptor<Device::OPENGL> Lod::get_descriptor<Device::OPENGL>();
template void Lod::update_attribute_descriptor<Device::CPU>(LodDescriptor<Device::CPU>&,
															   const std::vector<const char*>&,
															   const std::vector<const char*>&,
															   const std::vector<const char*>&);
template void Lod::update_attribute_descriptor<Device::CUDA>(LodDescriptor<Device::CUDA>&,
																const std::vector<const char*>&,
																const std::vector<const char*>&,
																const std::vector<const char*>&);
template void Lod::update_attribute_descriptor<Device::OPENGL>(LodDescriptor<Device::OPENGL>&,
																const std::vector<const char*>&,
																const std::vector<const char*>&,
																const std::vector<const char*>&);
} // namespace mufflon::scene