#include "lod.hpp"
#include "object.hpp"
#include "scenario.hpp"
#include "materials/material.hpp"
#include "profiler/cpu_profiler.hpp"
#include "core/scene/tessellation/tessellater.hpp"

namespace mufflon::scene {

void Lod::clear_accel_structure() {
	m_accelStruct.mark_invalid();
}

template < Device dev >
LodDescriptor<dev> Lod::get_descriptor(const bool allowSerialBvhBuild) {
	// TODO: LOD that shit
	LodDescriptor<dev> desc{
		m_geometry.get<geometry::Polygons>().get_descriptor<dev>(),
		m_geometry.get<geometry::Spheres>().get_descriptor<dev>(),
		0, // Below (very lengthy otherwise)
		AccelDescriptor{}
	};
	desc.numPrimitives = desc.polygon.numTriangles + desc.polygon.numQuads + desc.spheres.numSpheres;
	// If we're allowed to have a serial BVH build, we make it dependent on the number of primitives
	const bool parallelBuild = allowSerialBvhBuild ? (desc.numPrimitives >= 1000) : true;

	// (Re)build acceleration structure if necessary
	if(m_accelStruct.needs_rebuild()) {
		logPedantic("[Lod::get_descriptor] Building accelleration structure for object '",
					m_parent->get_name(), "' with ", desc.numPrimitives, " primitives (",
					desc.polygon.numTriangles, "T / ", desc.polygon.numQuads, "Q / ",
					desc.spheres.numSpheres, "S).");
		auto timer = Profiler::core().start<CpuProfileState>("[Lod::get_descriptor] build object BVH.");
		m_accelStruct.build(desc, get_bounding_box(), parallelBuild);
	}
	desc.accelStruct = m_accelStruct.acquire_const<dev>();
	return desc;
}
void Lod::update_flags(const Scenario& scenario, std::unordered_set<MaterialIndex>& uniqueMatCache) {
	auto& polys = this->template get_geometry<geometry::Polygons>();
	auto& spheres = this->template get_geometry<geometry::Spheres>();

	// Collect the unique material indices of the LoD
	const auto* polyMatIndices = polys.template acquire_const<Device::CPU, MaterialIndex>(polys.get_material_indices_hdl());
	const auto* sphereMatIndices = spheres.template acquire_const<Device::CPU, MaterialIndex>(spheres.get_material_indices_hdl());
	uniqueMatCache.clear();
	for(std::size_t i = 0u; i < polys.get_face_count(); ++i)
		uniqueMatCache.insert(polyMatIndices[i]);
	for(std::size_t i = 0u; i < spheres.get_sphere_count(); ++i)
		uniqueMatCache.insert(sphereMatIndices[i]);

	// Now check for each scenario if it is emissive and/or displaced
	// and flag the LoD accordingly
	for(const auto mat : uniqueMatCache) {
		const auto* material = scenario.get_assigned_material(mat);
		if(material->get_properties().is_emissive())
			m_flags |= (1llu << static_cast<u64>(2u * scenario.get_index()));
		if(material->get_displacement_map() != nullptr)
			m_flags |= (1llu << static_cast<u64>(2u * scenario.get_index() + 1u));
	}
}

void Lod::displace(tessellation::TessLevelOracle& tessellater, const Scenario& scenario) {
	m_geometry.for_each([&tessellater, &scenario](auto& elem) {
		elem.displace(tessellater, scenario);
	});
	m_accelStruct.mark_invalid();
}

void Lod::tessellate(tessellation::TessLevelOracle& oracle, const Scenario* scenario,
					 const bool usePhong) {
	m_geometry.for_each([&oracle, scenario, usePhong](auto& elem) {
		elem.tessellate(oracle, scenario, usePhong);
	});
	m_accelStruct.mark_invalid();
}

template < Device dev >
void Lod::update_attribute_descriptor(LodDescriptor<dev>& descriptor,
										 const std::vector<AttributeIdentifier>& vertexAttribs,
										 const std::vector<AttributeIdentifier>& faceAttribs,
										 const std::vector<AttributeIdentifier>& sphereAttribs) {
	m_geometry.get<geometry::Polygons>().update_attribute_descriptor<dev>(descriptor.polygon, vertexAttribs, faceAttribs);
	m_geometry.get<geometry::Spheres>().update_attribute_descriptor<dev>(descriptor.spheres, sphereAttribs);
}

template LodDescriptor<Device::CPU> Lod::get_descriptor<Device::CPU>(const bool allowSerialBvhBuild);
template LodDescriptor<Device::CUDA> Lod::get_descriptor<Device::CUDA>(const bool allowSerialBvhBuild);
template LodDescriptor<Device::OPENGL> Lod::get_descriptor<Device::OPENGL>(const bool allowSerialBvhBuild);
template void Lod::update_attribute_descriptor<Device::CPU>(LodDescriptor<Device::CPU>&,
															   const std::vector<AttributeIdentifier>&,
															   const std::vector<AttributeIdentifier>&,
															   const std::vector<AttributeIdentifier>&);
template void Lod::update_attribute_descriptor<Device::CUDA>(LodDescriptor<Device::CUDA>&,
																const std::vector<AttributeIdentifier>&,
																const std::vector<AttributeIdentifier>&,
																const std::vector<AttributeIdentifier>&);
template void Lod::update_attribute_descriptor<Device::OPENGL>(LodDescriptor<Device::OPENGL>&,
																const std::vector<AttributeIdentifier>&,
																const std::vector<AttributeIdentifier>&,
																const std::vector<AttributeIdentifier>&);
} // namespace mufflon::scene
