#include "lod.hpp"
#include "object.hpp"
#include "scenario.hpp"
#include "materials/material.hpp"
#include "profiler/cpu_profiler.hpp"
#include "core/scene/tessellation/tessellater.hpp"

namespace mufflon::scene {

namespace {

template < class I1, class I2 >
constexpr bool sorted_share_elements(const I1 beginA, const I1 endA, const I2 beginB, const I2 endB) noexcept {
	auto currA = beginA;
	auto currB = beginB;
	while(currA != endA && currB != endB) {
		if(*currA == *currB)
			return true;
		if(*currA < *currB)
			++currA;
		else
			++currB;
	}
	return false;
}

} // namespace

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

void Lod::update_material_indices() noexcept {
	auto& polys = m_geometry.template get<geometry::Polygons>();
	auto& spheres = m_geometry.template get<geometry::Spheres>();

	m_uniqueMatIndices.clear();
	m_uniqueMatIndices.reserve(polys.get_face_count() + spheres.get_sphere_count());
	const auto* polyMatIndices = polys.template acquire_const<Device::CPU, MaterialIndex>(polys.get_material_indices_hdl());
	const auto* sphereMatIndices = spheres.template acquire_const<Device::CPU, MaterialIndex>(spheres.get_material_indices_hdl());
	for(std::size_t i = 0u; i < polys.get_face_count(); ++i)
		m_uniqueMatIndices.push_back(polyMatIndices[i]);
	for(std::size_t i = 0u; i < spheres.get_sphere_count(); ++i)
		m_uniqueMatIndices.push_back(sphereMatIndices[i]);
	std::sort(m_uniqueMatIndices.begin(), m_uniqueMatIndices.end());
	const auto end = std::unique(m_uniqueMatIndices.begin(), m_uniqueMatIndices.end());
	m_uniqueMatIndices.resize(end - m_uniqueMatIndices.begin());
}

bool Lod::is_emissive(const std::vector<MaterialIndex>& emissiveMatIndices) const noexcept {
	mAssert(!m_uniqueMatIndices.empty());
	return sorted_share_elements(emissiveMatIndices.cbegin(), emissiveMatIndices.cend(),
								 m_uniqueMatIndices.cbegin(), m_uniqueMatIndices.cend());
}
// Is there any displaced polygon in this object
bool Lod::is_displaced(const std::vector<MaterialIndex>& displacedMatIndices) const noexcept {
	mAssert(!m_uniqueMatIndices.empty());
	return sorted_share_elements(displacedMatIndices.cbegin(), displacedMatIndices.cend(),
								 m_uniqueMatIndices.cbegin(), m_uniqueMatIndices.cend());
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

void Lod::apply_animation(u32 frame, const Bone* bones) {
	if(!has_bone_animation()) return;
	if(m_appliedFrame == frame) return;
	if(m_appliedFrame != ~0u)
		logWarning("[Lod::apply_animation] There is a different animation frame applied. The new animation will be made on top of that.");
	bool hasChanged = false;
	m_geometry.for_each([&hasChanged, frame, bones](auto& elem) {
		hasChanged |= elem.apply_animation(frame, bones);
	});
	if(hasChanged)
		m_accelStruct.mark_invalid();
	m_appliedFrame = frame;
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
