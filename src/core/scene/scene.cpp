#include "scene.hpp"
#include "descriptors.hpp"
#include "scenario.hpp"
#include "core/cameras/camera.hpp"
#include "core/cameras/pinhole.hpp"
#include "core/cameras/focus.hpp"
#include "core/scene/lod.hpp"
#include "core/scene/object.hpp"
#include "core/scene/materials/medium.hpp"
#include "core/scene/materials/material.hpp"
#include "core/scene/materials/point_medium.hpp"
#include "core/scene/geometry/polygon.hpp"
#include "core/scene/geometry/sphere.hpp"
#include "core/scene/lights/background.hpp"
#include "profiler/cpu_profiler.hpp"

namespace mufflon { namespace scene {

bool Scene::is_sane() const noexcept {
	if(m_scenario.get_camera() == nullptr) {
		logWarning("[Scene::is_sane] No camera given.");
		return false;
	}
	if(!m_lightTree.get_envLight()) {
		// No envLight: we need some kind of light
		if(m_lightTree.get_light_count() == 0u) {
			logWarning("[Scene::is_sane] No light sources given.");
			return false;
		}
	}
	return true;
}

void Scene::add_instance(InstanceHandle hdl) {
	auto iter = m_objects.find(&hdl->get_object());
	if(iter == m_objects.end())
		m_objects.emplace(&hdl->get_object(), std::vector<InstanceHandle>{hdl}).first;
	else
		iter->second.push_back(hdl);
	// Check if we already have the object somewhere
	m_boundingBox = ei::Box{ m_boundingBox, hdl->get_bounding_box(m_scenario.get_effective_lod(hdl)) };
	clear_accel_structure();
}

void Scene::load_media(const std::vector<materials::Medium>& media) {
	m_media.resize(sizeof(materials::Medium) * media.size());
	materials::Medium* dst = as<materials::Medium>(m_media.acquire<Device::CPU>());
	memcpy(dst, media.data(), m_media.size());
	m_media.mark_changed(Device::CPU);
}

template< Device dev >
void Scene::load_materials() {
	// 1. Pass get the sizes for the index -> material offset table
	// TODO: if multiple slots bind the same material it would be possible to copy
	// the offset and to upload less materials in total.
	std::vector<int> offsets;
	// Store in one block -> table size is offset of first material and align the offset to the required alignment of material descriptors 
	std::size_t offset = round_to_align<alignof(materials::MaterialDescriptorBase)>(sizeof(int) * m_scenario.get_num_material_slots());
	for(MaterialIndex i = 0; i < m_scenario.get_num_material_slots(); ++i) {
		mAssert(offset <= std::numeric_limits<i32>::max());
		offsets.push_back(i32(offset));
		//offset += materials::get_handle_pack_size();
		offset += m_scenario.get_assigned_material(i)->get_descriptor_size(dev);
	}
	// Allocate the memory
	m_materials.resize(offset);
	auto mem = m_materials.acquire<dev>();
	copy(mem, as<char>(offsets.data()), 0, sizeof(int) * m_scenario.get_num_material_slots());
	// 2. Pass get all the material descriptors
	char buffer[materials::MAX_MATERIAL_DESCRIPTOR_SIZE()];
	for(MaterialIndex i = 0; i < m_scenario.get_num_material_slots(); ++i) {
		ConstMaterialHandle mat = m_scenario.get_assigned_material(i);
		mAssert(mat->get_descriptor_size(dev) <= materials::MAX_MATERIAL_DESCRIPTOR_SIZE());
		std::size_t size = mat->get_descriptor(dev, buffer) - buffer;
		copy(mem, buffer, offsets[i], size);
	}
	m_materials.mark_synced(dev); // Avoid overwrites with data from different devices.
}

template < Device dev >
const SceneDescriptor<dev>& Scene::get_descriptor(const std::vector<const char*>& vertexAttribs,
												  const std::vector<const char*>& faceAttribs,
												  const std::vector<const char*>& sphereAttribs,
												  const ei::IVec2& resolution) {
	synchronize<dev>();
	SceneDescriptor<dev>& sceneDescriptor = m_descStore.template get<SceneDescriptor<dev>>();

	// Check if we need to update attributes
	auto& lastVertexAttribs = m_lastAttributeNames.template get<AttributeNames<dev>>().lastVertexAttribs;
	auto& lastFaceAttribs = m_lastAttributeNames.template get<AttributeNames<dev>>().lastFaceAttribs;
	auto& lastSphereAttribs = m_lastAttributeNames.template get<AttributeNames<dev>>().lastSphereAttribs;
	bool sameAttribs = lastVertexAttribs.size() == vertexAttribs.size()
		&& lastFaceAttribs.size() == faceAttribs.size()
		&& lastSphereAttribs.size() == sphereAttribs.size();
	if(sameAttribs)
		for(auto name : vertexAttribs) {
			if(std::find_if(lastVertexAttribs.cbegin(), lastVertexAttribs.cend(), [name](const char* n) { return std::strcmp(name, n) != 0; }) != lastVertexAttribs.cend()) {
				sameAttribs = false;
				lastVertexAttribs = vertexAttribs;
				break;
			}
		}
	if(sameAttribs)
		for(auto name : faceAttribs) {
			if(std::find_if(lastFaceAttribs.cbegin(), lastFaceAttribs.cend(), [name](const char* n) { return std::strcmp(name, n) != 0; }) != lastFaceAttribs.cend()) {
				sameAttribs = false;
				lastFaceAttribs = faceAttribs;
				break;
			}
		}
	if(sameAttribs)
		for(auto name : sphereAttribs) {
			if(std::find_if(lastSphereAttribs.cbegin(), lastSphereAttribs.cend(), [name](const char* n) { return std::strcmp(name, n) != 0; }) != lastSphereAttribs.cend()) {
				sameAttribs = false;
				lastSphereAttribs = sphereAttribs;
				break;
			}
		}


	// Check if we need to update the object descriptors
	// TODO: this currently assumes that we do not add or alter geometry, which is clearly wrong
	// TODO: also needs to check for changed LoDs
	const bool geometryChanged = m_accelStruct.needs_rebuild();
	if(geometryChanged || !sceneDescriptor.lodIndices) {
		// Invalidate other descriptors
		if(geometryChanged)
			m_descStore.for_each([](auto& elem) { elem.lodIndices = {}; });

		std::vector<ei::Mat3x4> instanceTransformations;
		std::vector<ei::Mat3x4> invInstanceTransformations;
		std::vector<u32> lodIndices;
		std::vector<LodDescriptor<dev>> lodDescs;
		std::vector<ei::Box> lodAabbs;

		m_boundingBox.max = ei::Vec3{ -std::numeric_limits<float>::max() };
		m_boundingBox.min = ei::Vec3{ std::numeric_limits<float>::max() };

		// Create the object and instance descriptors
		std::size_t instanceCount = 0u;
		u32 lodCount = 0u;
		for(auto& obj : m_objects) {
			mAssert(obj.first != nullptr);
			mAssert(obj.second.size() != 0u);
			// First determine which LoDs are actually needed
			std::vector<u8> lods(obj.first->get_lod_slot_count(), 0u);

			u32 prevLevel = std::numeric_limits<u32>::max();
			for(InstanceHandle inst : obj.second) {
				mAssert(inst != nullptr);
				const u32 instanceLod = m_scenario.get_effective_lod(inst);
				if(!lods[instanceLod]) {
					mAssert(instanceLod < obj.first->get_lod_slot_count());
					if(prevLevel != std::numeric_limits<u32>::max())
						lodDescs.back().next = instanceLod;
					Lod& lod = obj.first->get_lod(instanceLod);
					lodDescs.push_back(lod.template get_descriptor<dev>());
					lodDescs.back().previous = prevLevel;
					lodAabbs.push_back(lod.get_bounding_box());
					if(!sameAttribs)
						lod.update_attribute_descriptor(lodDescs.back(), vertexAttribs, faceAttribs, sphereAttribs);
					lods[instanceLod] = true;
				}
				instanceTransformations.push_back(inst->get_transformation_matrix());
				invInstanceTransformations.push_back(inst->get_inverse_transformation_matrix());
				lodIndices.push_back(lodCount);
				// Also expand scene bounding box
				m_boundingBox = ei::Box(m_boundingBox, inst->get_bounding_box(instanceLod));
			}
			lodDescs.back().previous = prevLevel;

			instanceCount += obj.second.size();
			++lodCount;
		}

		// Allocate the device memory and copy over the descriptors
		auto& lodDevDesc = m_lodDevDesc.template get<unique_device_ptr<dev, LodDescriptor<dev>[]>>();
		lodDevDesc = make_udevptr_array<dev, LodDescriptor<dev>>(lodDescs.size());
		copy(lodDevDesc.get(), lodDescs.data(), 0, lodDescs.size() * sizeof(LodDescriptor<dev>));

		auto& instTransformsDesc = m_instTransformsDesc.template get<unique_device_ptr<dev, ei::Mat3x4[]>>();
		instTransformsDesc = make_udevptr_array<dev, ei::Mat3x4>(instanceTransformations.size());
		copy(instTransformsDesc.get(), instanceTransformations.data(), 0, sizeof(ei::Mat3x4) * instanceTransformations.size());

		auto& invInstTransformsDesc = m_invInstTransformsDesc.template get<unique_device_ptr<dev, ei::Mat3x4[]>>();
		invInstTransformsDesc = make_udevptr_array<dev, ei::Mat3x4>(invInstanceTransformations.size());
		copy(invInstTransformsDesc.get(), invInstanceTransformations.data(), 0, sizeof(ei::Mat3x4) * invInstanceTransformations.size());

		auto& instLodIndicesDesc = m_instLodIndicesDesc.template get<unique_device_ptr<dev, u32[]>>();
		instLodIndicesDesc = make_udevptr_array<dev, u32>(lodIndices.size());
		copy(instLodIndicesDesc.get(), lodIndices.data(), 0, sizeof(u32) * lodIndices.size());

		auto& lodAabbsDesc = m_lodAabbsDesc.template get<unique_device_ptr<dev, ei::Box[]>>();
		lodAabbsDesc = make_udevptr_array<dev, ei::Box>(lodAabbs.size());
		copy(lodAabbsDesc.get(), lodAabbs.data(), 0, sizeof(ei::Box) * lodAabbs.size());

		sceneDescriptor.numLods = static_cast<u32>(lodDescs.size());
		sceneDescriptor.numInstances = static_cast<i32>(instanceCount);
		sceneDescriptor.diagSize = len(m_boundingBox.max - m_boundingBox.min);
		sceneDescriptor.aabb = m_boundingBox;
		sceneDescriptor.lods = lodDevDesc.get();
		sceneDescriptor.aabbs = lodAabbsDesc.get();
		sceneDescriptor.instanceToWorld = instTransformsDesc.get();
		sceneDescriptor.worldToInstance = invInstTransformsDesc.get();
		sceneDescriptor.lodIndices = instLodIndicesDesc.get();
	} else if(!sameAttribs) {
		// Only update the descriptors and reupload them
		std::vector<LodDescriptor<dev>> lodDescs;
		for(auto& obj : m_objects) {
			mAssert(obj.first != nullptr);
			mAssert(obj.second.size() != 0u);

			// First determine which LoDs are actually used
			std::vector<u8> lods(obj.first->get_lod_slot_count(), 0u);

			u32 prevLevel = std::numeric_limits<u32>::max();
			for(InstanceHandle inst : obj.second) {
				const u32 instanceLod = m_scenario.get_effective_lod(inst);
				mAssert(inst != nullptr);
				if(!lods[instanceLod]) {
					mAssert(instanceLod < obj.first->get_lod_slot_count());
					if(prevLevel != std::numeric_limits<u32>::max())
						lodDescs.back().next = instanceLod;
					Lod& lod = obj.first->get_lod(instanceLod);
					lodDescs.push_back(lod.template get_descriptor<dev>());
					lodDescs.back().previous = prevLevel;
					lod.update_attribute_descriptor(lodDescs.back(), vertexAttribs, faceAttribs, sphereAttribs);
				}
			}

			lodDescs.back().previous = prevLevel;
		}
		// Allocate the device memory and copy over the descriptors
		auto& lodDevDesc = m_lodDevDesc.get<unique_device_ptr<dev, LodDescriptor<dev>[]>>();
		lodDevDesc = make_udevptr_array<dev, LodDescriptor<dev>>(lodDescs.size());
		copy(lodDevDesc.get(), lodDescs.data(), 0, lodDescs.size() * sizeof(LodDescriptor<dev>));

		sceneDescriptor.lods = lodDevDesc.get();
	}

	if(m_scenario.materials_dirty_reset() || !m_materials.template is_resident<dev>())
		load_materials<dev>();

	// Camera
	if(m_cameraDescChanged.template get<ChangedFlag<dev>>().changed) {
		get_camera()->get_parameter_pack(&sceneDescriptor.camera.get(), resolution);
	}

	// Light tree
	if(m_lightTreeDescChanged.template get<ChangedFlag<dev>>().changed) {
		sceneDescriptor.lightTree = m_lightTree.template acquire_const<dev>(m_boundingBox);
		m_lightTreeDescChanged.template get<ChangedFlag<dev>>().changed = false;
	}

	// TODO: query media/materials only if needed?
	sceneDescriptor.media = as<ConstArrayDevHandle_t<dev, materials::Medium>, ConstArrayDevHandle_t<dev, char>>(m_media.template acquire_const<dev>());
	sceneDescriptor.materials = as<ConstArrayDevHandle_t<dev, int>, ConstArrayDevHandle_t<dev, char>>(m_materials.template acquire_const<dev>());

	// Rebuild Instance BVH?
	if(geometryChanged) {
		auto scope = Profiler::instance().start<CpuProfileState>("build_instance_bvh");
		m_accelStruct.build(sceneDescriptor);
		m_cameraDescChanged.template get<ChangedFlag<dev>>().changed = true;
		m_lightTreeNeedsMediaUpdate.template get<ChangedFlag<dev>>().changed = true;
	}
	sceneDescriptor.accelStruct = m_accelStruct.template acquire_const<dev>();

	// Camera doesn't get a media-changed flag because it's relatively cheap to determine?
	if(m_cameraDescChanged.template get<ChangedFlag<dev>>().changed) {
		this->update_camera_medium(sceneDescriptor);
		m_cameraDescChanged.template get<ChangedFlag<dev>>().changed = false;
	}
	
	if(m_lightTreeNeedsMediaUpdate.template get<ChangedFlag<dev>>().changed) {
		m_lightTree.update_media(sceneDescriptor);
		m_lightTreeNeedsMediaUpdate.template get<ChangedFlag<dev>>().changed = false;
	}
	
	return sceneDescriptor;
}

void Scene::set_lights(std::vector<lights::PositionalLights>&& posLights,
					   std::vector<lights::DirectionalLight>&& dirLights) {
	m_lightTree.build(std::move(posLights), std::move(dirLights),
					  m_boundingBox);
	m_lightTreeDescChanged.for_each([](auto &elem) { elem.changed = true; });
	m_lightTreeNeedsMediaUpdate.for_each([](auto &elem) { elem.changed = true; });
}

void Scene::set_background(lights::Background& envLight) {
	if(&envLight != m_lightTree.get_envLight()) {
		m_lightTree.set_envLight(envLight);
		m_lightTreeDescChanged.for_each([](auto &elem) { elem.changed = true; });
	}
}

ConstCameraHandle Scene::get_camera() const noexcept {
	return m_scenario.get_camera();
}

template < Device dev >
void Scene::update_camera_medium(SceneDescriptor<dev>& descriptor) {
	if constexpr(dev == Device::CPU)
		update_camera_medium_cpu(descriptor);
	else if constexpr(dev == Device::CUDA)
		scene_detail::update_camera_medium_cuda(descriptor);
	else
		mAssert(false);
}

void Scene::update_camera_medium_cpu(SceneDescriptor<Device::CPU>& scene) {
	cameras::CameraParams& params = scene.camera.get();
	switch(params.type) {
		case cameras::CameraModel::PINHOLE:
			params.mediumIndex = materials::get_point_medium(scene, reinterpret_cast<cameras::PinholeParams&>(params).position);
			break;
		case cameras::CameraModel::FOCUS:
			params.mediumIndex = materials::get_point_medium(scene, reinterpret_cast<cameras::FocusParams&>(params).position);
			break;
		default: mAssert(false);
	}
}

template void Scene::load_materials<Device::CPU>();
template void Scene::load_materials<Device::CUDA>();
template void Scene::load_materials<Device::OPENGL>();
template void Scene::update_camera_medium<Device::CPU>(SceneDescriptor<Device::CPU>& descriptor);
template void Scene::update_camera_medium<Device::CUDA>(SceneDescriptor<Device::CUDA>& descriptor);
template void Scene::update_camera_medium<Device::OPENGL>(SceneDescriptor<Device::OPENGL>& descriptor);
template const SceneDescriptor<Device::CPU>& Scene::get_descriptor<Device::CPU>(const std::vector<const char*>&,
																			const std::vector<const char*>&,
																			const std::vector<const char*>&,
																			const ei::IVec2&);
template const SceneDescriptor<Device::CUDA>& Scene::get_descriptor<Device::CUDA>(const std::vector<const char*>&,
																			const std::vector<const char*>&,
																			const std::vector<const char*>&,
																			const ei::IVec2&);
template const SceneDescriptor<Device::OPENGL>& Scene::get_descriptor<Device::OPENGL>(const std::vector<const char*>&,
																			const std::vector<const char*>&,
																			const std::vector<const char*>&,
																			const ei::IVec2&);

}} // namespace mufflon::scene
