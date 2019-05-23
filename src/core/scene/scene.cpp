#include "scene.hpp"
#include "descriptors.hpp"
#include "scenario.hpp"
#include "world_container.hpp"
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
#include "core/scene/tessellation/cam_dist.hpp"
#include "core/scene/tessellation/uniform.hpp"
#include "mffloader/interface/interface.h"
#include "profiler/cpu_profiler.hpp"
#include <ei/3dintersection.hpp>

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
	const std::size_t MAT_SLOTS = m_scenario.get_num_material_slots();
	std::size_t offset = round_to_align<alignof(materials::MaterialDescriptorBase)>(sizeof(int) * MAT_SLOTS);
	for(MaterialIndex i = 0; i < m_scenario.get_num_material_slots(); ++i) {
		mAssert(offset <= std::numeric_limits<i32>::max());
		offsets.push_back(i32(offset));
		//offset += materials::get_handle_pack_size();
		offset += m_scenario.get_assigned_material(i)->get_descriptor_size(dev);
	}
	// Allocate the memory
	m_materials.resize(offset);
	m_alphaTextures.resize(sizeof(textures::ConstTextureDevHandle_t<dev>) * MAT_SLOTS);

	// Temporary storage to only copy once
	auto cpuTexHdlBuffer = std::make_unique<textures::ConstTextureDevHandle_t<dev>[]>(MAT_SLOTS);

	auto mem = m_materials.acquire<dev>();
	copy(mem, as<char>(offsets.data()), sizeof(int) * m_scenario.get_num_material_slots());
	// 2. Pass get all the material descriptors
	char buffer[materials::MAX_MATERIAL_DESCRIPTOR_SIZE()];
	for(MaterialIndex i = 0; i < MAT_SLOTS; ++i) {
		ConstMaterialHandle mat = m_scenario.get_assigned_material(i);
		mAssert(mat->get_descriptor_size(dev) <= materials::MAX_MATERIAL_DESCRIPTOR_SIZE());
		std::size_t size = mat->get_descriptor(dev, buffer) - buffer;
		copy(mem + size_t(offsets[i]), buffer, size);
		if(mat->get_alpha_texture() != nullptr)
			cpuTexHdlBuffer[i] = mat->get_alpha_texture()->template acquire_const<dev>();
		else
			cpuTexHdlBuffer[i] = textures::ConstTextureDevHandle_t<dev>{};
	}

	// Coyp the alpha texture handles
	copy((ArrayDevHandle_t<dev, textures::ConstTextureDevHandle_t<dev>>)(m_alphaTextures.acquire<dev>()),
		 cpuTexHdlBuffer.get(), sizeof(textures::ConstTextureDevHandle_t<dev>) * MAT_SLOTS);

	m_alphaTextures.mark_synced(dev);
	m_materials.mark_synced(dev); // Avoid overwrites with data from different devices.
}

template < Device dev >
const SceneDescriptor<dev>& Scene::get_descriptor(const std::vector<const char*>& vertexAttribs,
												  const std::vector<const char*>& faceAttribs,
												  const std::vector<const char*>& sphereAttribs) {
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
	// This part is affected by e.g. camera movement due to displacement mapping
	// (which uses tessellation), thus we may need to rebuild some BVHs even
	// if the geometry itself is unchanged
	// But since this is pretty expensive we only perform this if really necessary
	// (ie. some material actually has displacement mapping)
	// TODO: abstract the displacement heuristic and let it decide on which changes
	// it needs to be re-evaluated

	// TODO: this currently assumes that we do not add or alter geometry, which is clearly wrong
	// TODO: also needs to check for changed LoDs
	const bool geometryChanged = m_accelStruct.needs_rebuild();
	if(geometryChanged || sceneDescriptor.lodIndices != nullptr) {
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
		// This keeps track of instances for a given LoD
		std::unordered_map<u32, std::vector<InstanceHandle>> lodMapping;

		for(auto& obj : m_objects) {
			mAssert(obj.first != nullptr);
			mAssert(obj.second.size() != 0u);

			// First gather which LoDs have which instance
			lodMapping.clear();
			for(InstanceHandle inst : obj.second) {
				mAssert(inst != nullptr);
				const u32 instanceLod = m_scenario.get_effective_lod(inst);
				mAssert(instanceLod < obj.first->get_lod_slot_count());
				lodMapping[instanceLod].push_back(inst);

				instanceTransformations.push_back(inst->get_transformation_matrix());
				invInstanceTransformations.push_back(inst->get_inverse_transformation_matrix());
			}

			// Now that we know all instances a LoD has we can create the descriptors uniquely
			// and also perform displacement mapping if necessary
			u32 prevLevel = std::numeric_limits<u32>::max();
			for(const auto& mapping : lodMapping) {
				// Now we can do the per-LoD things like dispalcement mapping and fetching descriptors
				if(prevLevel != std::numeric_limits<u32>::max())
					lodDescs.back().next = mapping.first;
				Lod* lod = &obj.first->get_lod(mapping.first);

				lodDescs.push_back(lod->template get_descriptor<dev>());
				lodDescs.back().previous = prevLevel;
				lodAabbs.push_back(lod->get_bounding_box());
				if(!sameAttribs)
					lod->update_attribute_descriptor(lodDescs.back(), vertexAttribs, faceAttribs, sphereAttribs);

				// Gotta expand the scene bounding box
				for(InstanceHandle inst : mapping.second) {
					m_boundingBox = ei::Box(m_boundingBox, inst->get_bounding_box(mapping.first));
					lodIndices.push_back(lodCount);
				}
				++lodCount;
			}
			lodDescs.back().previous = prevLevel;

			instanceCount += obj.second.size();
		}

		// Allocate the device memory and copy over the descriptors
		auto& lodDevDesc = m_lodDevDesc.template get<unique_device_ptr<dev, LodDescriptor<dev>[]>>();
		lodDevDesc = make_udevptr_array<dev, LodDescriptor<dev>>(lodDescs.size());
		copy(lodDevDesc.get(), lodDescs.data(), lodDescs.size() * sizeof(LodDescriptor<dev>));

		auto& instTransformsDesc = m_instTransformsDesc.template get<unique_device_ptr<dev, ei::Mat3x4[]>>();
		instTransformsDesc = make_udevptr_array<dev, ei::Mat3x4>(instanceTransformations.size());
		copy(instTransformsDesc.get(), instanceTransformations.data(), sizeof(ei::Mat3x4) * instanceTransformations.size());

		auto& invInstTransformsDesc = m_invInstTransformsDesc.template get<unique_device_ptr<dev, ei::Mat3x4[]>>();
		invInstTransformsDesc = make_udevptr_array<dev, ei::Mat3x4>(invInstanceTransformations.size());
		copy(invInstTransformsDesc.get(), invInstanceTransformations.data(), sizeof(ei::Mat3x4) * invInstanceTransformations.size());

		auto& instLodIndicesDesc = m_instLodIndicesDesc.template get<unique_device_ptr<dev, u32[]>>();
		instLodIndicesDesc = make_udevptr_array<dev, u32>(lodIndices.size());
		copy<u32>(instLodIndicesDesc.get(), lodIndices.data(), sizeof(u32) * lodIndices.size());

		auto& lodAabbsDesc = m_lodAabbsDesc.template get<unique_device_ptr<dev, ei::Box[]>>();
		lodAabbsDesc = make_udevptr_array<dev, ei::Box>(lodAabbs.size());
		copy(lodAabbsDesc.get(), lodAabbs.data(), sizeof(ei::Box) * lodAabbs.size());

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
		// This keeps track of instances for a given LoD
		std::unordered_map<u32, std::vector<InstanceHandle>> lodMapping;

		for(auto& obj : m_objects) {
			mAssert(obj.first != nullptr);
			mAssert(obj.second.size() != 0u);

			// First gather which LoDs have which instance
			lodMapping.clear();
			for(InstanceHandle inst : obj.second) {
				mAssert(inst != nullptr);
				const u32 instanceLod = m_scenario.get_effective_lod(inst);
				mAssert(instanceLod < obj.first->get_lod_slot_count());
				lodMapping[instanceLod].push_back(inst);
			}

			// Now that we know all instances a LoD has we can create the descriptors uniquely
			u32 prevLevel = std::numeric_limits<u32>::max();
			for(const auto& mapping : lodMapping) {
				// Now we can do per-LoD things like displacement mapping
				if(prevLevel != std::numeric_limits<u32>::max())
					lodDescs.back().next = mapping.first;
				Lod& lod = obj.first->get_lod(mapping.first);
				lodDescs.push_back(lod.template get_descriptor<dev>());
				if(!sameAttribs)
					lod.update_attribute_descriptor(lodDescs.back(), vertexAttribs, faceAttribs, sphereAttribs);
			}
			lodDescs.back().previous = prevLevel;
		}
		// Allocate the device memory and copy over the descriptors
		auto& lodDevDesc = m_lodDevDesc.get<unique_device_ptr<dev, LodDescriptor<dev>[]>>();
		lodDevDesc = make_udevptr_array<dev, LodDescriptor<dev>>(lodDescs.size());
		copy(lodDevDesc.get(), lodDescs.data(), lodDescs.size() * sizeof(LodDescriptor<dev>));
		sceneDescriptor.lods = lodDevDesc.get();
	}

	// Materials
	if(m_scenario.materials_dirty_reset() || !m_materials.template is_resident<dev>())
		load_materials<dev>();
	// This query should be cheap. The above if already made the information resident.
	sceneDescriptor.media = (ArrayDevHandle_t<dev, materials::Medium>)(m_media.template acquire_const<dev>());
	sceneDescriptor.materials = (ArrayDevHandle_t<dev, int>)(m_materials.template acquire_const<dev>());
	sceneDescriptor.alphaTextures = (ArrayDevHandle_t<dev, textures::ConstTextureDevHandle_t<dev>>)(m_alphaTextures.template acquire_const<dev>());
	
	// Camera
	if(m_cameraDescChanged.template get<ChangedFlag<dev>>().changed) {
		get_camera()->get_parameter_pack(&sceneDescriptor.camera.get(), m_scenario.get_resolution(),
										 std::min(get_camera()->get_path_segment_count() - 1u, m_animationPathIndex));
	}

	// Light tree
	// TODO: rebuild light tree if area light got tessellated
	if(m_lightTreeDescChanged.template get<ChangedFlag<dev>>().changed) {
		sceneDescriptor.lightTree = m_lightTree.template acquire_const<dev>(m_boundingBox);
		m_lightTreeDescChanged.template get<ChangedFlag<dev>>().changed = false;
	}

	// Rebuild Instance BVH?
	if(m_accelStruct.needs_rebuild()) {
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
	// Need the materials for area lights
	if(m_scenario.materials_dirty_reset() || !m_materials.template is_resident<Device::CPU>())
		load_materials<Device::CPU>();
	const int* materials = as<int>(m_materials.template acquire_const<Device::CPU>());

	m_lightTree.build(std::move(posLights), std::move(dirLights),
					  m_boundingBox, materials);
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

bool Scene::retessellate(const float tessLevel) {
	if(!m_scenario.has_displacement_mapped_material())
		return false;

	// Track if we did perform tessellation and if we need to rebuild the light tree
	bool performedTessellation = false;
	bool needLighttreeRebuild = false;
	std::unordered_map<u32, std::vector<InstanceHandle>> lodMapping;
	std::vector<ei::Mat3x4> instTrans;
	for(auto& obj : m_objects) {
		lodMapping.clear();

		// First gather the instance-LoD-mapping
		for(InstanceHandle inst : obj.second) {
			const u32 instanceLod = m_scenario.get_effective_lod(inst);
			lodMapping[instanceLod].push_back(inst);
		}

		for(const auto& mapping : lodMapping) {
			// Check if we have displacement
			Lod* lod = &obj.first->get_lod(mapping.first);

			if(lod->has_displacement_mapping(m_scenario)) {
				// First get the relevant instance transformations
				instTrans.clear();
				for(InstanceHandle inst : mapping.second)
					instTrans.push_back(inst->get_transformation_matrix());
				// Then we may adaptively tessellate

				// TODO: more adequate tessellation level and more adequate tessellator
				tessellation::CameraDistanceOracle tess(tessLevel, get_camera(), m_animationPathIndex,
														m_scenario.get_resolution(), instTrans);
				// Check if we need to load the LoD back from disk (and hope it got cached)
				// TODO: would it be preferential to keep the untessellated LoD in memory as well?
				if(lod->was_displacement_mapping_applied()) {
					obj.first->remove_lod(mapping.first);
					if(!WorldContainer::instance().load_lod(*obj.first, mapping.first))
						throw std::runtime_error("Failed to re-load LoD for displacement map tessellation");
					lod = &obj.first->get_lod(mapping.first);
				}
				lod->displace(tess, m_scenario);
				lod->clear_accel_structure();
				performedTessellation = true;
				needLighttreeRebuild |= lod->is_emissive(m_scenario);
			}
		}
	}
	if(performedTessellation)
		m_accelStruct.mark_invalid();

	return needLighttreeRebuild;
}

template void Scene::load_materials<Device::CPU>();
template void Scene::load_materials<Device::CUDA>();
template void Scene::load_materials<Device::OPENGL>();
template void Scene::update_camera_medium<Device::CPU>(SceneDescriptor<Device::CPU>& descriptor);
template void Scene::update_camera_medium<Device::CUDA>(SceneDescriptor<Device::CUDA>& descriptor);
template void Scene::update_camera_medium<Device::OPENGL>(SceneDescriptor<Device::OPENGL>& descriptor);
template const SceneDescriptor<Device::CPU>& Scene::get_descriptor<Device::CPU>(const std::vector<const char*>&,
																				const std::vector<const char*>&,
																				const std::vector<const char*>&);
template const SceneDescriptor<Device::CUDA>& Scene::get_descriptor<Device::CUDA>(const std::vector<const char*>&,
																				  const std::vector<const char*>&,
																				  const std::vector<const char*>&);
template const SceneDescriptor<Device::OPENGL>& Scene::get_descriptor<Device::OPENGL>(const std::vector<const char*>&,
																					  const std::vector<const char*>&,
																					  const std::vector<const char*>&);

}} // namespace mufflon::scene
