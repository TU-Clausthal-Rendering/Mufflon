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
#include "profiler/cpu_profiler.hpp"
#include <ei/3dintersection.hpp>
#ifdef __cpp_lib_execution
#include <execution>
#else // __cpp_lib_execution
#warning "Parallel STL algorithms are not supported by your standard library. This may result in a slight renderer slowdown"
#endif // __cpp_lib_execution

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
		mAssert(offset <= static_cast<std::size_t>(std::numeric_limits<i32>::max()));
		offsets.push_back(i32(offset));
		//offset += materials::get_handle_pack_size();
		offset += m_scenario.get_assigned_material(i)->get_descriptor_size(dev);
	}
	// Allocate the memory
	// TODO: this is a workaround to stop erasing previous material memory when other devices already got some
	if(m_materials.size() < offset)
		m_materials.resize(offset);
	m_alphaTextures.resize(sizeof(textures::ConstTextureDevHandle_t<dev>) * MAT_SLOTS);

	// Temporary storage to only copy once
	auto cpuTexHdlBuffer = std::make_unique<textures::ConstTextureDevHandle_t<dev>[]>(MAT_SLOTS);

	auto mem = m_materials.acquire<dev>(false);
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
}

template < Device dev >
const SceneDescriptor<dev>& Scene::get_descriptor(const std::vector<AttributeIdentifier>& vertexAttribs,
												  const std::vector<AttributeIdentifier>& faceAttribs,
												  const std::vector<AttributeIdentifier>& sphereAttribs) {
	synchronize<dev>();
	SceneDescriptor<dev>& sceneDescriptor = m_descStore.template get<SceneDescriptor<dev>>();
	if constexpr(dev == Device::OPENGL) {
		sceneDescriptor.cpuDescriptor = &get_descriptor<Device::CPU>(vertexAttribs, faceAttribs, sphereAttribs);
	}

	// Check if we need to update attributes
	auto& lastVertexAttribs = m_lastAttributeNames.template get<AttributeNames<dev>>().lastVertexAttribs;
	auto& lastFaceAttribs = m_lastAttributeNames.template get<AttributeNames<dev>>().lastFaceAttribs;
	auto& lastSphereAttribs = m_lastAttributeNames.template get<AttributeNames<dev>>().lastSphereAttribs;
	bool sameAttribs = lastVertexAttribs.size() == vertexAttribs.size()
		&& lastFaceAttribs.size() == faceAttribs.size()
		&& lastSphereAttribs.size() == sphereAttribs.size();
	if(sameAttribs)
		for(auto ident : vertexAttribs) {
			if(std::find_if(lastVertexAttribs.cbegin(), lastVertexAttribs.cend(), [ident](const auto& n) { return ident == n; }) != lastVertexAttribs.cend()) {
				sameAttribs = false;
				lastVertexAttribs = vertexAttribs;
				break;
			}
		}
	if(sameAttribs)
		for(auto ident : faceAttribs) {
			if(std::find_if(lastFaceAttribs.cbegin(), lastFaceAttribs.cend(), [ident](const auto& n) { return ident == n; }) != lastFaceAttribs.cend()) {
				sameAttribs = false;
				lastFaceAttribs = faceAttribs;
				break;
			}
		}
	if(sameAttribs)
		for(auto ident : sphereAttribs) {
			if(std::find_if(lastSphereAttribs.cbegin(), lastSphereAttribs.cend(), [ident](const auto& n) { return ident == n; }) != lastSphereAttribs.cend()) {
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
	if(geometryChanged || sceneDescriptor.lodIndices == nullptr || !sameAttribs) {
		// Invalidate other descriptors
		if(geometryChanged)
			m_descStore.for_each([](auto& elem) { elem.lodIndices = {}; });

		std::unique_ptr<u32[]> lodIndices;
		std::unique_ptr<LodDescriptor<dev>[]> lodDescs;
		std::unique_ptr<ei::Box[]> lodAabbs;

		m_boundingBox.max = ei::Vec3{ -std::numeric_limits<float>::max() };
		m_boundingBox.min = ei::Vec3{ std::numeric_limits<float>::max() };

		// Create the object and instance descriptors
		// This keeps track of instances for a given LoD

		std::vector<std::vector<u32>> usedLods(get_max_thread_num());
		//std::vector<std::unordered_map<u32, u32>> usedLods(get_max_thread_num());

		// Preallocate the CPU-side arrays so we can multi-thread
		const auto totalInstanceCount = m_worldToInstanceTransformation.size();
		lodIndices = std::make_unique<u32[]>(totalInstanceCount);
		// All non-present instances get a LoD-index of 0xFFFFFFFF
		std::memset(lodIndices.get(), std::numeric_limits<u32>::max(),
					sizeof(u32) * totalInstanceCount);
		// TODO: this is a conservative estimate and wastes some memory for scenes
		// with many LoDs
		std::size_t descCountEstimate = 0u;
		for(auto& obj : m_objects)
			descCountEstimate += obj.first->get_lod_slot_count();
		lodDescs = std::make_unique<LodDescriptor<dev>[]>(descCountEstimate);
		lodAabbs = std::make_unique<ei::Box[]>(descCountEstimate);

		// We parallelize this part for cases with many objects
		const int objCount = static_cast<int>(m_objects.size());
		std::atomic_uint32_t lodIndex = 0u;

		const auto t0 = std::chrono::high_resolution_clock::now();
		const bool objLevelParallelism = dev == Device::CPU && objCount >= 50;
#pragma PARALLEL_FOR_COND_DYNAMIC(objLevelParallelism)
		for(int o = 0; o < objCount; ++o) {
			auto& obj = *(m_objects.begin() + o);
			mAssert(obj.first != nullptr);
			mAssert(obj.second.count != 0u);

			// First gather which LoDs of this objects are used and
			// collect the instance transformations
			const auto lodSlots = static_cast<u32>(obj.first->get_lod_slot_count());
			auto& currUsedLods = usedLods[get_current_thread_idx()];
			currUsedLods.clear();
			currUsedLods.resize(lodSlots);
			std::fill(currUsedLods.begin(), currUsedLods.end(), std::numeric_limits<u32>::max());
			const auto endIndex = obj.second.offset + obj.second.count;
			u32 lodCounter = 0u;
			for(std::size_t i = obj.second.offset; i < endIndex; ++i) {
				const auto inst = m_instances[i];
				mAssert(inst != nullptr);
				const u32 instanceLod = m_scenario.get_effective_lod(inst);
				mAssert(instanceLod < lodSlots);
				if(currUsedLods[instanceLod] == std::numeric_limits<u32>::max())
					currUsedLods[instanceLod] = lodCounter++;
			}

			// Reserve the proper amount of LoD descriptors
			const auto threadLodIndex = lodIndex.fetch_add(lodCounter);
			auto currLodIndex = threadLodIndex;
			// Now that we know all instances a LoD has we can create the descriptors uniquely
			// and also perform displacement mapping if necessary
			u32 prevLevel = std::numeric_limits<u32>::max();
			for(u32 i = 0u; i < lodSlots; i++) {
				if(currUsedLods[i] == std::numeric_limits<u32>::max())
					continue;
				// Now we can do the per-LoD things like displacement mapping and fetching descriptors
				if(prevLevel != std::numeric_limits<u32>::max())
					lodDescs[currLodIndex - 1u].next = i;
				Lod* lod = &obj.first->get_lod(i);

				// Determine if it's worth it to use a parallel build
				lodDescs[currLodIndex] = lod->template get_descriptor<dev>(objLevelParallelism);
				lodDescs[currLodIndex].previous = prevLevel;
				lodAabbs[currLodIndex] = lod->get_bounding_box();
				if(!sameAttribs)
					lod->update_attribute_descriptor(lodDescs[currLodIndex], vertexAttribs, faceAttribs, sphereAttribs);
				++currLodIndex;
			}
			if(!currUsedLods.empty())
				lodDescs[currLodIndex - 1u].previous = prevLevel;

			// Now write the proper instance indices and expand bounding box
			for(std::size_t i = obj.second.offset; i < endIndex; ++i) {
				const auto inst = m_instances[i];
				const u32 instanceLod = m_scenario.get_effective_lod(inst);
				const u32 index = currUsedLods[instanceLod];

				const auto instanceIndex = inst->get_index();
				lodIndices[instanceIndex] = threadLodIndex + index;
				const auto aabb = inst->get_bounding_box(instanceLod, m_worldToInstanceTransformation[inst->get_index()]);
				m_boundingBox = ei::Box(m_boundingBox, aabb);
			}
		}

		sceneDescriptor.validInstanceIndex = m_instances.front()->get_index();
		const auto t1 = std::chrono::high_resolution_clock::now();
		logInfo("[Scene::get_descriptor] Build descriptors for ", lodIndex.load(),
				" LoDs in ", std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count(),
				"ms");

		// Allocate the device memory and copy over the descriptors
		// Some of these don't need to be copied can be just taken when we're on the CPU
		auto& lodDevDesc = m_lodDevDesc.template get<unique_device_ptr<NotGl<dev>, LodDescriptor<dev>[]>>();
		auto& lodAabbsDesc = m_lodAabbsDesc.template get<unique_device_ptr<dev, ei::Box[]>>();
		if constexpr(dev == Device::CPU) {
			lodDevDesc.reset(Allocator<Device::CPU>::realloc(lodDescs.release(), descCountEstimate, lodIndex.load()));
			lodAabbsDesc.reset(Allocator<Device::CPU>::realloc(lodAabbs.release(), descCountEstimate, lodIndex.load()));
			sceneDescriptor.worldToInstance = m_worldToInstanceTransformation.data();
		} else {
			auto& instTransformsDesc = m_instTransformsDesc.template get<unique_device_ptr<dev, ei::Mat3x4[]>>();
			lodDevDesc = make_udevptr_array<NotGl<dev>, LodDescriptor<dev>>(lodIndex.load());
			lodAabbsDesc = make_udevptr_array<dev, ei::Box>(lodIndex.load());
			copy(lodDevDesc.get(), lodDescs.get(), sizeof(LodDescriptor<dev>) * lodIndex.load());
			copy(lodAabbsDesc.get(), lodAabbs.get(), sizeof(ei::Box) * lodIndex.load());

			// Transformation matrices only have to be uploaded for non-CPU devices
			instTransformsDesc = make_udevptr_array<dev, ei::Mat3x4>(totalInstanceCount);
			copy(instTransformsDesc.get(), m_worldToInstanceTransformation.data(), sizeof(ei::Mat3x4) * totalInstanceCount);
			sceneDescriptor.worldToInstance = instTransformsDesc.get();

			if constexpr(dev == Device::OPENGL) {
				// Only OpenGL renderers really need the instance-to-world transformations
				auto& instToWorldTransformsDesc = m_instToWorldTransformsDesc.template get<unique_device_ptr<dev, ei::Mat3x4[]>>();
				instToWorldTransformsDesc = make_udevptr_array<dev, ei::Mat3x4>(m_instances.size());
				// Compute the inverse instance transformations
				auto instToWorldTransformation = std::make_unique<ei::Mat3x4[]>(totalInstanceCount);

				// TODO: enable parallelism (OpenMP compiling bugs out)
				const auto begin = m_worldToInstanceTransformation.data();
				const auto end = begin + totalInstanceCount;
				const auto outBegin = instToWorldTransformation.get();
				std::transform(
#ifdef __cpp_lib_execution
							   std::execution::par_unseq,
#endif // __cpp_lib_execution
							   begin, end, outBegin, [](const ei::Mat3x4& matrix) {
					return InstanceData<Device::CPU>::compute_instance_to_world_transformation(matrix);
				});

				copy(instToWorldTransformsDesc.get(), instToWorldTransformation.get(), sizeof(ei::Mat3x4) * m_instances.size());
				sceneDescriptor.instanceToWorld = instToWorldTransformsDesc.get();
			}
		}

		if constexpr(dev != Device::OPENGL) {
			auto& instLodIndicesDesc = m_instLodIndicesDesc.template get<unique_device_ptr<dev, u32[]>>();
			if constexpr(dev == Device::CPU) {
				instLodIndicesDesc.reset(lodIndices.release());
			} else {
				instLodIndicesDesc = make_udevptr_array<dev, u32>(totalInstanceCount);
				copy<u32>(instLodIndicesDesc.get(), lodIndices.get(), sizeof(u32) * totalInstanceCount);
			}
			sceneDescriptor.lodIndices = instLodIndicesDesc.get();
		} else {
			// We cannot create a new lodIndices array here, because the type for OpenGL is the same as for the CPU side,
			// which then overwrites the array in m_instLodIndicesDesc and leaves a dangling pointer in the CPU descriptor
			sceneDescriptor.lodIndices = sceneDescriptor.cpuDescriptor->lodIndices;
		}

		sceneDescriptor.numLods = static_cast<u32>(lodIndex.load());
		sceneDescriptor.numInstances = static_cast<i32>(totalInstanceCount);
		sceneDescriptor.diagSize = len(m_boundingBox.max - m_boundingBox.min);
		sceneDescriptor.aabb = m_boundingBox;
		sceneDescriptor.lods = lodDevDesc.get();
		sceneDescriptor.aabbs = lodAabbsDesc.get();
	}

	// Materials
	if(m_scenario.materials_dirty_reset() || !m_materials.template is_resident<dev>())
		load_materials<dev>();
	// This query should be cheap. The above if already made the information resident.
	sceneDescriptor.media = (ArrayDevHandle_t<dev, materials::Medium>)(m_media.template acquire_const<dev>());
	sceneDescriptor.materials = (ArrayDevHandle_t<dev, int>)(m_materials.template acquire_const<dev>(false));
	sceneDescriptor.alphaTextures = (ArrayDevHandle_t<dev, textures::ConstTextureDevHandle_t<dev>>)(m_alphaTextures.template acquire_const<dev>(false));
	
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

    if(dev != Device::OPENGL) {
		// Rebuild Instance BVH?
		if (m_accelStruct.needs_rebuild()) {
			auto scope = Profiler::instance().start<CpuProfileState>("build_instance_bvh");

			const auto t0 = std::chrono::high_resolution_clock::now();
			m_accelStruct.build(sceneDescriptor, static_cast<u32>(m_instances.size()));
			m_cameraDescChanged.template get<ChangedFlag<dev>>().changed = true;
			m_lightTreeNeedsMediaUpdate.template get<ChangedFlag<dev>>().changed = true;

			const auto t1 = std::chrono::high_resolution_clock::now();
			logInfo("[Scene::get_descriptor] Build instance BVH for ", m_instances.size(),
					" instances in ", std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count(),
					"ms");
		}
		sceneDescriptor.accelStruct = m_accelStruct.template acquire_const<dev>();
    }
	
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
	if(m_scenario.materials_dirty_reset() || !m_materials.template is_resident<Device::CPU>()) {
		load_materials<Device::CPU>();
		// TODO: this is currently a workaround until we have have a resource class
		// that semantically must carry the same information but deviated byte-wise
		// (for e.g. texture handles)
		m_materials.mark_changed(Device::CPU);
	}
	const int* materials = as<int>(m_materials.template acquire_const<Device::CPU>());

	m_lightTree.build(std::move(posLights), std::move(dirLights),
					  m_boundingBox, materials);
	m_lightTreeDescChanged.for_each([](auto &elem) { elem.changed = true; });
	m_lightTreeNeedsMediaUpdate.for_each([](auto &elem) { elem.changed = true; });
}

void Scene::set_background(lights::Background& envLight) {
	m_lightTree.set_envLight(envLight);
	m_lightTreeDescChanged.for_each([](auto &elem) { elem.changed = true; });
}

ConstCameraHandle Scene::get_camera() const noexcept {
	return m_scenario.get_camera();
}

template < Device dev >
void Scene::update_camera_medium(SceneDescriptor<dev>& descriptor) {
	if constexpr (dev == Device::CPU)
		update_camera_medium_cpu(descriptor);
	else if constexpr (dev == Device::CUDA)
		scene_detail::update_camera_medium_cuda(descriptor);
	else if constexpr (dev == Device::OPENGL); // TODO gl?
		//update_camera_medium_cpu(descriptor);
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
	if(!m_scenario.has_displacement_mapped_material() && !m_scenario.has_object_tessellation())
		return false;

	// Track if we did perform tessellation and if we need to rebuild the light tree
	bool performedTessellation = false;
	bool needLighttreeRebuild = false;
	// TODO: get rid of the mapping (too low performance)
	std::unordered_map<u32, std::vector<InstanceHandle>> lodMapping;
	std::vector<ei::Mat3x4> instTrans;
	for(auto& obj : m_objects) {
		lodMapping.clear();

		// First gather the instance-LoD-mapping
		const auto endIndex = obj.second.offset + obj.second.count;
		for(std::size_t i = obj.second.offset; i < endIndex; ++i) {
			const auto inst = m_instances[i];
			const u32 instanceLod = m_scenario.get_effective_lod(inst);
			lodMapping[instanceLod].push_back(inst);
		}

		for(const auto& mapping : lodMapping) {
			// Check if we have displacement or plain tessellation
			Lod* lod = &obj.first->get_lod(mapping.first);

			if(lod->is_displaced(m_scenario)) {
				// First get the relevant instance transformations
				instTrans.clear();
				/*for(InstanceHandle inst : mapping.second)
					instTrans.push_back(inst->get_transformation_matrix());*/
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
				performedTessellation = true;
				needLighttreeRebuild |= lod->is_emissive(m_scenario);
			} else {
				// We may not have displacement mapping, but still may wanna tessellate the object
				const auto objTessInfo = m_scenario.get_tessellation_info(obj.first);
				if(objTessInfo.has_value()) {
					// Reload the LoD if it has been modified previously
					if(lod->was_displacement_mapping_applied()) {
						obj.first->remove_lod(mapping.first);
						if(!WorldContainer::instance().load_lod(*obj.first, mapping.first))
							throw std::runtime_error("Failed to re-load LoD for displacement map tessellation");
						lod = &obj.first->get_lod(mapping.first);
					}

					const auto objTessLevel = objTessInfo->level.value_or(tessLevel);
					if(objTessInfo->adaptive) {
						// First get the relevant instance transformations
						instTrans.clear();
						/*for(InstanceHandle inst : mapping.second)
							instTrans.push_back(inst->get_transformation_matrix());*/
						tessellation::CameraDistanceOracle oracle{ objTessLevel, get_camera(), m_animationPathIndex,
																	  m_scenario.get_resolution(), instTrans };
						lod->tessellate(oracle, &m_scenario, objTessInfo->usePhong);
					} else {
						tessellation::Uniform oracle{ u32(objTessLevel), u32(objTessLevel) };
						lod->tessellate(oracle, nullptr, objTessInfo->usePhong);
					}
					performedTessellation = true;
					needLighttreeRebuild |= lod->is_emissive(m_scenario);
				}
			}
		}
	}
	if(performedTessellation)
		m_accelStruct.mark_invalid();

	return needLighttreeRebuild;
}


void Scene::compute_curvature() {
	for(auto& obj : m_objects) {
		for(u32 level = 0; level < obj.first->get_lod_slot_count(); ++level) {
			if(obj.first->has_lod_available(level)) {
				Lod& lod = obj.first->get_lod(level);
				geometry::Polygons& polygons = lod.get_geometry<geometry::Polygons>();
				polygons.compute_curvature();
			}
		}
	}
}

void Scene::remove_curvature() {
	for(auto& obj : m_objects) {
		for(u32 level = 0; level < obj.first->get_lod_slot_count(); ++level) {
			if(obj.first->has_lod_available(level)) {
				Lod& lod = obj.first->get_lod(level);
				geometry::Polygons& polygons = lod.get_geometry<geometry::Polygons>();
				try {
					polygons.remove_curvature();
				} catch(...) {}
			}
		}
	}
}


template void Scene::load_materials<Device::CPU>();
template void Scene::load_materials<Device::CUDA>();
template void Scene::load_materials<Device::OPENGL>();
template void Scene::update_camera_medium<Device::CPU>(SceneDescriptor<Device::CPU>& descriptor);
template void Scene::update_camera_medium<Device::CUDA>(SceneDescriptor<Device::CUDA>& descriptor);
template void Scene::update_camera_medium<Device::OPENGL>(SceneDescriptor<Device::OPENGL>& descriptor);
template const SceneDescriptor<Device::CPU>& Scene::get_descriptor<Device::CPU>(const std::vector<AttributeIdentifier>&,
																				const std::vector<AttributeIdentifier>&,
																				const std::vector<AttributeIdentifier>&);
template const SceneDescriptor<Device::CUDA>& Scene::get_descriptor<Device::CUDA>(const std::vector<AttributeIdentifier>&,
																				  const std::vector<AttributeIdentifier>&,
																				  const std::vector<AttributeIdentifier>&);
template const SceneDescriptor<Device::OPENGL>& Scene::get_descriptor<Device::OPENGL>(const std::vector<AttributeIdentifier>&,
																					  const std::vector<AttributeIdentifier>&,
																					  const std::vector<AttributeIdentifier>&);

}} // namespace mufflon::scene
