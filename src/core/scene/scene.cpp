#include "scene.hpp"
#include "descriptors.hpp"
#include "core/cameras/camera.hpp"
#include "core/cameras/pinhole.hpp"
#include "core/cameras/focus.hpp"
#include "core/scene/materials/medium.hpp"
#include "core/scene/materials/material.hpp"
#include "core/scene/materials/point_medium.hpp"
#include "core/scene/geometry/polygon.hpp"
#include "core/scene/geometry/sphere.hpp"
#include "core/scene/lights/background.hpp"
#include "profiler/cpu_profiler.hpp"

namespace mufflon { namespace scene {

bool Scene::is_sane() const noexcept {
	if(m_camera == nullptr)
		return false;
	if(m_lightTree.get_envLight()->get_type() != lights::BackgroundType::ENVMAP) {
		// No envmap: we need some kind of light
		if(m_lightTree.get_light_count() == 0u)
			return false;
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
	std::vector<int> offsets;
	std::size_t offset = sizeof(int) * m_materialsRef.size(); // Store in one block -> table size is offset of first material
	for(const auto& mat : m_materialsRef) {
		mAssert(offset <= std::numeric_limits<i32>::max());
		offsets.push_back(i32(offset));
		//offset += materials::get_handle_pack_size();
		offset += mat->get_descriptor_size(dev);
	}
	// Allocate the memory
	m_materials.resize(offset);
	char* mem = m_materials.acquire<dev>();
	copy(mem, as<char>(offsets.data()), sizeof(int) * m_materialsRef.size());
	// 2. Pass get all the material descriptors
	char buffer[materials::MAX_MATERIAL_PARAMETER_SIZE];
	int i = 0;
	for(const auto& mat : m_materialsRef) {
		mAssert(mat->get_descriptor_size(dev) <= materials::MAX_MATERIAL_PARAMETER_SIZE);
		std::size_t size = mat->get_descriptor(dev, buffer) - buffer;
		copy(mem + offsets[i], buffer, size);
		++i;
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
	bool sameAttribs = m_lastVertexAttribs.size() == vertexAttribs.size()
		&& m_lastFaceAttribs.size() == faceAttribs.size()
		&& m_lastSphereAttribs.size() == sphereAttribs.size();
	if(sameAttribs)
		for(auto name : vertexAttribs) {
			if(std::find_if(m_lastVertexAttribs.cbegin(), m_lastVertexAttribs.cend(), [name](const char* n) { return std::strcmp(name, n) != 0; }) != m_lastVertexAttribs.cend()) {
				sameAttribs = false;
				break;
			}
		}
	if(sameAttribs)
		for(auto name : faceAttribs) {
			if(std::find_if(m_lastFaceAttribs.cbegin(), m_lastFaceAttribs.cend(), [name](const char* n) { return std::strcmp(name, n) != 0; }) != m_lastFaceAttribs.cend()) {
				sameAttribs = false;
				break;
			}
		}
	if(sameAttribs)
		for(auto name : sphereAttribs) {
			if(std::find_if(m_lastSphereAttribs.cbegin(), m_lastSphereAttribs.cend(), [name](const char* n) { return std::strcmp(name, n) != 0; }) != m_lastSphereAttribs.cend()) {
				sameAttribs = false;
				break;
			}
		}

	// Check if we need to update the object descriptors
	// TODO: this currently assumes that we do not add or alter geometry, which is clearly wrong
	const bool geometryChanged = m_accelStruct.needs_rebuild<dev>();
	if(geometryChanged) {
		std::vector<ei::Mat3x4> instanceTransformations;
		std::vector<u32> objectIndices;
		std::vector<ObjectDescriptor<dev>> objectDescs;
		std::vector<ei::Box> objectAabbs;

		// Create the object and instance descriptors
		std::size_t instanceCount = 0u;
		u32 objectCount = 0u;
		for(auto& obj : m_objects) {
			mAssert(obj.first != nullptr);
			objectDescs.push_back(obj.first->get_descriptor<dev>());
			objectAabbs.push_back(obj.first->get_bounding_box());

			for(InstanceHandle inst : obj.second) {
				mAssert(inst != nullptr);
				// TODO: crash
				instanceTransformations.push_back(inst->get_transformation_matrix());
				objectIndices.push_back(objectCount);
			}

			if(!sameAttribs)
				obj.first->update_attribute_descriptor(objectDescs.back(), vertexAttribs, faceAttribs, sphereAttribs);

			instanceCount += obj.second.size();
			++objectCount;
		}

		// Allocate the device memory and copy over the descriptors
		auto& objDevDesc = m_objDevDesc.get<unique_device_ptr<dev, ObjectDescriptor<dev>[]>>();
		objDevDesc = make_udevptr_array<dev, ObjectDescriptor<dev>>(objectDescs.size());
		copy(objDevDesc.get(), objectDescs.data(), objectDescs.size() * sizeof(ObjectDescriptor<dev>));

		auto& instTransformsDesc = m_instTransformsDesc.get<unique_device_ptr<dev, ei::Mat3x4[]>>();
		instTransformsDesc = make_udevptr_array<dev, ei::Mat3x4>(instanceTransformations.size());
		copy(instTransformsDesc.get(), instanceTransformations.data(), sizeof(ei::Mat3x4) * instanceTransformations.size());

		auto& instObjIndicesDesc = m_instObjIndicesDesc.get<unique_device_ptr<dev, u32[]>>();
		instObjIndicesDesc = make_udevptr_array<dev, u32>(objectIndices.size());
		copy(instObjIndicesDesc.get(), objectIndices.data(), sizeof(u32) * objectIndices.size());

		auto& objAabbsDesc = m_objAabbsDesc.get<unique_device_ptr<dev, ei::Box[]>>();
		objAabbsDesc = make_udevptr_array<dev, ei::Box>(objectAabbs.size());
		copy(objAabbsDesc.get(), objectAabbs.data(), sizeof(ei::Box) * objectAabbs.size());

		sceneDescriptor.numObjects = static_cast<u32>(objectDescs.size());
		sceneDescriptor.numInstances = static_cast<i32>(instanceCount);
		sceneDescriptor.aabb = m_boundingBox;
		sceneDescriptor.objects = objDevDesc.get();
		sceneDescriptor.aabbs = objAabbsDesc.get();
		sceneDescriptor.transformations = instTransformsDesc.get();
		sceneDescriptor.objectIndices = instObjIndicesDesc.get();
	} else if(!sameAttribs) {
		// Only update the descriptors and reupload them
		std::vector<ObjectDescriptor<dev>> objectDescs;
		for(auto& obj : m_objects) {
			objectDescs.push_back(obj.first->get_descriptor<dev>());
			obj.first->update_attribute_descriptor(objectDescs.back(), vertexAttribs, faceAttribs, sphereAttribs);
		}
		// Allocate the device memory and copy over the descriptors
		auto& objDevDesc = m_objDevDesc.get<unique_device_ptr<dev, ObjectDescriptor<dev>[]>>();
		objDevDesc = make_udevptr_array<dev, ObjectDescriptor<dev>>(objectDescs.size());
		copy(objDevDesc.get(), objectDescs.data(), objectDescs.size() * sizeof(ObjectDescriptor<dev>));

		sceneDescriptor.objects = objDevDesc.get();
	}

	load_materials<dev>();

	// Camera
	if(m_cameraDescChanged) {
		m_camera->get_parameter_pack(&sceneDescriptor.camera.get(), resolution);
		m_cameraDescChanged = false;
	}

	// Light tree
	if(m_lightTreeDescChanged) {
		sceneDescriptor.lightTree = m_lightTree.acquire_const<dev>(m_boundingBox);
		m_lightTreeDescChanged = false;
	}

	// TODO: query media/materials only if needed?
	sceneDescriptor.media = as<materials::Medium>(m_media.acquire_const<dev>());
	sceneDescriptor.materials = as<int>(m_materials.acquire_const<dev>());

	// Rebuild Instance BVH?
	if(geometryChanged) {
		auto scope = Profiler::instance().start<CpuProfileState>("build_instance_bvh");
		m_accelStruct.build(sceneDescriptor);
		sceneDescriptor.accelStruct = m_accelStruct.acquire_const<dev>();
		// For each light determine the medium
		m_lightTree.update_media(sceneDescriptor);
		// For the camera as well
		this->update_camera_medium(sceneDescriptor);
	}

	// Camera doesn't get a media-changed flag because it's relatively cheap to determine?
	if(m_cameraDescChanged)
		this->update_camera_medium(sceneDescriptor);
	
	if(m_lightTreeNeedsMediaUpdate) {
		m_lightTree.update_media(sceneDescriptor);
		m_lightTreeNeedsMediaUpdate = false;
	}

	return sceneDescriptor;
}

void Scene::set_lights(std::vector<lights::PositionalLights>&& posLights,
				std::vector<lights::DirectionalLight>&& dirLights) {
	m_lightTree.build(std::move(posLights), std::move(dirLights),
						m_boundingBox);
	m_lightTreeDescChanged = true;
	m_lightTreeNeedsMediaUpdate = true;
}

void Scene::set_background(lights::Background& envLight) {
	if(&envLight != m_lightTree.get_envLight()) {
		m_lightTree.set_envLight(envLight);
		m_lightTreeDescChanged = true;
	}
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
//template void Scene::load_materials<Device::OPENGL>();
template void Scene::update_camera_medium<Device::CPU>(SceneDescriptor<Device::CPU>& descriptor);
template void Scene::update_camera_medium<Device::CUDA>(SceneDescriptor<Device::CUDA>& descriptor);
//template void update_camera_medium< Device::OPENGL>(const SceneDescriptor<Device::OPENGL>& descriptor);
template const SceneDescriptor<Device::CPU>& Scene::get_descriptor<Device::CPU>(const std::vector<const char*>&,
																			const std::vector<const char*>&,
																			const std::vector<const char*>&,
																			const ei::IVec2&);
template const SceneDescriptor<Device::CUDA>& Scene::get_descriptor<Device::CUDA>(const std::vector<const char*>&,
																			const std::vector<const char*>&,
																			const std::vector<const char*>&,
																			const ei::IVec2&);
/*template const SceneDescriptor<Device::OPENGL>& Scene::get_descriptor<Device::OPENGL>(const std::vector<const char*>&,
																			const std::vector<const char*>&,
																			const std::vector<const char*>&,
																			const ei::IVec2&);*/

}} // namespace mufflon::scene
