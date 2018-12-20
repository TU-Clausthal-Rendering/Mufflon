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

namespace mufflon { namespace scene {

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
SceneDescriptor<dev> Scene::get_descriptor(const std::vector<const char*>& vertexAttribs,
										   const std::vector<const char*>& faceAttribs,
										   const std::vector<const char*>& sphereAttribs,
										   const ei::IVec2& resolution) {
	synchronize<dev>();
	std::vector<ObjectDescriptor<dev>> objectDescs;
	std::vector<ei::Mat3x4> instanceTransformations;
	std::vector<u32> objectIndices;
	std::vector<ei::Box> objectAabbs;
	// We need this to ensure we only create one descriptor per object
	std::unordered_map<Object*, u32> objectDescMap;

	// Create the object and instance descriptors hand-in-hand
	for(InstanceHandle inst : m_instances) {
		// Create the object descriptor, if not already present, and its index
		Object* objHdl = &inst->get_object();
		auto entry = objectDescMap.find(objHdl);
		if(entry == objectDescMap.end()) {
			entry = objectDescMap.emplace(objHdl, static_cast<u32>(objectDescs.size())).first;
			objectDescs.push_back(objHdl->get_descriptor<dev>(vertexAttribs, faceAttribs, sphereAttribs));
			objectAabbs.push_back(objHdl->get_bounding_box());
		}
		instanceTransformations.push_back(inst->get_transformation_matrix());
		objectIndices.push_back(entry->second);
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

	load_materials<dev>();

	// Camera
	CameraDescriptor camera{};
	m_camera->get_parameter_pack(&camera.get(), resolution);

	SceneDescriptor<dev> sceneDesc{
		camera,
		static_cast<u32>(objectDescs.size()),
		static_cast<i32>(m_instances.size()),
		m_boundingBox,
		objDevDesc.get(),
		AccelDescriptor{},
		instTransformsDesc.get(),
		instObjIndicesDesc.get(),
		objAabbsDesc.get(),
		m_lightTree.acquire_const<dev>(),
		as<materials::Medium>(m_media.acquire_const<dev>()),
		as<int>(m_materials.acquire_const<dev>())
	};

	// Rebuild Instance BVH?
	if(m_accelStruct.needs_rebuild<dev>()) {
		m_accelStruct.build(sceneDesc);
	}
	sceneDesc.accelStruct = m_accelStruct.acquire_const<dev>();

	// For each light determine the medium
	m_lightTree.update_media(sceneDesc);

	// For the camera as well
	this->update_camera_medium(sceneDesc);

	return sceneDesc;
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
template SceneDescriptor<Device::CPU> Scene::get_descriptor<Device::CPU>(const std::vector<const char*>&,
																			const std::vector<const char*>&,
																			const std::vector<const char*>&,
																			const ei::IVec2&);
template SceneDescriptor<Device::CUDA> Scene::get_descriptor<Device::CUDA>(const std::vector<const char*>&,
																			const std::vector<const char*>&,
																			const std::vector<const char*>&,
																			const ei::IVec2&);
/*template SceneDescriptor<Device::OPENGL> Scene::get_descriptor<Device::OPENGL>(const std::vector<const char*>&,
																			const std::vector<const char*>&,
																			const std::vector<const char*>&,
																			const ei::IVec2&);*/

}} // namespace mufflon::scene