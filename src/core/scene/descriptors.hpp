#pragma once

#include "util/int_types.hpp"
#include "lights/light_tree.hpp"
#include "core/memory/residency.hpp"
#include "core/scene/accel_structs/accel_struct_info.hpp"
#include "core/cameras/camera.hpp"
#include "handles.hpp"
#include <ei/vector.hpp>
#include <ei/3dtypes.hpp>

namespace mufflon {

enum class Device : unsigned char;

namespace scene {

/**
 * These descriptors will be filled with the proper data when a renderer
 * requests the scene data for a given device. As such, they are ONLY
 * valid on said device. They also may be outdated if a different scenario
 * gets created.
 */

// Exchangable acceleration structure header
struct AccelDescriptor {
	accel_struct::AccelType type { accel_struct::AccelType::NONE };
	u8 accelParameters[accel_struct::MAX_ACCEL_STRUCT_PARAMETER_SIZE];
};

// Geometric descriptors
template < Device dev >
struct PolygonsDescriptor {
	u32 numVertices;
	u32 numTriangles;
	u32 numQuads;
	u32 numVertexAttributes;
	u32 numFaceAttributes;
	ConstArrayDevHandle_t<dev, ei::Vec3> vertices;
	ConstArrayDevHandle_t<dev, ei::Vec3> normals;
	ConstArrayDevHandle_t<dev, ei::Vec2> uvs;
	// Ordered and per face: first come triangles, then come quads
	ConstArrayDevHandle_t<dev, u16> matIndices;
	// First come triangles, then come quads
	ConstArrayDevHandle_t<dev, u32> vertexIndices;
	// Access to these must be followed by a mark_dirty/aquire after
	ConstArrayDevHandle_t<dev, ArrayDevHandle_t<dev, void>> vertexAttributes;
	ConstArrayDevHandle_t<dev, ArrayDevHandle_t<dev, void>> faceAttributes;
};

template < Device dev >
struct SpheresDescriptor {
	u32 numSpheres;
	u32 numAttributes;
	ConstArrayDevHandle_t<dev, ei::Sphere> spheres;
	ConstArrayDevHandle_t<dev, u16> matIndices;
	// Access to these must be followed by a mark_dirty/aquire after
	ConstArrayDevHandle_t<dev, ArrayDevHandle_t<dev, void>> attributes;
};

template < Device dev >
struct LodDescriptor {
	static constexpr Device DEVICE = dev;
	PolygonsDescriptor<dev> polygon;
	SpheresDescriptor<dev> spheres;
	i32 numPrimitives;
	AccelDescriptor accelStruct;
	// Sort-of linked list, to-be-set not by the LoD itself
	// but rather by the scene upon descriptor creation
	u32 previous = std::numeric_limits<u32>::max();
	u32 next = std::numeric_limits<u32>::max();
};

struct CameraDescriptor {
	u8 cameraParameters[cameras::MAX_CAMERA_PARAM_SIZE];

	CUDA_FUNCTION const cameras::CameraParams& get() const {
		return *as<cameras::CameraParams>(cameraParameters);
	}
	CUDA_FUNCTION cameras::CameraParams& get() {
		return *as<cameras::CameraParams>(cameraParameters);
	}
};

template < Device dev >
struct InstanceData {
	ConstArrayDevHandle_t<dev, ei::Mat3x4> worldToInstance;		// Full inverse transformation Scale⁻¹ * Rotation⁻¹ * Translation⁻¹
	ConstArrayDevHandle_t<dev, u32> lodIndices;

	CUDA_FUNCTION __forceinline__ ei::Mat3x4 compute_instance_to_world_transformation(const i32 instanceIndex) const noexcept {
		return compute_instance_to_world_transformation(this->worldToInstance[instanceIndex]);
	}

	static CUDA_FUNCTION ei::Mat3x4 compute_instance_to_world_transformation(const ei::Mat3x4& matrix) noexcept {
		// Experiments determined this to be the fastest way by a factor of ~2 over naϊvely inverting
		const ei::Mat3x3 rotScale{ matrix };
		// Faster invert of 3x3 than LU decomposition (another speedup of factor ~5)
		const auto m00 = rotScale.m11 * rotScale.m22 - rotScale.m21 * rotScale.m12;
		const auto m01 = rotScale.m21 * rotScale.m02 - rotScale.m01 * rotScale.m22;
		const auto m02 = rotScale.m01 * rotScale.m12 - rotScale.m11 * rotScale.m02;
		const auto m10 = rotScale.m20 * rotScale.m12 - rotScale.m10 * rotScale.m22;
		const auto m11 = rotScale.m00 * rotScale.m22 - rotScale.m20 * rotScale.m02;
		const auto m12 = rotScale.m10 * rotScale.m02 - rotScale.m00 * rotScale.m12;
		const auto m20 = rotScale.m10 * rotScale.m21 - rotScale.m20 * rotScale.m11;
		const auto m21 = rotScale.m20 * rotScale.m01 - rotScale.m00 * rotScale.m21;
		const auto m22 = rotScale.m00 * rotScale.m11 - rotScale.m10 * rotScale.m01;
		const auto invRotScale = (1.f / ei::determinant(rotScale)) * ei::Mat3x3{ m00, m01, m02, m10, m11, m12, m20, m21, m22 };
		const ei::Vec3 translation{ matrix, 0u, 3u };
		const auto invTranslation = -(invRotScale * translation);
		return ei::Mat3x4{
			invRotScale.m00, invRotScale.m01, invRotScale.m02, invTranslation.x,
			invRotScale.m10, invRotScale.m11, invRotScale.m12, invTranslation.y,
			invRotScale.m20, invRotScale.m21, invRotScale.m22, invTranslation.z
		};
	}
};

template <>
struct InstanceData<Device::OPENGL> {
	ConstArrayDevHandle_t<Device::OPENGL, ei::Mat3x4> instanceToWorld;		// Full transformation Translation * Rotation * Scale
	ConstArrayDevHandle_t<Device::OPENGL, ei::Mat3x4> worldToInstance;		// Full inverse transformation Scale⁻¹ * Rotation⁻¹ * Translation⁻¹
	ConstArrayDevHandle_t<Device::CPU, u32> lodIndices;

	const SceneDescriptor<Device::CPU>* cpuDescriptor;
};

// Light, camera etc.
template < Device dev >
struct SceneDescriptor : public InstanceData<dev> {
	static constexpr Device DEVICE = dev;
	CameraDescriptor camera;
	u32 numLods;
	i32 numInstances;
	u32 validInstanceIndex;	// An index of a valid instance which should be used for eg. medium checks
	float diagSize;	// len(aabb.max - aabb.min)
	ei::Box aabb;	// Scene-wide bounding box
	// The receiver of this struct is responsible for deallocating these two arrays!
	ConstArrayDevHandle_t<NotGl<dev>, LodDescriptor<dev>> lods;

	AccelDescriptor accelStruct;
	ConstArrayDevHandle_t<dev, ei::Box> aabbs; // For each object.

	// The receiver of this struct is responsible for deallocating this memory!
	lights::LightTree<dev> lightTree;
	ConstArrayDevHandle_t<dev, materials::Medium> media;
	ConstArrayDevHandle_t<dev, int> materials;	// Offsets + HandlePacks
	ConstArrayDevHandle_t<dev, textures::ConstTextureDevHandle_t<dev>> alphaTextures;

	static constexpr CUDA_FUNCTION bool is_instance_present(const u32 lodIndex) noexcept {
		return lodIndex != std::numeric_limits<u32>::max();
	}

	CUDA_FUNCTION MaterialIndex get_material_index(PrimitiveHandle primitive) const {
		const LodDescriptor<dev>& object = lods[this->lodIndices[primitive.instanceId]];
		const u32 faceCount = object.polygon.numTriangles + object.polygon.numQuads;
		if(static_cast<u32>(primitive.primId) < faceCount)
			return object.polygon.matIndices[primitive.primId];
		else
			return object.spheres.matIndices[primitive.primId];
	}

	CUDA_FUNCTION const materials::MaterialDescriptorBase& get_material(MaterialIndex matIdx) const {
		return *as<materials::MaterialDescriptorBase>(as<char>(materials) + materials[matIdx]);
	}
	CUDA_FUNCTION const materials::MaterialDescriptorBase& get_material(PrimitiveHandle primitive) const {
		return get_material(get_material_index(primitive));
	}


	CUDA_FUNCTION textures::ConstTextureDevHandle_t<dev> get_alpha_texture(MaterialIndex matIdx) const {
		return alphaTextures[matIdx];
	}
	CUDA_FUNCTION textures::ConstTextureDevHandle_t<dev> get_alpha_texture(PrimitiveHandle primitive) const {
		return get_alpha_texture(get_material_index(primitive));
	}

	CUDA_FUNCTION bool has_alpha(MaterialIndex matIdx) const {
		return alphaTextures[matIdx] != textures::ConstTextureDevHandle_t<dev>{};
	}
	CUDA_FUNCTION bool has_alpha(PrimitiveHandle primitive) const {
		return has_alpha(get_material_index(primitive));
	}
};

}} // namespace mufflon::scene