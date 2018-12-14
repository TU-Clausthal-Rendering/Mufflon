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
struct ObjectDescriptor {
	PolygonsDescriptor<dev> polygon;
	SpheresDescriptor<dev> spheres;
	AccelDescriptor accelStruct;
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

// Light, camera etc.
template < Device dev >
struct SceneDescriptor {
	CameraDescriptor camera;
	u32 numObjects;
	u32 numInstances;
	ei::Box aabb;	// Scene-wide bounding box
	// The receiver of this struct is responsible for deallocating these two arrays!
	ArrayDevHandle_t<dev, ObjectDescriptor<dev>> objects;

	AccelDescriptor accelStruct;
	ArrayDevHandle_t<dev, ei::Mat3x4> transformations;
	ArrayDevHandle_t<dev, u32> objectIndices;
	ArrayDevHandle_t<dev, ei::Box> aabbs; // For each object.

	// The receiver of this struct is responsible for deallocating this memory!
	const lights::LightTree<dev> lightTree;
	ConstArrayDevHandle_t<dev, materials::Medium> media;
	ConstArrayDevHandle_t<dev, int> materials;	// Offsets + HandlePacks

	CUDA_FUNCTION const materials::HandlePack& get_material(MaterialIndex matIdx) const {
		return *as<materials::HandlePack>(as<char>(materials) + materials[matIdx]);
	}
};

}} // namespace mufflon::scene