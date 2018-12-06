#pragma once

#include "util/int_types.hpp"
#include "lights/light_tree.hpp"
#include "core/memory/residency.hpp"
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

// Geometric descriptors
template < Device dev >
struct PolymeshDescriptor {
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
	ConstArrayDevHandle_t<dev, ConstArrayDevHandle_t<dev, void>> vertexAttributes;
	ConstArrayDevHandle_t<dev, ConstArrayDevHandle_t<dev, void>> faceAttributes;
};

template < Device dev >
struct SpheresDescriptor {
	u32 numSpheres;
	u32 numAttributes;
	ConstArrayDevHandle_t<dev, ei::Vec4> radiiPositions;
	ConstArrayDevHandle_t<dev, u16> matIndices;
	ConstArrayDevHandle_t<dev, ConstArrayDevHandle_t<dev, void>> attributes;
};

template < Device dev >
struct ObjectDescriptor {
	ei::Box aabb;
	PolymeshDescriptor<dev> polygon;
	SpheresDescriptor<dev> spheres;
	ArrayDevHandle_t<dev, void> bvhData;
};

template < Device dev >
struct InstanceDescriptor {
	ei::Matrix<Real, 4, 3> transformation;
	// TODO: pointer or index?
	ArrayDevHandle_t<dev, ObjectDescriptor<dev>> object;
};

// Light, camera etc.
template < Device dev >
struct SceneDescriptor {
	u32 numInstances;
	u32 numObjects;
	ei::Box aabb;	// Scene-wide bounding box
	// TODO: objects etc

	lights::LightTree<dev>* lightTree;
	// TODO: materials, cameras
};

}} // namespace mufflon::scene