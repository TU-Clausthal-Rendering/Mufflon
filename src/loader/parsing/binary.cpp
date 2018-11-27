#include "binary.hpp"
#include "core/export/interface.h"
#include "util/log.hpp"
#include "util/punning.hpp"
#include <ei/conversions.hpp>
#include <ei/3dtypes.hpp>
#include <ei/vector.hpp>
#include <cstdio>

namespace loader::binary {

using namespace mufflon;

#define FUNCTION_NAME __func__

namespace {

// RAII wrapper around C file descriptor
class FileDescriptor {
public:
	FileDescriptor(fs::path file, const char* mode) :
		m_desc(std::fopen(file.string().c_str(), mode))
	{
		if(m_desc == nullptr)
			throw std::ios_base::failure("Failed to open file '" + file.string()
										 + "' with mode '" + mode + "'");
	}

	~FileDescriptor() {
		this->close();
	}

	void close() {
		if(m_desc != nullptr)
			if(std::fclose(m_desc) != 0)
				throw std::ios_base::failure("Failed to close file descriptor");
	}

	FILE* get() {
		return m_desc;
	}

	// Advances a C file descriptor (possibly in multiple steps)
	FileDescriptor& seek(u64 offset, std::ios_base::seekdir dir = std::ios_base::cur) {
		int origin = 0u;
		switch(dir) {
			case std::ios_base::beg: origin = SEEK_SET; break;
			case std::ios_base::end: origin = SEEK_END; break;
			case std::ios_base::cur:
			default: origin = SEEK_CUR;
		}
		while(offset > std::numeric_limits<long>::max()) {
			if(std::fseek(m_desc, std::numeric_limits<long>::max(), origin) != 0)
				throw std::runtime_error("Failed to seek C file descriptor to desired position");
			offset -= std::numeric_limits<long>::max();
		}
		if(std::fseek(m_desc, static_cast<long>(offset), origin) != 0)
			throw std::runtime_error("Failed to seek C file descriptor to desired position");
		return *this;
	}

private:
	FILE* m_desc;
};

} // namespace


// Read a string as specified by the file format
template <>
std::string BinaryLoader::read<std::string>() {
	std::string str;
	str.resize(read<u32>());
	m_fileStream.read(str.data(), str.length());
	return str;
}
// Read a Vec3 as specified by the file format
template <>
ei::Vec3 BinaryLoader::read<ei::Vec3>() {
	return ei::Vec3{ read<float>(), read<float>(), read<float>() };
}
// Read a UVec3 as specified by the file format
template <>
ei::UVec3 BinaryLoader::read<ei::UVec3>() {
	return ei::UVec3{ read<u32>(), read<u32>(), read<u32>() };
}
// Read a UVec4 as specified by the file format
template <>
ei::UVec4 BinaryLoader::read<ei::UVec4>() {
	return ei::UVec4{ read<u32>(), read<u32>(),
		read<u32>(), read<u32>() };
}
// Read a 3x4 matrix as specified by the file format
template <>
Mat4x3 BinaryLoader::read<Mat4x3>() {
	return Mat4x3{ {
		read<float>(), read<float>(), read<float>(), read<float>(),
		read<float>(), read<float>(), read<float>(), read<float>(),
		read<float>(), read<float>(), read<float>(), read<float>()
	} };
}
// Read a AABB as specified by the file format
template <>
ei::Box BinaryLoader::read<ei::Box>() {
	return ei::Box{ read<ei::Vec3>(), read<ei::Vec3>() };
}

// Clears the loader state
void BinaryLoader::clear_state() {
	if(m_fileStream.is_open())
		m_fileStream.close();
	m_currObjState = ObjectState{};
	m_materialNames.clear();
	m_aabb = ei::Box{};
}

AttribDesc BinaryLoader::map_bin_attrib_type(AttribType type) {
	switch(type) {
		case AttribType::CHAR: return AttribDesc{ AttributeType::ATTR_CHAR, 1u };
		case AttribType::UCHAR: return AttribDesc{ AttributeType::ATTR_UCHAR, 1u };
		case AttribType::SHORT: return AttribDesc{ AttributeType::ATTR_SHORT, 1u };
		case AttribType::USHORT: return AttribDesc{ AttributeType::ATTR_USHORT, 1u };
		case AttribType::INT: return AttribDesc{ AttributeType::ATTR_INT, 1u };
		case AttribType::UINT: return AttribDesc{ AttributeType::ATTR_UINT, 1u };
		case AttribType::LONG: return AttribDesc{ AttributeType::ATTR_LONG, 1u };
		case AttribType::ULONG: return AttribDesc{ AttributeType::ATTR_ULONG, 1u };
		case AttribType::FLOAT: return AttribDesc{ AttributeType::ATTR_FLOAT, 1u };
		case AttribType::DOUBLE: return AttribDesc{ AttributeType::ATTR_DOUBLE, 1u };
		case AttribType::UCHAR2: return AttribDesc{ AttributeType::ATTR_UCHAR, 2u };
		case AttribType::UCHAR3: return AttribDesc{ AttributeType::ATTR_UCHAR, 3u };
		case AttribType::UCHAR4: return AttribDesc{ AttributeType::ATTR_UCHAR, 4u };
		case AttribType::INT2: return AttribDesc{ AttributeType::ATTR_INT, 2u };
		case AttribType::INT3: return AttribDesc{ AttributeType::ATTR_INT, 3u };
		case AttribType::INT4: return AttribDesc{ AttributeType::ATTR_INT, 4u };
		case AttribType::FLOAT2: return AttribDesc{ AttributeType::ATTR_FLOAT, 2u };
		case AttribType::FLOAT3: return AttribDesc{ AttributeType::ATTR_FLOAT, 3u };
		case AttribType::FLOAT4: return AttribDesc{ AttributeType::ATTR_FLOAT, 4u };
		default: return AttribDesc{ AttributeType::ATTR_COUNT, 0u };
	}
}

// Read vertices with applied normal compression, but no deflating
void BinaryLoader::read_normal_compressed_vertices() {
	if(m_currObjState.numVertices == 0)
		return;
	const std::ifstream::off_type currOffset = m_fileStream.tellg() - m_fileStart;
	FileDescriptor points{ m_filePath, "rb" };
	FileDescriptor normals{ m_filePath, "rb" };
	FileDescriptor uvs{ m_filePath, "rb" };

	points.seek(currOffset, std::ios_base::beg);
	normals.seek(currOffset + 3u * sizeof(float) * m_currObjState.numVertices, std::ios_base::beg);
	uvs.seek(currOffset + (2u + 3u) * sizeof(float) * m_currObjState.numVertices, std::ios_base::beg);
	std::size_t pointsRead = 0u;
	std::size_t uvsRead = 0u;
	if(polygon_add_vertex_bulk_aabb_no_normals(m_currObjState.objHdl, m_currObjState.numVertices, points.get(),
												uvs.get(), util::pun<Vec3>(m_currObjState.aabb.min),
												util::pun<Vec3>(m_currObjState.aabb.max), &pointsRead,
												&uvsRead) == INVALID_INDEX)
		throw std::runtime_error("Failed to bulk-read vertices for object '" + m_currObjState.name + "', LoD "
									+ std::to_string(m_currObjState.lodLevel));
	if(pointsRead != m_currObjState.numVertices || uvsRead != m_currObjState.numVertices)
		throw std::runtime_error("Not all vertices were fully read'" + m_currObjState.name
								 + "', LoD " + std::to_string(m_currObjState.lodLevel));
	// Unpack the normals
	m_fileStream.seekg(3u * sizeof(float) * m_currObjState.numVertices, std::ios_base::cur);
	for(u32 n = 0u; n < m_currObjState.numVertices; ++n) {
		const ei::Vec3 normal = ei::normalize(ei::unpackOctahedral32(read<u32>()));
		if(!polygon_set_vertex_normal(m_currObjState.objHdl, static_cast<VertexHdl>(n), util::pun<Vec3>(normal)))
			throw std::runtime_error("Failed to set normal for object '" + m_currObjState.name + "', LoD "
										+ std::to_string(m_currObjState.lodLevel));
	}

	// Seek to the faces
	m_fileStream.seekg((3u + 2u * 2u) * sizeof(float) * m_currObjState.numVertices, std::ios_base::cur);
}

// Read vertices without deflation and without normal compression
void BinaryLoader::read_normal_uncompressed_vertices() {
	if(m_currObjState.numVertices == 0)
		return;
	const std::ifstream::off_type currOffset = m_fileStream.tellg() - m_fileStart;
	FileDescriptor points{ m_filePath, "rb" };
	FileDescriptor normals{ m_filePath, "rb" };
	FileDescriptor uvs{ m_filePath, "rb" };

	points.seek(currOffset, std::ios_base::beg);
	normals.seek(currOffset + 3u * sizeof(float) * m_currObjState.numVertices, std::ios_base::beg);
	uvs.seek(currOffset + 2u * 3u * sizeof(float) * m_currObjState.numVertices, std::ios_base::beg);
	std::size_t pointsRead = 0u;
	std::size_t normalsRead = 0u;
	std::size_t uvsRead = 0u;
	if(polygon_add_vertex_bulk_aabb(m_currObjState.objHdl, m_currObjState.numVertices, points.get(),
									normals.get(), uvs.get(),
									util::pun<Vec3>(m_currObjState.aabb.min),
									util::pun<Vec3>(m_currObjState.aabb.max), &pointsRead,
									&normalsRead, &uvsRead) == INVALID_INDEX)
		throw std::runtime_error("Failed to bulk-read vertices for object '" + m_currObjState.name + "', LoD "
								 + std::to_string(m_currObjState.lodLevel));
	if(pointsRead != m_currObjState.numVertices || normalsRead != m_currObjState.numVertices
	   || uvsRead != m_currObjState.numVertices)
		throw std::runtime_error("Not all vertices were fully read for object '" + m_currObjState.name
								 + "', LoD " + std::to_string(m_currObjState.lodLevel));
	// Seek to the faces
	m_fileStream.seekg((2u * 3u + 2u) * sizeof(float) * m_currObjState.numVertices, std::ios_base::cur);
}

// Reads the attribute parameters, but not the actual data
BinaryLoader::AttribState BinaryLoader::read_uncompressed_attribute() {
	return AttribState{
		read<std::string>(),
		read<std::string>(),
		read<u32>(),
		map_bin_attrib_type(static_cast<AttribType>(read<u32>())),
		read<u64>()
	};
}

// Reads the acctual data into the object
void BinaryLoader::read_uncompressed_vertex_attributes() {
	if(m_currObjState.numVertices == 0 || m_currObjState.numVertAttribs == 0)
		return;
	FileDescriptor attr{ m_filePath, "rb" };
	attr.seek(m_fileStream.tellg() - m_fileStart, std::ios_base::beg);
	for(u32 i = 0u; i < m_currObjState.numVertAttribs; ++i) {
		if(read<u32>() != ATTRIBUTE_MAGIC)
			throw std::runtime_error("Invalid attribute magic constant (object '" + m_currObjState.name + "', LoD "
									 + std::to_string(m_currObjState.lodLevel) + ')');
		AttribState state = read_uncompressed_attribute();
		auto attrHdl = polygon_request_vertex_attribute(m_currObjState.objHdl, state.name.c_str(), state.type);
		if(attrHdl.openMeshIndex == INVALID_INDEX)
			throw std::runtime_error("Failed to add vertex attribute to object '" + m_currObjState.name
									 + "', LoD " + std::to_string(m_currObjState.lodLevel));
		if(polygon_set_vertex_attribute_bulk(m_currObjState.objHdl, &attrHdl, 0u,
											 m_currObjState.numVertices,
											 attr.get()) == INVALID_SIZE)
			throw std::runtime_error("Failed to set vertex attribute data for object '" + m_currObjState.name
									 + "', LoD " + std::to_string(m_currObjState.lodLevel));
		// Seek past the attribute
		m_fileStream.seekg(state.bytes, std::ios_base::cur);
	}
}

// Reads the acctual data into the object
void BinaryLoader::read_uncompressed_face_attributes() {
	if((m_currObjState.numTriangles == 0 && m_currObjState.numQuads == 0) || m_currObjState.numFaceAttribs == 0)
		return;
	FileDescriptor attr{ m_filePath, "rb" };
	attr.seek(m_fileStream.tellg() - m_fileStart, std::ios_base::beg);
	for(u32 i = 0u; i < m_currObjState.numVertAttribs; ++i) {
		if(read<u32>() != ATTRIBUTE_MAGIC)
			throw std::runtime_error("Invalid attribute magic constant (object '" + m_currObjState.name + "', LoD "
									 + std::to_string(m_currObjState.lodLevel) + ')');
		AttribState state = read_uncompressed_attribute();
		auto attrHdl = polygon_request_face_attribute(m_currObjState.objHdl, state.name.c_str(), state.type);
		if(attrHdl.openMeshIndex == INVALID_INDEX)
			throw std::runtime_error("Failed to add face attribute to object '" + m_currObjState.name
									 + "', LoD " + std::to_string(m_currObjState.lodLevel));
		if(polygon_set_face_attribute_bulk(m_currObjState.objHdl, &attrHdl, 0u,
										   m_currObjState.numTriangles + m_currObjState.numQuads,
										   attr.get()) == INVALID_SIZE)
			throw std::runtime_error("Failed to set face attribute data for object '" + m_currObjState.name
									 + "', LoD " + std::to_string(m_currObjState.lodLevel));
		// Seek past the attribute
		m_fileStream.seekg(state.bytes, std::ios_base::cur);
	}
}

// Reads the acctual data into the object
void BinaryLoader::read_uncompressed_sphere_attributes() {
	if(m_currObjState.numSpheres == 0 || m_currObjState.numSphereAttribs == 0)
		return;
	FileDescriptor attr{ m_filePath, "rb" };
	attr.seek(m_fileStream.tellg() - m_fileStart, std::ios_base::beg);
	for(u32 i = 0u; i < m_currObjState.numVertAttribs; ++i) {
		if(read<u32>() != ATTRIBUTE_MAGIC)
			throw std::runtime_error("Invalid attribute magic constant (object '" + m_currObjState.name + "', LoD "
									 + std::to_string(m_currObjState.lodLevel) + ')');
		AttribState state = read_uncompressed_attribute();
		auto attrHdl = spheres_request_attribute(m_currObjState.objHdl, state.name.c_str(), state.type);
		if(attrHdl.index == INVALID_INDEX)
			throw std::runtime_error("Failed to add sphere attribute to object '" + m_currObjState.name
									 + "', LoD " + std::to_string(m_currObjState.lodLevel));
		if(spheres_set_attribute_bulk(m_currObjState.objHdl, &attrHdl, 0u,
									  m_currObjState.numSpheres,
									  attr.get()) == INVALID_SIZE)
			throw std::runtime_error("Failed to set sphere attribute data for object '" + m_currObjState.name
									 + "', LoD " + std::to_string(m_currObjState.lodLevel));
		// Seek past the attribute
		m_fileStream.seekg(state.bytes, std::ios_base::cur);
	}
}

void BinaryLoader::read_uncompressed_face_materials() {
	if(m_currObjState.numTriangles == 0 && m_currObjState.numQuads == 0)
		return;
	FileDescriptor matIdxs{ m_filePath, "rb" };
	matIdxs.seek(m_fileStream.tellg() - m_fileStart, std::ios_base::beg);
	const u32 faces = m_currObjState.numTriangles + m_currObjState.numQuads;
	if(polygon_set_material_idx_bulk(m_currObjState.objHdl, static_cast<FaceHdl>(0u),
									 faces, matIdxs.get()) == INVALID_SIZE)
		throw std::runtime_error("Failed to set face material for object '"
								 + m_currObjState.name + "', LoD "
								 + std::to_string(m_currObjState.lodLevel));
	// Seek to face attributes
	m_fileStream.seekg(sizeof(u16) * faces, std::ios_base::cur);
}

void BinaryLoader::read_uncompressed_sphere_materials() {
	if(m_currObjState.numSpheres == 0)
		return;
	FileDescriptor matIdxs{ m_filePath, "rb" };
	matIdxs.seek(m_fileStream.tellg() - m_fileStart, std::ios_base::beg);
	if(spheres_set_material_idx_bulk(m_currObjState.objHdl, static_cast<SphereHdl>(0u),
									 m_currObjState.numSpheres, matIdxs.get()) == INVALID_SIZE)
		throw std::runtime_error("Failed to set face material for object '"
								 + m_currObjState.name + "', LoD "
								 + std::to_string(m_currObjState.lodLevel));
	// Seek to face attributes
	m_fileStream.seekg(sizeof(u16) * m_currObjState.numSpheres, std::ios_base::cur);
}

void BinaryLoader::read_uncompressed_triangles() {
	if(m_currObjState.numTriangles == 0)
		return;
	// Read the faces (cannot do that bulk-like)
	for(u32 tri = 0u; tri < m_currObjState.numTriangles; ++tri) {
		const ei::UVec3 indices = read<ei::UVec3>();
		if(polygon_add_triangle(m_currObjState.objHdl, util::pun<UVec3>(indices)) == INVALID_INDEX)
			throw std::runtime_error("Failed to add triangle to object '" + m_currObjState.name
									 + "', LoD " + std::to_string(m_currObjState.lodLevel));
	}
	FileDescriptor matIdxs{ m_filePath, "rb" };
	matIdxs.seek(m_fileStream.tellg() - m_fileStart, std::ios_base::beg);
	if(polygon_set_material_idx_bulk(m_currObjState.objHdl, static_cast<FaceHdl>(0u),
									 m_currObjState.numTriangles, matIdxs.get()) == INVALID_SIZE)
		throw std::runtime_error("Failed to set triangle material for object '" + m_currObjState.name + "', LoD "
								 + std::to_string(m_currObjState.lodLevel));
	// Seek to the quads
	m_fileStream.seekg(3u * sizeof(u32) * m_currObjState.numTriangles, std::ios_base::cur);

}

void BinaryLoader::read_uncompressed_quads() {
	if(m_currObjState.numQuads == 0)
		return;
	// Read the faces (cannot do that bulk-like)
	for(u32 quad = 0u; quad < m_currObjState.numQuads; ++quad) {
		const ei::UVec4 indices = read<ei::UVec4>();
		if(polygon_add_quad(m_currObjState.objHdl, util::pun<UVec4>(indices)) == INVALID_INDEX)
			throw std::runtime_error("Failed to add quad to object '" + m_currObjState.name
									 + "', LoD " + std::to_string(m_currObjState.lodLevel));
	}
	FileDescriptor matIdxs{ m_filePath, "rb" };
	matIdxs.seek(m_fileStream.tellg() - m_fileStart, std::ios_base::beg);
	if(polygon_set_material_idx_bulk(m_currObjState.objHdl, static_cast<FaceHdl>(0u),
									 m_currObjState.numQuads, matIdxs.get()) == INVALID_SIZE)
		throw std::runtime_error("Failed to set quad material for object '" + m_currObjState.name + "', LoD "
								 + std::to_string(m_currObjState.lodLevel));
	// Seek to the material indices
	m_fileStream.seekg(4u * sizeof(u32) * m_currObjState.numQuads, std::ios_base::cur);
}

void BinaryLoader::read_uncompressed_spheres() {
	if(m_currObjState.numSpheres == 0)
		return;
	FileDescriptor spheres{ m_filePath, "rb" };
	spheres.seek(m_fileStream.tellg() - m_fileStart, std::ios_base::beg);
	std::size_t spheresRead = 0u;
	if(spheres_add_sphere_bulk(m_currObjState.objHdl, m_currObjState.numSpheres,
							   spheres.get(), &spheresRead) == INVALID_INDEX)
		throw std::runtime_error("Failed to add quad to object '" + m_currObjState.name
									 + "', LoD " + std::to_string(m_currObjState.lodLevel));
	if(spheresRead != m_currObjState.numSpheres)
		throw std::runtime_error("Not all spheres were fully read for object '" + m_currObjState.name
								 + "', LoD " + std::to_string(m_currObjState.lodLevel));
	// Seek to the material indices
	m_fileStream.seekg(4u * sizeof(float) * m_currObjState.numSpheres, std::ios_base::cur);

	read_uncompressed_sphere_materials();
	read_uncompressed_sphere_attributes();
}

void BinaryLoader::read_lod() {
	if(m_currObjState.flags.is_set(ObjectFlag::DEFLATE)) {
			// TODO
		throw std::runtime_error("Compressed data is not supported yet");
	} else {
		// First comes vertex data
		if(m_currObjState.flags.is_set(ObjectFlag::COMPRESSED_NORMALS))
			read_normal_compressed_vertices();
		else
			read_normal_uncompressed_vertices();
		read_uncompressed_vertex_attributes();
		read_uncompressed_triangles();
		read_uncompressed_quads();
		read_uncompressed_face_materials();
		read_uncompressed_face_attributes();
		read_uncompressed_spheres();

		logInfo("[", FUNCTION_NAME, "] Loaded LoD ", m_currObjState.lodLevel, " of object object'",
				m_currObjState.name, "' with ", m_currObjState.numVertices, " vertices, ",
				m_currObjState.numTriangles, " triangles, ", m_currObjState.numQuads, " quads, ",
				m_currObjState.numSpheres, " spheres");
	}
}

void BinaryLoader::read_object(const u64 globalLod,
							   const std::unordered_map<std::string_view, u64>& localLods) {
	m_currObjState.name = read<std::string>();
	m_currObjState.keyframe = read<u32>();
	m_currObjState.animObjId = read<u32>();
	m_currObjState.aabb = read<ei::Box>();

	// Read the specified LoD
	m_currObjState.lodLevel = [&localLods, globalLod](const std::string& name) {
		const auto iter = localLods.find(std::string_view(name));
		if(iter != localLods.cend())
			return iter->second;
		else
			return globalLod;
	}(m_currObjState.name);
	// Jump table
	const u32 numLods = read<u32>();
	if(m_currObjState.lodLevel >= numLods)
		throw std::runtime_error("Not enough LoDs specified for object '" + m_currObjState.name
								 + "' (found " + std::to_string(numLods) + ", need "
								 + std::to_string(m_currObjState.lodLevel) + ')');
	// Jump to proper LoD
	m_fileStream.seekg(m_currObjState.lodLevel * sizeof(u64), std::ifstream::cur);
	m_fileStream.seekg(read<u64>(), std::ifstream::beg);

	// Load the LoD data
	if(read<u32>() != LOD_MAGIC)
		throw std::runtime_error("Invalid LoD magic constant (object '" + m_currObjState.name + "', LoD "
								 + std::to_string(m_currObjState.lodLevel) + ')');
	m_currObjState.numTriangles = read<u32>();
	m_currObjState.numQuads = read<u32>();
	m_currObjState.numSpheres = read<u32>();
	m_currObjState.numVertices = read<u32>();
	m_currObjState.numEdges = read<u32>();
	m_currObjState.numVertAttribs = read<u32>();
	m_currObjState.numFaceAttribs = read<u32>();
	m_currObjState.numSphereAttribs = read<u32>();

	// Create the object for direct writing
	m_currObjState.objHdl = world_create_object(m_currObjState.name.c_str());
	m_objectHandles.push_back(m_currObjState.objHdl);
	if(m_currObjState.objHdl == nullptr)
		throw std::runtime_error("Failed to create object '" + m_currObjState.name + "', LoD "
								 + std::to_string(m_currObjState.lodLevel));

	read_lod();
}

void BinaryLoader::read_instances() {
	std::vector<uint8_t> hasInstance(m_objectHandles.size(), false);
	const u32 numInstances = read<u32>();
	for(u32 i = 0u; i < numInstances; ++i) {
		const u32 objId = read<u32>();
		const u32 keyframe = read<u32>();
		const u32 animInstId = read<u32>();
		const Mat4x3 transMat = read<Mat4x3>();
		InstanceHdl instHdl = world_create_instance(m_objectHandles[objId]);
		if(instHdl == nullptr)
			throw std::runtime_error("Failed to create instance for object ID "
									 + std::to_string(objId));
		if(!instance_set_transformation_matrix(instHdl, &transMat))
			throw std::runtime_error("Failed to set transformation matrix for instance of object ID "
									 + std::to_string(objId));

		Vec3 min;
		Vec3 max;
		if(!instance_get_bounding_box(instHdl, &min, &max))
			throw std::runtime_error("Failed to get bounding box for instance of object ID "
									 + std::to_string(objId));

		m_aabb = ei::Box(m_aabb, ei::Box(util::pun<ei::Vec3>(min), util::pun<ei::Vec3>(max)));
	}

	for(std::size_t i = 0u; i < hasInstance.size(); ++i) {
		if(!hasInstance[i]) {
			// Add default instance
			const Mat4x3 transMat = util::pun<Mat4x3>(ei::Matrix<float, 4, 3>{
				1.f, 0.f, 0.f,
				0.f, 1.f, 0.f,
				0.f, 0.f, 1.f,
				0.f, 0.f, 0.f
			});
			InstanceHdl instHdl = world_create_instance(m_objectHandles[i]);
			if(instHdl == nullptr)
				throw std::runtime_error("Failed to create instance for object ID "
										 + std::to_string(i));
			if(!instance_set_transformation_matrix(instHdl, &transMat))
				throw std::runtime_error("Failed to set transformation matrix for instance of object ID "
										 + std::to_string(i));
		}
	}

	// Create identity instances for objects not having one yet
}

void BinaryLoader::load_file(const u64 globalLod,
							 const std::unordered_map<std::string_view, u64>& localLods) {
	try {
		if(!fs::exists(m_filePath))
			throw std::runtime_error("Binary file '" + m_filePath.string() + "' does not exist");

		// Open the binary file and enable exception management
		m_fileStream = std::ifstream(m_filePath, std::ios_base::binary);
		if(m_fileStream.bad() || m_fileStream.fail())
			throw std::runtime_error("Failed to open binary file '" + m_filePath.string() + '\'');
		m_fileStream.exceptions(std::ifstream::failbit);
		// Needed to get a C file descriptor offset
		const std::ifstream::pos_type fileStart = m_fileStream.tellg();
		logInfo("[", FUNCTION_NAME, "] Reading binary file '", m_filePath.string(), "'");

		// Read the materials header
		if(read<u32>() != MATERIALS_HEADER_MAGIC)
			throw std::runtime_error("Invalid materials header magic constant");
		const u64 objectStart = read<u64>();
		const u32 numMaterials = read<u32>();
		// Read the material names (and implicitly their indices)
		m_materialNames.reserve(numMaterials);
		for(u32 i = 0u; i < numMaterials; ++i) {
			std::string& name = m_materialNames.emplace_back();
			u32 nameLength = read<u32>();
			name.resize(nameLength + 6u); // Extra 6 chars for [mat:...] wrapping
			for(u32 i = 0u; i < nameLength; ++i)
				name[i + 5u] = read<char>();
		}

		// Jump to the location of objects
		m_fileStream.seekg(objectStart, std::ifstream::beg);
		// Parse the object header
		if(read<u32>() != OBJECTS_HEADER_MAGIC)
			throw std::runtime_error("Invalid objects header magic constant");
		const u64 instanceStart = read<u64>();

		// Parse the object jumptable
		std::vector<u64> objJumpTable(read<u32>());
		for(std::size_t i = 0u; i < objJumpTable.size(); ++i)
			objJumpTable[i] = read<u64>();

		// Next come the objects
		for(std::size_t i = 0u; i < objJumpTable.size(); ++i) {
			// Jump to the right position in file
			m_fileStream.seekg(objJumpTable[i], std::ifstream::beg);
			// Read the object flags
			m_currObjState.flags = ObjectFlag{ read<u32>() };

			// Read object
			if(read<u32>() != OBJECT_MAGIC)
				throw std::runtime_error("Invalid object magic constant (object " + std::to_string(i) + ')');
			read_object(globalLod, localLods);
		}

		// Now come instances
		m_fileStream.seekg(instanceStart, std::ios_base::beg);
		if(read<u32>() != INSTANCE_MAGIC)
			throw std::runtime_error("Invalid instance magic constant");
		read_instances();

		this->clear_state();
	} catch(const std::exception&) {
		// Clean up before leaving throwing
		this->clear_state();
		throw;
	}
}

} // namespace loader::binary