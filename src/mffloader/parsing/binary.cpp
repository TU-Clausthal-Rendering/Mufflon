#include "binary.hpp"
#include "core/export/interface.h"
#include "util/log.hpp"
#include "util/punning.hpp"
#include "profiler/cpu_profiler.hpp"
#include <ei/conversions.hpp>
#include <ei/3dtypes.hpp>
#include <ei/vector.hpp>
#include <miniz/miniz.h>
#include <cstdio>

namespace mff_loader::binary {

using namespace mufflon;

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
		if(m_desc != nullptr) {
			// Nothing we can do if this fails - not like we care, either
			(void) std::fclose(m_desc);
			m_desc = nullptr;
		}
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
	FILE* m_desc = nullptr;
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
Mat3x4 BinaryLoader::read<Mat3x4>() {
	return Mat3x4{ {
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

template <>
std::string BinaryLoader::read<std::string>(const unsigned char*& data) {
	std::string str;
	u32 size = read<u32>(data);
	str.resize(size);
	for(u32 i = 0u; i < size; ++i)
		str[i] = read<char>(data);
	return str;
}

// Clears the loader state
void BinaryLoader::clear_state() {
	if(m_fileStream.is_open())
		m_fileStream.close();
	m_objects.clear();
	m_objJumpTable.clear();
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
void BinaryLoader::read_normal_compressed_vertices(const ObjectState& object, const LodState& lod) {
	if(lod.numVertices == 0)
		return;
	const std::ifstream::off_type currOffset = m_fileStream.tellg() - m_fileStart;
	FileDescriptor points{ m_filePath, "rb" };
	FileDescriptor normals{ m_filePath, "rb" };
	FileDescriptor uvs{ m_filePath, "rb" };

	points.seek(currOffset, std::ios_base::beg);
	normals.seek(currOffset + 3u * sizeof(float) * lod.numVertices, std::ios_base::beg);
	uvs.seek(currOffset + (3u * sizeof(float) + sizeof(u32)) * lod.numVertices, std::ios_base::beg);
	std::size_t pointsRead = 0u;
	std::size_t uvsRead = 0u;

	BulkLoader pointsBulk{ BulkLoader::BULK_FILE, { points.get() } };
	BulkLoader uvsBulk{ BulkLoader::BULK_FILE, { uvs.get() } };
	AABB aabb{
		util::pun<Vec3>(object.aabb.min),
		util::pun<Vec3>(object.aabb.max)
	};
	if(polygon_add_vertex_bulk(lod.lodHdl, lod.numVertices, &pointsBulk, nullptr, &uvsBulk,
							   &aabb, &pointsRead, nullptr, &uvsRead) == INVALID_INDEX)
		throw std::runtime_error("Failed to bulk-read vertices for object '" + object.name + "'");
	logPedantic("[BinaryLoader::read_normal_compressed_vertices] Object '", object.name, "': Read ",
				pointsRead, "'", uvsRead, " point/UVs bulk");
	if(pointsRead != lod.numVertices || uvsRead != lod.numVertices)
		throw std::runtime_error("Not all vertices were fully read for object '" + object.name + "'");
	// Unpack the normals
	m_fileStream.seekg(3u * sizeof(float) * lod.numVertices, std::ios_base::cur);
	for(u32 n = 0u; n < lod.numVertices; ++n) {
		const ei::Vec3 normal = ei::normalize(ei::unpackOctahedral32(read<u32>()));
		if(!polygon_set_vertex_normal(lod.lodHdl, static_cast<VertexHdl>(n), util::pun<Vec3>(normal)))
			throw std::runtime_error("Failed to set normal for object '" + object.name + "'");
	}
	// Skip past the UV coordinates again
	m_fileStream.seekg(2u * sizeof(float) * lod.numVertices, std::ios_base::cur);
}

// Read vertices without deflation and without normal compression
void BinaryLoader::read_normal_uncompressed_vertices(const ObjectState& object, const LodState& lod) {
	auto scope = Profiler::instance().start<CpuProfileState>("BinaryLoader::read_normal_uncompressed_vertices");
	if(lod.numVertices == 0)
		return;
	const std::ifstream::off_type currOffset = m_fileStream.tellg() - m_fileStart;
	FileDescriptor points{ m_filePath, "rb" };
	FileDescriptor normals{ m_filePath, "rb" };
	FileDescriptor uvs{ m_filePath, "rb" };

	points.seek(currOffset, std::ios_base::beg);
	normals.seek(currOffset + 3u * sizeof(float) * lod.numVertices, std::ios_base::beg);
	uvs.seek(currOffset + 2u * 3u * sizeof(float) * lod.numVertices, std::ios_base::beg);
	std::size_t pointsRead = 0u;
	std::size_t normalsRead = 0u;
	std::size_t uvsRead = 0u;
	BulkLoader pointsBulk{ BulkLoader::BULK_FILE, { points.get() } };
	BulkLoader normalsBulk{ BulkLoader::BULK_FILE, { normals.get() } };
	BulkLoader uvsBulk{ BulkLoader::BULK_FILE, { uvs.get() } };
	AABB aabb{
		util::pun<Vec3>(object.aabb.min),
		util::pun<Vec3>(object.aabb.max)
	};
	if(polygon_add_vertex_bulk(lod.lodHdl, lod.numVertices, &pointsBulk, &normalsBulk, &uvsBulk,
							   &aabb, &pointsRead, &normalsRead, &uvsRead) == INVALID_INDEX)
		throw std::runtime_error("Failed to bulk-read vertices for object '" + object.name + "'");
	logPedantic("[BinaryLoader::read_normal_uncompressed_vertices] Object '", object.name, "': Read ",
				pointsRead, "'", normalsRead, "'", uvsRead, " point/normals/UVs bulk");
	if(pointsRead != lod.numVertices || normalsRead != lod.numVertices
	   || uvsRead != lod.numVertices)
		throw std::runtime_error("Not all vertices were fully read for object '" + object.name + "'");
	// Seek to the attributes
	m_fileStream.seekg((2u * 3u + 2u) * sizeof(float) * lod.numVertices, std::ios_base::cur);
}

// Reads the attribute parameters, but not the actual data
BinaryLoader::AttribState BinaryLoader::read_uncompressed_attribute(const ObjectState& object, const LodState& lod) {
	return AttribState{
		read<std::string>(),
		read<std::string>(),
		read<u32>(),
		map_bin_attrib_type(static_cast<AttribType>(read<u32>())),
		read<u64>()
	};
}
BinaryLoader::AttribState BinaryLoader::read_compressed_attribute(const unsigned char*& data) {
	return AttribState{
		read<std::string>(data),
		read<std::string>(data),
		read<u32>(data),
		map_bin_attrib_type(static_cast<AttribType>(read<u32>(data))),
		read<u64>(data)
	};
}

// Reads the acctual data into the object
void BinaryLoader::read_uncompressed_vertex_attributes(const ObjectState& object, const LodState& lod) {
	if(lod.numVertices == 0 || lod.numVertAttribs == 0)
		return;
	FileDescriptor attr{ m_filePath, "rb" };
	attr.seek(m_fileStream.tellg() - m_fileStart, std::ios_base::beg);
	BulkLoader attrBulk{ BulkLoader::BULK_FILE, { attr.get() } };

	if(read<u32>() != ATTRIBUTE_MAGIC)
		throw std::runtime_error("Invalid attribute magic constant (object '" + object.name + "'");
	for(u32 i = 0u; i < lod.numVertAttribs; ++i) {
		AttribState state = read_uncompressed_attribute(object, lod);
		auto attrHdl = polygon_request_vertex_attribute(lod.lodHdl, state.name.c_str(), state.type);
		if(attrHdl.index == INVALID_INDEX)
			throw std::runtime_error("Failed to add vertex attribute to object '" + object.name + "'");
		if(polygon_set_vertex_attribute_bulk(lod.lodHdl, &attrHdl, 0u,
											 lod.numVertices,
											 &attrBulk) == INVALID_SIZE)
			throw std::runtime_error("Failed to set vertex attribute data for object '" + object.name + "'");
		// Seek past the attribute
		m_fileStream.seekg(state.bytes, std::ios_base::cur);
	}
	logPedantic("[BinaryLoader::read_uncompressed_vertex_attributes] Object '", object.name, "': Read ",
				lod.numVertAttribs, " vertex attributes");
}

// Reads the acctual data into the object
void BinaryLoader::read_uncompressed_face_attributes(const ObjectState& object, const LodState& lod) {
	if((lod.numTriangles == 0 && lod.numQuads == 0) || lod.numFaceAttribs == 0)
		return;
	FileDescriptor attr{ m_filePath, "rb" };
	attr.seek(m_fileStream.tellg() - m_fileStart, std::ios_base::beg);
	BulkLoader attrBulk{ BulkLoader::BULK_FILE, { attr.get() } };

	if(read<u32>() != ATTRIBUTE_MAGIC)
		throw std::runtime_error("Invalid attribute magic constant (object '" + object.name + "'");
	for(u32 i = 0u; i < lod.numVertAttribs; ++i) {
		AttribState state = read_uncompressed_attribute(object, lod);
		auto attrHdl = polygon_request_face_attribute(lod.lodHdl, state.name.c_str(), state.type);
		if(attrHdl.index == INVALID_INDEX)
			throw std::runtime_error("Failed to add face attribute to object '" + object.name + "'");
		if(polygon_set_face_attribute_bulk(lod.lodHdl, &attrHdl, 0u,
										   lod.numTriangles + lod.numQuads,
										   &attrBulk) == INVALID_SIZE)
			throw std::runtime_error("Failed to set face attribute data for object '" + object.name + "'");
		// Seek past the attribute
		m_fileStream.seekg(state.bytes, std::ios_base::cur);
	}
	logPedantic("[BinaryLoader::read_uncompressed_face_attributes] Object '", object.name, "': Read ",
				lod.numFaceAttribs, " face attributes");
}

// Reads the acctual data into the object
void BinaryLoader::read_uncompressed_sphere_attributes(const ObjectState& object, const LodState& lod) {
	if(lod.numSpheres == 0 || lod.numSphereAttribs == 0)
		return;
	FileDescriptor attr{ m_filePath, "rb" };
	attr.seek(m_fileStream.tellg() - m_fileStart, std::ios_base::beg);
	BulkLoader attrBulk{ BulkLoader::BULK_FILE, { attr.get() } };

	if(read<u32>() != ATTRIBUTE_MAGIC)
		throw std::runtime_error("Invalid attribute magic constant (object '" + object.name + "'");
	for(u32 i = 0u; i < lod.numVertAttribs; ++i) {
		AttribState state = read_uncompressed_attribute(object, lod);
		auto attrHdl = spheres_request_attribute(lod.lodHdl, state.name.c_str(), state.type);
		if(attrHdl.index == INVALID_INDEX)
			throw std::runtime_error("Failed to add sphere attribute to object '" + object.name + "'");
		if(spheres_set_attribute_bulk(lod.lodHdl, &attrHdl, 0u,
									  lod.numSpheres,
									  &attrBulk) == INVALID_SIZE)
			throw std::runtime_error("Failed to set sphere attribute data for object '" + object.name + "'");
		// Seek past the attribute
		m_fileStream.seekg(state.bytes, std::ios_base::cur);
	}
	logPedantic("[BinaryLoader::read_uncompressed_sphere_attributes] Object '", object.name, "': Read ",
				lod.numSphereAttribs, " sphere attributes");
}

void BinaryLoader::read_uncompressed_face_materials(const ObjectState& object, const LodState& lod) {
	if(lod.numTriangles == 0 && lod.numQuads == 0)
		return;
	FileDescriptor matIdxs{ m_filePath, "rb" };
	matIdxs.seek(m_fileStream.tellg() - m_fileStart, std::ios_base::beg);
	BulkLoader matsBulk{ BulkLoader::BULK_FILE, { matIdxs.get() } };

	const u32 faces = lod.numTriangles + lod.numQuads;
	if(polygon_set_material_idx_bulk(lod.lodHdl, static_cast<FaceHdl>(0u),
									 faces, &matsBulk) == INVALID_SIZE)
		throw std::runtime_error("Failed to set face material for object '"
								 + object.name + "'");
	logPedantic("[BinaryLoader::read_uncompressed_face_materials] Object '", object.name, "': Read ",
				faces, " face material indices");
	// Seek to face attributes
	m_fileStream.seekg(sizeof(u16) * faces, std::ios_base::cur);
}

void BinaryLoader::read_uncompressed_sphere_materials(const ObjectState& object, const LodState& lod) {
	if(lod.numSpheres == 0)
		return;
	FileDescriptor matIdxs{ m_filePath, "rb" };
	matIdxs.seek(m_fileStream.tellg() - m_fileStart, std::ios_base::beg);
	BulkLoader matsBulk{ BulkLoader::BULK_FILE, { matIdxs.get() } };

	if(spheres_set_material_idx_bulk(lod.lodHdl, static_cast<SphereHdl>(0u),
									 lod.numSpheres, &matsBulk) == INVALID_SIZE)
		throw std::runtime_error("Failed to set face material for object '"
								 + object.name + "'");
	logPedantic("[BinaryLoader::read_uncompressed_face_materials] Object '", object.name, "': Read ",
				lod.numSpheres, " sphere material indices");
	// Seek to face attributes
	m_fileStream.seekg(sizeof(u16) * lod.numSpheres, std::ios_base::cur);
}

void BinaryLoader::read_uncompressed_triangles(const ObjectState& object, const LodState& lod) {
	auto scope = Profiler::instance().start<CpuProfileState>("BinaryLoader::read_uncompressed_triangles");
	if(lod.numTriangles == 0)
		return;
	// Read the faces (cannot do that bulk-like)
	for(u32 tri = 0u; tri < lod.numTriangles; ++tri) {
		const ei::UVec3 indices = read<ei::UVec3>();
		if(polygon_add_triangle(lod.lodHdl, util::pun<UVec3>(indices)) == INVALID_INDEX)
			throw std::runtime_error("Failed to add triangle to object '" + object.name + "'");
	}
	logPedantic("[BinaryLoader::read_uncompressed_triangles] Object '", object.name, "': Read ",
				lod.numTriangles, " triangles");
}

void BinaryLoader::read_uncompressed_quads(const ObjectState& object, const LodState& lod) {
	auto scope = Profiler::instance().start<CpuProfileState>("BinaryLoader::read_uncompressed_quads");
	if(lod.numQuads == 0)
		return;
	// Read the faces (cannot do that bulk-like)
	for(u32 quad = 0u; quad < lod.numQuads; ++quad) {
		const ei::UVec4 indices = read<ei::UVec4>();
		if(polygon_add_quad(lod.lodHdl, util::pun<UVec4>(indices)) == INVALID_INDEX)
			throw std::runtime_error("Failed to add quad to object '" + object.name + "'");
	}
	logPedantic("[BinaryLoader::read_uncompressed_quads] Object '", object.name, "': Read ",
				lod.numQuads, " quads");
}

void BinaryLoader::read_uncompressed_spheres(const ObjectState& object, const LodState& lod) {
	if(lod.numSpheres == 0)
		return;
	FileDescriptor spheres{ m_filePath, "rb" };
	spheres.seek(m_fileStream.tellg() - m_fileStart, std::ios_base::beg);
	BulkLoader spheresBulk{ BulkLoader::BULK_FILE, { spheres.get() } };
	AABB aabb{
		util::pun<Vec3>(object.aabb.min),
		util::pun<Vec3>(object.aabb.max)
	};

	std::size_t spheresRead = 0u;
	if(spheres_add_sphere_bulk(lod.lodHdl, lod.numSpheres,
							   &spheresBulk, &aabb, &spheresRead) == INVALID_INDEX)
		throw std::runtime_error("Failed to add spheres to object '" + object.name + "'");
	logPedantic("[BinaryLoader::read_uncompressed_spheres] Object '", object.name, "': Read ",
				spheresRead, " spheres bulk");
	if(spheresRead != lod.numSpheres)
		throw std::runtime_error("Not all spheres were fully read for object '" + object.name + "'");
	// Seek to the material indices
	m_fileStream.seekg(4u * sizeof(float) * lod.numSpheres, std::ios_base::cur);

	read_uncompressed_sphere_materials(object, lod);
	read_uncompressed_sphere_attributes(object, lod);
}

std::vector<unsigned char> BinaryLoader::decompress() {
	u32 compressedBytes = read<u32>();
	u32 decompressedBytes = read<u32>();
	std::vector<unsigned char> in(compressedBytes);
	std::vector<unsigned char> out(decompressedBytes);
	// First read the bytes from the stream
	m_fileStream.read(reinterpret_cast<char*>(in.data()), compressedBytes);
	unsigned long outBytes = decompressedBytes;
	int success = uncompress(out.data(), &outBytes,
							 in.data(), static_cast<unsigned long>(in.size()));
	if(success != MZ_OK)
		throw std::runtime_error("Failed to deflate stream (error code " + std::to_string(success) + ")");
	if(outBytes != static_cast<unsigned long>(decompressedBytes))
		throw std::runtime_error("Mismatch between expected and actual decompressed byte count");
	out.resize(outBytes);
	return out;
}

void BinaryLoader::read_compressed_normal_compressed_vertices(const ObjectState& object, const LodState& lod) {
	if(lod.numVertices == 0)
		return;

	std::vector<unsigned char> vertexData = decompress();
	const Vec3* points = reinterpret_cast<const Vec3*>(vertexData.data());
	const u32* packedNormals = reinterpret_cast<const u32*>(vertexData.data() + lod.numVertices
													  * 3u * sizeof(float));
	const Vec2* uvs = reinterpret_cast<const Vec2*>(vertexData.data() + lod.numVertices
													* (3u * sizeof(float) + sizeof(u32)));

	BulkLoader pointsBulk{ BulkLoader::BULK_ARRAY };
	pointsBulk.descriptor.bytes = reinterpret_cast<const char*>(points);
	BulkLoader uvsBulk{ BulkLoader::BULK_ARRAY };
	uvsBulk.descriptor.bytes = reinterpret_cast<const char*>(uvs);
	std::size_t pointsRead = 0u;
	std::size_t uvsRead = 0u;
	AABB aabb{
		util::pun<Vec3>(object.aabb.min),
		util::pun<Vec3>(object.aabb.max)
	};
	if(polygon_add_vertex_bulk(lod.lodHdl, lod.numVertices, &pointsBulk, nullptr, &uvsBulk,
							   &aabb, &pointsRead, nullptr, &uvsRead) == INVALID_INDEX)
		throw std::runtime_error("Failed to bulk-read vertices for object '" + object.name + "'");
	if(pointsRead != lod.numVertices || uvsRead != lod.numVertices)
		throw std::runtime_error("Not all vertices were fully read for object '" + object.name + "'");

	for(u32 vertex = 0u; vertex < lod.numVertices; ++vertex) {
		Vec3 unpackedNormal = util::pun<Vec3>(ei::normalize(ei::unpackOctahedral32(packedNormals[vertex])));
		if(!polygon_set_vertex_normal(lod.lodHdl, vertex, unpackedNormal))
			throw std::runtime_error("Failed to set vertex normal for vertex " + std::to_string(vertex)
									 + " to object '" + object.name + "'");
	}
	logPedantic("[BinaryLoader::read_compressed_normal_compressed_vertices] Object '", object.name, "': Read ",
				lod.numVertices, " deflated and normal-compressed vertices");
}

// Read vertices without deflation and without normal compression
void BinaryLoader::read_compressed_normal_uncompressed_vertices(const ObjectState& object, const LodState& lod) {
	if(lod.numVertices == 0)
		return;

	std::vector<unsigned char> vertexData = decompress();
	const Vec3* points = reinterpret_cast<const Vec3*>(vertexData.data());
	const Vec3* normals = reinterpret_cast<const Vec3*>(vertexData.data() + lod.numVertices * 3u * sizeof(float));
	const Vec2* uvs = reinterpret_cast<const Vec2*>(vertexData.data() + lod.numVertices * (3u + 3u) * sizeof(float));

	std::size_t pointsRead = 0u;
	std::size_t normalsRead = 0u;
	std::size_t uvsRead = 0u;
	BulkLoader pointsBulk{ BulkLoader::BULK_ARRAY };
	BulkLoader normalsBulk{ BulkLoader::BULK_ARRAY };
	BulkLoader uvsBulk{ BulkLoader::BULK_ARRAY };
	pointsBulk.descriptor.bytes = reinterpret_cast<const char*>(points);
	normalsBulk.descriptor.bytes = reinterpret_cast<const char*>(normals);
	uvsBulk.descriptor.bytes = reinterpret_cast<const char*>(uvs);
	AABB aabb{
		util::pun<Vec3>(object.aabb.min),
		util::pun<Vec3>(object.aabb.max)
	};
	if(polygon_add_vertex_bulk(lod.lodHdl, lod.numVertices, &pointsBulk, &normalsBulk, &uvsBulk,
							   &aabb, &pointsRead, &normalsRead, &uvsRead) == INVALID_INDEX)
		throw std::runtime_error("Failed to bulk-read vertices for object '" + object.name + "'");
	logPedantic("[BinaryLoader::read_compressed_normal_uncompressed_vertices] Object '", object.name, "': Read ",
				pointsRead, "'", normalsRead, "'", uvsRead, " point/normals/UVs bulk deflated");
}

void BinaryLoader::read_compressed_triangles(const ObjectState& object, const LodState& lod) {
	if(lod.numTriangles == 0)
		return;

	std::vector<unsigned char> triangleData = decompress();
	const UVec3* indices = reinterpret_cast<const UVec3*>(triangleData.data());
	// Read the faces (cannot do that bulk-like)
	for(u32 tri = 0u; tri < lod.numTriangles; ++tri) {
		if(polygon_add_triangle(lod.lodHdl, indices[tri]) == INVALID_INDEX)
			throw std::runtime_error("Failed to add triangle to object '" + object.name + "'");
	}
	logPedantic("[BinaryLoader::read_compressed_triangles] Object '", object.name, "': Read ",
				lod.numTriangles, " deflated triangles");
}

void BinaryLoader::read_compressed_quads(const ObjectState& object, const LodState& lod) {
	if(lod.numQuads == 0)
		return;

	std::vector<unsigned char> quadData = decompress();
	const UVec4* indices = reinterpret_cast<const UVec4*>(quadData.data());
	// Read the faces (cannot do that bulk-like)
	for(u32 quad = 0u; quad < lod.numQuads; ++quad) {
		if(polygon_add_quad(lod.lodHdl, indices[quad]) == INVALID_INDEX)
			throw std::runtime_error("Failed to add quad to object '" + object.name + "'");
	}
	logPedantic("[BinaryLoader::read_compressed_quads] Object '", object.name, "': Read ",
				lod.numQuads, " deflated quads");
}

void BinaryLoader::read_compressed_spheres(const ObjectState& object, const LodState& lod) {
	if(lod.numSpheres == 0)
		return;

	std::vector<unsigned char> sphereData = decompress();
	// Attributes and material indices are compressed together with sphere
	const Vec4* spheres = reinterpret_cast<const Vec4*>(sphereData.data());
	const u16* matIndices = reinterpret_cast<const u16*>(sphereData.data() + lod.numSpheres
														 * 4u * sizeof(float));

	std::size_t readSpheres = 0u;
	BulkLoader spheresBulk{ BulkLoader::BULK_ARRAY };
	spheresBulk.descriptor.bytes = reinterpret_cast<const char*>(spheres);
	BulkLoader matsBulk{ BulkLoader::BULK_ARRAY };
	matsBulk.descriptor.bytes = reinterpret_cast<const char*>(matIndices);
	AABB aabb{
		util::pun<Vec3>(object.aabb.min),
		util::pun<Vec3>(object.aabb.max)
	};

	if(spheres_add_sphere_bulk(lod.lodHdl, lod.numSpheres, &spheresBulk, &aabb, &readSpheres) == INVALID_INDEX)
		throw std::runtime_error("Failed to add spheres to object '" + object.name + "'");
	if(spheres_set_material_idx_bulk(lod.lodHdl, static_cast<SphereHdl>(0u),
									 lod.numSpheres, &matsBulk) == INVALID_SIZE)
		throw std::runtime_error("Failed to set sphere materials for object '"
								 + object.name + "'");
	logPedantic("[BinaryLoader::read_compressed_spheres] Object '", object.name, "': Read ",
				lod.numSpheres, " deflated spheres");
	read_compressed_sphere_attributes(object, lod);
}
void BinaryLoader::read_compressed_vertex_attributes(const ObjectState& object, const LodState& lod) {
	if(lod.numVertices == 0 || lod.numVertAttribs == 0)
		return;

	if(read<u32>() != ATTRIBUTE_MAGIC)
		throw std::runtime_error("Invalid attribute magic constant (object '"
								 + object.name + "'");

	std::vector<unsigned char> attributeData = decompress();
	const unsigned char* attributes = attributeData.data();
	BulkLoader attrBulk{ BulkLoader::BULK_ARRAY };
	attrBulk.descriptor.bytes = reinterpret_cast<const char*>(attributes);

	for(u32 i = 0u; i < lod.numVertAttribs; ++i) {
		AttribState state = read_compressed_attribute(attributes);
		auto attrHdl = polygon_request_vertex_attribute(lod.lodHdl, state.name.c_str(),
														state.type);
		if(attrHdl.index == INVALID_INDEX)
			throw std::runtime_error("Failed to add vertex attribute to object '" + object.name + "'");

		if(polygon_set_vertex_attribute_bulk(lod.lodHdl, &attrHdl, 0, lod.numVertices, &attrBulk) == INVALID_SIZE)
			throw std::runtime_error("Failed to set vertex attribute data for object '" + object.name + "'");
		
		attrBulk.descriptor.bytes += state.bytes;
	}
	logPedantic("[BinaryLoader::read_compressed_vertex_attributes] Object '", object.name, "': Read ",
				lod.numVertAttribs, " deflated vertex attributes");
}

void BinaryLoader::read_compressed_face_attributes(const ObjectState& object, const LodState& lod) {
	if((lod.numTriangles == 0 && lod.numQuads == 0)
	   || lod.numFaceAttribs == 0)
		return;

	if(read<u32>() != ATTRIBUTE_MAGIC)
		throw std::runtime_error("Invalid attribute magic constant (object '"
								 + object.name + "'");

	std::vector<unsigned char> attributeData = decompress();
	const unsigned char* attributes = attributeData.data();
	BulkLoader attrBulk{ BulkLoader::BULK_ARRAY };
	attrBulk.descriptor.bytes = reinterpret_cast<const char*>(attributes);

	for(u32 i = 0u; i < lod.numFaceAttribs; ++i) {
		AttribState state = read_compressed_attribute(attributes);
		auto attrHdl = polygon_request_face_attribute(lod.lodHdl, state.name.c_str(),
														state.type);
		if(attrHdl.index == INVALID_INDEX)
			throw std::runtime_error("Failed to add face attribute to object '" + object.name + "'");

		if(polygon_set_face_attribute_bulk(lod.lodHdl, &attrHdl, 0, lod.numTriangles + lod.numQuads, &attrBulk) == INVALID_SIZE)
			throw std::runtime_error("Failed to set face attribute data for object '" + object.name + "'");

		attrBulk.descriptor.bytes += state.bytes;
	}
	logPedantic("[BinaryLoader::read_compressed_face_attributes] Object '", object.name, "': Read ",
				lod.numVertAttribs, " deflated face attributes");
}

void BinaryLoader::read_compressed_face_materials(const ObjectState& object, const LodState& lod) {
	if(lod.numTriangles == 0 && lod.numQuads == 0)
		return;

	std::vector<unsigned char> matData = decompress();
	const u16* matIndices = reinterpret_cast<const u16*>(matData.data());
	BulkLoader matsBulk{ BulkLoader::BULK_ARRAY };
	matsBulk.descriptor.bytes = reinterpret_cast<const char*>(matIndices);

	if(polygon_set_material_idx_bulk(lod.lodHdl, 0, lod.numTriangles + lod.numQuads, &matsBulk) == INVALID_SIZE)
		throw std::runtime_error("Failed to set face material for object '"
									+ object.name + "'");
	logPedantic("[BinaryLoader::read_compressed_face_materials] Object '", object.name, "': Read ",
				lod.numTriangles + lod.numQuads, " deflated face material indices");
}

void BinaryLoader::read_compressed_sphere_attributes(const ObjectState& object, const LodState& lod) {
	if(lod.numSpheres == 0 || lod.numSphereAttribs == 0)
		return;

	if(read<u32>() != ATTRIBUTE_MAGIC)
		throw std::runtime_error("Invalid attribute magic constant (object '"
								 + object.name + "'");
	std::vector<unsigned char> attributeData = decompress();
	const unsigned char* attributes = attributeData.data();
	BulkLoader attrBulk{ BulkLoader::BULK_ARRAY };
	attrBulk.descriptor.bytes = reinterpret_cast<const char*>(attributes);

	for(u32 i = 0u; i < lod.numSphereAttribs; ++i) {
		AttribState state = read_compressed_attribute(attributes);
		auto attrHdl = spheres_request_attribute(lod.lodHdl, state.name.c_str(), state.type);
		if(attrHdl.index == INVALID_INDEX)
			throw std::runtime_error("Failed to add sphere attribute to object '" + object.name + "'");

		if(spheres_set_attribute_bulk(lod.lodHdl, &attrHdl, 0, lod.numSpheres, &attrBulk) == INVALID_SIZE)
			throw std::runtime_error("Failed to set sphere attribute data for object '" + object.name + "'");

		attrBulk.descriptor.bytes += state.bytes;
	}
	logPedantic("[BinaryLoader::read_compressed_sphere_attributes] Object '", object.name, "': Read ",
				lod.numSpheres, " deflated sphere material indices");
}

void BinaryLoader::read_lod(const ObjectState& object, u32 lod) {
	auto scope = Profiler::instance().start<CpuProfileState>("BinaryLoader::read_lod");

	// Remember where we were in the file
	const std::ifstream::off_type currOffset = m_fileStream.tellg() - m_fileStart;
	// Jump to the object
	m_fileStream.seekg(object.offset + sizeof(u32), std::ifstream::beg);
	// Skip the object name + find the jump table
	m_fileStream.seekg(read<u32>() + sizeof(u32) * 9u, std::ifstream::cur);
	// Jump to the LoD
	const u32 lods = read<u32>();
	if(lod >= lods)
		throw std::runtime_error("LoD out of range (" + std::to_string(lod) + " >= " + std::to_string(lods));
	m_fileStream.seekg(sizeof(u64) * lod, std::ifstream::cur);
	m_fileStream.seekg(read<u64>(), std::ifstream::beg);

	if(read<u32>() != LOD_MAGIC)
		throw std::runtime_error("Invalid LoD magic constant (object '" + object.name + "', LoD "
								 + std::to_string(lod) + ")");

	// Read the LoD information
	LodState lodState;
	lodState.numTriangles = read<u32>();
	lodState.numQuads = read<u32>();
	lodState.numSpheres = read<u32>();
	lodState.numVertices = read<u32>();
	lodState.numEdges = read<u32>();
	lodState.numVertAttribs = read<u32>();
	lodState.numFaceAttribs = read<u32>();
	lodState.numSphereAttribs = read<u32>();
	lodState.lodHdl = object_add_lod(object.objHdl, lod);

	logPedantic("[BinaryLoader::read_lod] Loading LoD ", lod, " for object '", object.name, "'...");

	// Reserve memory for the current LOD
	if(!polygon_reserve(lodState.lodHdl, lodState.numVertices,
					   lodState.numEdges, lodState.numTriangles,
					   lodState.numQuads))
		throw std::runtime_error("Failed to reserve LoD polygon memory");
	if (!spheres_reserve(lodState.lodHdl, lodState.numSpheres))
		throw std::runtime_error("Failed to reserve LoD sphere memory");
	if(object.globalFlags.is_set(GlobalFlag::DEFLATE)) {
		if(object.globalFlags.is_set(GlobalFlag::COMPRESSED_NORMALS))
			read_compressed_normal_compressed_vertices(object, lodState);
		else
			read_compressed_normal_uncompressed_vertices(object, lodState);
		read_compressed_vertex_attributes(object, lodState);
		read_compressed_triangles(object, lodState);
		read_compressed_quads(object, lodState);
		read_compressed_face_materials(object, lodState);
		read_compressed_face_attributes(object, lodState);
		read_compressed_spheres(object, lodState);
	} else {
		// First comes vertex data
		if(object.globalFlags.is_set(GlobalFlag::COMPRESSED_NORMALS))
			read_normal_compressed_vertices(object, lodState);
		else
			read_normal_uncompressed_vertices(object, lodState);
		read_uncompressed_vertex_attributes(object, lodState);
		read_uncompressed_triangles(object, lodState);
		read_uncompressed_quads(object, lodState);
		read_uncompressed_face_materials(object, lodState);
		read_uncompressed_face_attributes(object, lodState);
		read_uncompressed_spheres(object, lodState);

		logInfo("[BinaryLoader::read_lod] Loaded LoD ", lod, " of object object '",
				object.name, "' with ", lodState.numVertices, " vertices, ",
				lodState.numTriangles, " triangles, ", lodState.numQuads, " quads, ",
				lodState.numSpheres, " spheres");
	}

	// Restore the file state
	m_fileStream.seekg(currOffset, std::ifstream::beg);
}

void BinaryLoader::read_object() {
	m_objects.back().name = read<std::string>();
	logInfo("[BinaryLoader::read_object] Loading object '", m_objects.back().name, "'");
	m_objects.back().flags = static_cast<ObjectFlags>( read<u32>() );
	m_objects.back().keyframe = read<u32>();
	m_objects.back().animObjId = read<u32>();
	m_objects.back().aabb = read<ei::Box>();

	logPedantic("[BinaryLoader::read_object] Read object '", m_objects.back().name,
				"' with key frame ", m_objects.back().keyframe, " and animation object ID ",
				m_objects.back().animObjId);

	// Create the object for direct writing
	m_objects.back().objHdl = world_create_object(m_objects.back().name.c_str(), m_objects.back().flags);
	if(m_objects.back().objHdl == nullptr)
		throw std::runtime_error("Failed to create object '" + m_objects.back().name);
}

bool BinaryLoader::read_instances(const u32 globalLod,
								  const std::unordered_map<std::string_view, u32>& objectLods,
								  const std::unordered_map<std::string_view, u32>& instanceLods) {
	auto scope = Profiler::instance().start<CpuProfileState>("BinaryLoader::read_instances");
	std::vector<uint8_t> hasInstance(m_objects.size(), false);
	const u32 numInstances = read<u32>();
	for(u32 i = 0u; i < numInstances; ++i) {
		if(m_abort)
			return false;
		const std::string name = read<std::string>();
		const u32 objId = read<u32>();
		const u32 keyframe = read<u32>();
		const u32 animInstId = read<u32>();
		const Mat3x4 transMat = read<Mat3x4>();
		// Determine what level-of-detail should be applied for this instance
		u32 lod = globalLod;
		if(auto iter = objectLods.find(m_objects[objId].name); iter != objectLods.end())
			lod = iter->second;
		if(auto iter = instanceLods.find(name); iter != instanceLods.end())
			lod = iter->second;

		// Check if the instance scaling is uniform
		const ei::Mat3x3 rotScale{
			transMat.v[0u], transMat.v[1u], transMat.v[2u],
			transMat.v[4u], transMat.v[5u], transMat.v[6u],
			transMat.v[8u], transMat.v[9u], transMat.v[10u]
		};
		const float scaleX = ei::lensq(rotScale(0u));
		const float scaleY = ei::lensq(rotScale(1u));
		const float scaleZ = ei::lensq(rotScale(2u));
		if(!ei::approx(scaleX, scaleY) || !ei::approx(scaleX, scaleZ)) {
			logWarning("[BinaryLoader::read_instances] Instance ", i, " of object ", objId, " has non-uniform scaling (",
					   scaleX, "|", scaleY, "|", scaleZ, "), which we don't support; ignoring instance");
			continue;
		}
		logPedantic("[BinaryLoader::read_instances] Creating given instance (keyframe ", keyframe,
					", animInstId ", animInstId, ") for object '", name, "\'");
		InstanceHdl instHdl = world_create_instance(name.c_str(), m_objects[objId].objHdl);
		if(instHdl == nullptr)
			throw std::runtime_error("Failed to create instance for object ID "
									 + std::to_string(objId));

		// We now have a valid instance: time to check if we have the required LoD
		if(!object_has_lod(m_objects[objId].objHdl, lod)) {
			// We don't -> gotta load it
			read_lod(m_objects[objId], lod);
		}

		if(!instance_set_transformation_matrix(instHdl, &transMat))
			throw std::runtime_error("Failed to set transformation matrix for instance of object ID "
									 + std::to_string(objId));

		ei::Box instanceAabb;
		if(!instance_get_bounding_box(instHdl, reinterpret_cast<Vec3*>(&instanceAabb.min), reinterpret_cast<Vec3*>(&instanceAabb.max), lod))
			throw std::runtime_error("Failed to get bounding box for instance of object ID "
									 + std::to_string(objId));
		m_aabb = ei::Box(m_aabb, instanceAabb);

		hasInstance[objId] = true;
	}

	for(u32 i = 0u; i < static_cast<u32>(hasInstance.size()); ++i) {
		if(m_abort)
			return false;
		if(!hasInstance[i]) {
			std::string_view objName = m_objects[i].name;
			logPedantic("[BinaryLoader::read_instances] Creating default instance for object '",
						objName, "\'");
			// Add default instance
			std::string name = std::string(objName) + std::string("###defaultInstance");
			// Determine what level-of-detail should be applied for this instance
			u32 lod = globalLod;
			if(auto iter = objectLods.find(objName); iter != objectLods.end())
				lod = iter->second;
			InstanceHdl instHdl = world_create_instance(name.c_str(), m_objects[i].objHdl);
			if(instHdl == nullptr)
				throw std::runtime_error("Failed to create instance for object ID "
										 + std::to_string(i));

			// We now have a valid instance: time to check if we have the required LoD
			if(!object_has_lod(m_objects[i].objHdl, lod)) {
				// We don't -> gotta load it
				read_lod(m_objects[i], lod);
			}

			ei::Box instanceAabb;
			if(!instance_get_bounding_box(instHdl,  reinterpret_cast<Vec3*>(&instanceAabb.min),  reinterpret_cast<Vec3*>(&instanceAabb.max), lod))
				throw std::runtime_error("Failed to get bounding box for instance of object ID "
										 + std::to_string(i));
			m_aabb = ei::Box(m_aabb, instanceAabb);
		}
	}

	return true;
	// Create identity instances for objects not having one yet
}

void BinaryLoader::load_lod(const fs::path& file, mufflon::u32 objId, mufflon::u32 lod) {
	auto scope = Profiler::instance().start<CpuProfileState>("BinaryLoader::load_lod");
	m_filePath = file;

	if(!fs::exists(m_filePath))
		throw std::runtime_error("Binary file '" + m_filePath.string() + "' doesn't exist");

	logInfo("[BinaryLoader::load_lod] Loading LoD ", lod, " for object ID ", objId,
			" from file '", m_filePath.string(), "'");

	try {
		// Open the binary file and enable exception management
		m_fileStream = std::ifstream(m_filePath, std::ios_base::binary);
		if(m_fileStream.bad() || m_fileStream.fail())
			throw std::runtime_error("Failed to open binary file '" + m_filePath.string() + "\'");
		m_fileStream.exceptions(std::ifstream::failbit);

		// Skip over the materials header
		if(read<u32>() != MATERIALS_HEADER_MAGIC)
			throw std::runtime_error("Invalid materials header magic constant");

		m_fileStart = m_fileStream.tellg();

		const u64 objectStart = read<u64>();
		m_fileStream.seekg(objectStart, std::ifstream::beg);

		// Parse the object header
		if(read<u32>() != OBJECTS_HEADER_MAGIC)
			throw std::runtime_error("Invalid objects header magic constant");
		(void) read<u64>(); // Instance start
		GlobalFlag compressionFlags = GlobalFlag{ { read<u32>() } };

		// Jump to the desired object
		const u32 jumpCount = read<u32>();
		if(objId >= jumpCount)
			throw std::runtime_error("Object index out of bounds (" + std::to_string(objId)
									 + " >= " + std::to_string(jumpCount) + ")");
		m_fileStream.seekg(sizeof(u64) * objId, std::ifstream::cur);

		ObjectState object;
		object.offset = read<u64>();
		m_fileStream.seekg(object.offset, std::ifstream::beg);
		// Parse the object data
		object.globalFlags = compressionFlags;
		if(read<u32>() != OBJECT_MAGIC)
			throw std::runtime_error("Invalid object magic constant (object " + std::to_string(objId) + ")");
		object.name = read<std::string>();
		object.flags = static_cast<ObjectFlags>(read<u32>());
		object.keyframe = read<u32>();
		object.animObjId = read<u32>();
		object.aabb = read<ei::Box>();
		object.objHdl = world_get_object(object.name.c_str());
		if(object.objHdl == nullptr)
			throw std::runtime_error("Unknown object '" + object.name + ")");
		// Read the LoD
		if(!object_has_lod(object.objHdl, lod))
			read_lod(object, lod);

		this->clear_state();
	} catch(const std::exception&) {
		// Clean up before leaving throwing
		this->clear_state();
		throw;
	}
}

bool BinaryLoader::load_file(fs::path file, const u32 globalLod,
							 const std::unordered_map<std::string_view, mufflon::u32>& objectLods,
							 const std::unordered_map<std::string_view, mufflon::u32>& instanceLods) {
	auto scope = Profiler::instance().start<CpuProfileState>("BinaryLoader::load_file");
	m_filePath = std::move(file);
	if(!fs::exists(m_filePath))
		throw std::runtime_error("Binary file '" + m_filePath.string() + "' doesn't exist");
	m_aabb.min = ei::Vec3{ 1e30f };
	m_aabb.max = ei::Vec3{ -1e30f };
	logInfo("[BinaryLoader::load_file] Loading binary file '", m_filePath.string(), "'");
	try {
		// Open the binary file and enable exception management
		m_fileStream = std::ifstream(m_filePath, std::ios_base::binary);
		if(m_fileStream.bad() || m_fileStream.fail())
			throw std::runtime_error("Failed to open binary file '" + m_filePath.string() + "\'");
		m_fileStream.exceptions(std::ifstream::failbit);
		// Needed to get a C file descriptor offset
		m_fileStart = m_fileStream.tellg();

		if(m_abort)
			return false;

		// Read the materials header
		if(read<u32>() != MATERIALS_HEADER_MAGIC)
			throw std::runtime_error("Invalid materials header magic constant");
		const u64 objectStart = read<u64>();
		const u32 numMaterials = read<u32>();
		// Read the material names (and implicitly their indices)
		m_materialNames.reserve(numMaterials);
		for(u32 i = 0u; i < numMaterials; ++i) {
			m_materialNames.push_back(read<std::string>());
		}
		if(m_abort)
			return false;

		// Jump to the location of objects
		m_fileStream.seekg(objectStart, std::ifstream::beg);
		// Parse the object header
		if(read<u32>() != OBJECTS_HEADER_MAGIC)
			throw std::runtime_error("Invalid objects header magic constant");
		const u64 instanceStart = read<u64>();
		GlobalFlag compressionFlags = GlobalFlag{ { read<u32>() } };

		// Parse the object jumptable
		m_objJumpTable.resize(read<u32>());
		for(std::size_t i = 0u; i < m_objJumpTable.size(); ++i)
			m_objJumpTable[i] = read<u64>();

		// Next come the objects
		for(std::size_t i = 0u; i < m_objJumpTable.size(); ++i) {
			if(m_abort)
				return false;
			m_objects.push_back(ObjectState{});
			// Jump to the right position in file
			m_fileStream.seekg(m_objJumpTable[i], std::ifstream::beg);
			m_objects.back().offset = m_objJumpTable[i];
			m_objects.back().globalFlags = compressionFlags;

			// Read object
			if(read<u32>() != OBJECT_MAGIC)
				throw std::runtime_error("Invalid object magic constant (object " + std::to_string(i) + ")");
			read_object();
		}

		// Now come instances
		m_fileStream.seekg(instanceStart, std::ios_base::beg);
		if(read<u32>() != INSTANCE_MAGIC)
			throw std::runtime_error("Invalid instance magic constant");
		if(!read_instances(globalLod, objectLods, instanceLods))
			return false;

		this->clear_state();
	} catch(const std::exception&) {
		// Clean up before leaving throwing
		this->clear_state();
		throw;
	}
	return true;
}

} // namespace mff_loader::binary