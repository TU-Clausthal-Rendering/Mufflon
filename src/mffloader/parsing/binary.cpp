#include "binary.hpp"
#include "util/log.hpp"
#include "util/punning.hpp"
#include "util/string_view.hpp"
#include "profiler/cpu_profiler.hpp"
#include <ei/conversions.hpp>
#include <ei/3dtypes.hpp>
#include <ei/vector.hpp>
#include <miniz/miniz.h>
#include <cstdio>

namespace mff_loader::binary {

using namespace mufflon;

BinaryLoader::FileDescriptor::FileDescriptor(fs::path file) :
#ifdef _WIN32
	m_desc(_wfopen(file.wstring().c_str(), L"rb"))
#else // _WIN32
	m_desc(std::fopen(file.u8string().c_str(), "rb"))
#endif // _WIN32
{
	if(m_desc == nullptr)
		throw std::ios_base::failure("Failed to open file '" + file.u8string()
									 + "' with mode 'rb'");
}

BinaryLoader::FileDescriptor::FileDescriptor(FileDescriptor&& other) :
	m_desc{ other.m_desc } {
	other.m_desc = nullptr;
}

BinaryLoader::FileDescriptor& BinaryLoader::FileDescriptor::operator=(FileDescriptor&& other) {
	std::swap(m_desc, other.m_desc);
	return *this;
}

BinaryLoader::FileDescriptor::~FileDescriptor() {
	this->close();
}

void BinaryLoader::FileDescriptor::close() noexcept {
	if(m_desc != nullptr) {
		// Nothing we can do if this fails - not like we care, either
		(void)std::fclose(m_desc);
		m_desc = nullptr;
	}
}

// Advances a C file descriptor (possibly in multiple steps)
BinaryLoader::FileDescriptor& BinaryLoader::FileDescriptor::seek(u64 offset, std::ios_base::seekdir dir) {
	int origin = 0u;
	switch(dir) {
		case std::ios_base::beg: origin = SEEK_SET; break;
		case std::ios_base::end: origin = SEEK_END; break;
		case std::ios_base::cur:
		default: origin = SEEK_CUR;
	}
	while(offset > static_cast<u64>(std::numeric_limits<long>::max())) {
		if(std::fseek(m_desc, std::numeric_limits<long>::max(), origin) != 0)
			throw std::runtime_error("Failed to seek C file descriptor to desired position");
		offset -= std::numeric_limits<long>::max();
		origin = SEEK_CUR;
	}
	if(std::fseek(m_desc, static_cast<long>(offset), origin) != 0)
		throw std::runtime_error("Failed to seek C file descriptor to desired position");
	return *this;
}

struct ObjectInfoNoName {
	u32 flags;
	u32 keyframe;
	u32 prevAnimObjId;
	ei::Box aabb;
};
struct InstanceInfoNoName {
	u32 objId;
	u32 keyframe;
	u32 prevAnimInstId;
	Mat3x4 transMat;
};


// Read a string as specified by the file format
template <>
std::string BinaryLoader::read<std::string>() {
	std::string str;
	str.resize(read<u32>());
	m_fileStream.read(str.data(), str.length());
	return str;
}
template <>
StringView BinaryLoader::read<StringView>() {
	static thread_local std::string buffer;
	buffer.clear();
	buffer.resize(read<u32>());
	m_fileStream.read(buffer.data(), buffer.length());
	return buffer;
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
	// TODO: ensure this is packed!
	// TODO: ensure that we only read it at once if endianness matches!
	Mat3x4 mat;
	m_fileStream.read(reinterpret_cast<char*>(&mat), sizeof(mat));
	return mat;
}
// Read a AABB as specified by the file format
template <>
ei::Box BinaryLoader::read<ei::Box>() {
	// TODO: ensure this is packed!
	// TODO: ensure that we only read it at once if endianness matches!
	ei::Box box;
	m_fileStream.read(reinterpret_cast<char*>(&box), sizeof(box));
	return box;
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
// Read the entire struct at once
template <>
ObjectInfoNoName BinaryLoader::read<ObjectInfoNoName>() {
	// TODO: ensure this is packed!
	// TODO: ensure that we only read it at once if endianness matches!
	ObjectInfoNoName info;
	m_fileStream.read(reinterpret_cast<char*>(&info), sizeof(info));
	return info;
}
template <>
InstanceInfoNoName BinaryLoader::read<InstanceInfoNoName>() {
	// TODO: ensure this is packed!
	// TODO: ensure that we only read it at once if endianness matches!
	InstanceInfoNoName info;
	m_fileStream.read(reinterpret_cast<char*>(&info), sizeof(info));
	return info;
}

// Clears the loader state
void BinaryLoader::clear_state() {
	if(m_fileStream.is_open())
		m_fileStream.close();
	m_objects.clear();
	m_objJumpTable.clear();
	m_namePool.clear();
}

GeomAttributeType BinaryLoader::map_bin_attrib_type(AttribType type) {
	return static_cast<GeomAttributeType>(type);
}

// Read vertices with applied normal compression, but no deflating
void BinaryLoader::read_normal_compressed_vertices(const ObjectState& object, const LodState& lod) {
	if(lod.numVertices == 0)
		return;
	const std::ifstream::off_type currOffset = m_fileStream.tellg() - m_fileStart;

	m_fileDescs[0u].seek(currOffset, std::ios_base::beg);
	m_fileDescs[2u].seek(currOffset + (3u * sizeof(float) + sizeof(u32)) * lod.numVertices, std::ios_base::beg);
	std::size_t pointsRead = 0u;
	std::size_t uvsRead = 0u;

	BulkLoader pointsBulk{ BulkType::BULK_FILE, { m_fileDescs[0u].get() } };
	BulkLoader uvsBulk{ BulkType::BULK_FILE, { m_fileDescs[2u].get() } };
	AABB aabb{
		util::pun<Vec3>(object.aabb.min),
		util::pun<Vec3>(object.aabb.max)
	};
	if(polygon_add_vertex_bulk(lod.lodHdl, lod.numVertices, &pointsBulk, nullptr, &uvsBulk,
							   &aabb, &pointsRead, nullptr, &uvsRead) == INVALID_INDEX)
		throw std::runtime_error("Failed to bulk-read vertices for object '" + std::string(object.name) + "'");
	logPedantic("[BinaryLoader::read_normal_compressed_vertices] Object '", object.name, "': Read ",
				pointsRead, "'", uvsRead, " point/UVs bulk");
	if(pointsRead != lod.numVertices || uvsRead != lod.numVertices)
		throw std::runtime_error("Not all vertices were fully read for object '" + std::string(object.name) + "'");
	// Unpack the normals
	m_fileStream.seekg(3u * sizeof(float) * lod.numVertices, std::ios_base::cur);
	for(u32 n = 0u; n < lod.numVertices; ++n) {
		const ei::Vec3 normal = ei::normalize(ei::unpackOctahedral32(read<u32>()));
		if(!polygon_set_vertex_normal(lod.lodHdl, static_cast<VertexHdl>(n), util::pun<Vec3>(normal)))
			throw std::runtime_error("Failed to set normal for object '" + std::string(object.name) + "'");
	}
	// Skip past the UV coordinates again
	m_fileStream.seekg(2u * sizeof(float) * lod.numVertices, std::ios_base::cur);
}

// Read vertices without deflation and without normal compression
void BinaryLoader::read_normal_uncompressed_vertices(const ObjectState& object, const LodState& lod) {
	auto scope = Profiler::loader().start<CpuProfileState>("BinaryLoader::read_normal_uncompressed_vertices");
	if(lod.numVertices == 0)
		return;
	const std::ifstream::off_type currOffset = m_fileStream.tellg() - m_fileStart;

	m_fileDescs[0u].seek(currOffset, std::ios_base::beg);
	m_fileDescs[1u].seek(currOffset + 3u * sizeof(float) * lod.numVertices, std::ios_base::beg);
	m_fileDescs[2u].seek(currOffset + 2u * 3u * sizeof(float) * lod.numVertices, std::ios_base::beg);
	std::size_t pointsRead = 0u;
	std::size_t normalsRead = 0u;
	std::size_t uvsRead = 0u;
	BulkLoader pointsBulk{ BulkType::BULK_FILE, { m_fileDescs[0u].get() } };
	BulkLoader normalsBulk{ BulkType::BULK_FILE, { m_fileDescs[1u].get() } };
	BulkLoader uvsBulk{ BulkType::BULK_FILE, { m_fileDescs[2u].get() } };
	AABB aabb{
		util::pun<Vec3>(object.aabb.min),
		util::pun<Vec3>(object.aabb.max)
	};
	if(polygon_add_vertex_bulk(lod.lodHdl, lod.numVertices, &pointsBulk, &normalsBulk, &uvsBulk,
							   &aabb, &pointsRead, &normalsRead, &uvsRead) == INVALID_INDEX)
		throw std::runtime_error("Failed to bulk-read vertices for object '" + std::string(object.name) + "'");
	logPedantic("[BinaryLoader::read_normal_uncompressed_vertices] Object '", object.name, "': Read ",
				pointsRead, "'", normalsRead, "'", uvsRead, " point/normals/UVs bulk");
	if(pointsRead != lod.numVertices || normalsRead != lod.numVertices
	   || uvsRead != lod.numVertices)
		throw std::runtime_error("Not all vertices were fully read for object '" + std::string(object.name) + "'");
	// Seek to the attributes
	m_fileStream.seekg((2u * 3u + 2u) * sizeof(float) * lod.numVertices, std::ios_base::cur);
}

// Reads the attribute parameters, but not the actual data
void BinaryLoader::read_uncompressed_attribute() {
	read(m_attribStateBuffer.name);
	read(m_attribStateBuffer.meta);
	m_attribStateBuffer.metaFlags = read<u32>();
	m_attribStateBuffer.type = map_bin_attrib_type(static_cast<AttribType>(read<u32>()));
	m_attribStateBuffer.bytes = read<u64>();
}
void BinaryLoader::read_compressed_attribute(const unsigned char*& data) {
	read(m_attribStateBuffer.name, data);
	read(m_attribStateBuffer.meta, data);
	m_attribStateBuffer.metaFlags = read<u32>(data);
	m_attribStateBuffer.type = map_bin_attrib_type(static_cast<AttribType>(read<u32>(data)));
	m_attribStateBuffer.bytes = read<u64>(data);
}

// Reads the acctual data into the object
void BinaryLoader::read_uncompressed_vertex_attributes(const ObjectState& object, const LodState& lod) {
	if(lod.numVertices == 0 || lod.numVertAttribs == 0)
		return;
	BulkLoader attrBulk{ BulkType::BULK_FILE, { m_fileDescs[0u].get() } };

	if(read<u32>() != ATTRIBUTE_MAGIC)
		throw std::runtime_error("Invalid attribute magic constant (object '" + std::string(object.name) + "'");
	for(u32 i = 0u; i < lod.numVertAttribs; ++i) {
		read_uncompressed_attribute();
		m_fileDescs[0u].seek(m_fileStream.tellg() - m_fileStart, std::ios_base::beg);
		auto attrHdl = polygon_request_vertex_attribute(lod.lodHdl, m_attribStateBuffer.name.c_str(),
														m_attribStateBuffer.type);
		if(attrHdl.name == nullptr)
			throw std::runtime_error("Failed to add vertex attribute to object '" + std::string(object.name) + "'");
		if(polygon_set_vertex_attribute_bulk(lod.lodHdl, attrHdl, 0u,
											 lod.numVertices,
											 &attrBulk) == INVALID_SIZE)
			throw std::runtime_error("Failed to set vertex attribute data for object '" + std::string(object.name) + "'");
		// Seek past the attribute
		m_fileStream.seekg(m_attribStateBuffer.bytes, std::ios_base::cur);
	}
	logPedantic("[BinaryLoader::read_uncompressed_vertex_attributes] Object '", object.name, "': Read ",
				lod.numVertAttribs, " vertex attributes");
}

// Reads the acctual data into the object
void BinaryLoader::read_uncompressed_face_attributes(const ObjectState& object, const LodState& lod) {
	if((lod.numTriangles == 0 && lod.numQuads == 0) || lod.numFaceAttribs == 0)
		return;
	m_fileDescs[0u].seek(m_fileStream.tellg() - m_fileStart, std::ios_base::beg);
	BulkLoader attrBulk{ BulkType::BULK_FILE, { m_fileDescs[0u].get() } };

	if(read<u32>() != ATTRIBUTE_MAGIC)
		throw std::runtime_error("Invalid attribute magic constant (object '" + std::string(object.name) + "'");
	for(u32 i = 0u; i < lod.numVertAttribs; ++i) {
		read_uncompressed_attribute();
		auto attrHdl = polygon_request_face_attribute(lod.lodHdl, m_attribStateBuffer.name.c_str(),
													  m_attribStateBuffer.type);
		if(attrHdl.name == nullptr)
			throw std::runtime_error("Failed to add face attribute to object '" + std::string(object.name) + "'");
		if(polygon_set_face_attribute_bulk(lod.lodHdl, attrHdl, 0u,
										   lod.numTriangles + lod.numQuads,
										   &attrBulk) == INVALID_SIZE)
			throw std::runtime_error("Failed to set face attribute data for object '" + std::string(object.name) + "'");
		// Seek past the attribute
		m_fileStream.seekg(m_attribStateBuffer.bytes, std::ios_base::cur);
	}
	logPedantic("[BinaryLoader::read_uncompressed_face_attributes] Object '", object.name, "': Read ",
				lod.numFaceAttribs, " face attributes");
}

// Reads the acctual data into the object
void BinaryLoader::read_uncompressed_sphere_attributes(const ObjectState& object, const LodState& lod) {
	if(lod.numSpheres == 0 || lod.numSphereAttribs == 0)
		return;
	m_fileDescs[0u].seek(m_fileStream.tellg() - m_fileStart, std::ios_base::beg);
	BulkLoader attrBulk{ BulkType::BULK_FILE, { m_fileDescs[0u].get() } };

	if(read<u32>() != ATTRIBUTE_MAGIC)
		throw std::runtime_error("Invalid attribute magic constant (object '" + std::string(object.name) + "'");
	for(u32 i = 0u; i < lod.numVertAttribs; ++i) {
		read_uncompressed_attribute();
		auto attrHdl = spheres_request_attribute(lod.lodHdl, m_attribStateBuffer.name.c_str(),
												 m_attribStateBuffer.type);
		if(attrHdl.name == nullptr)
			throw std::runtime_error("Failed to add sphere attribute to object '" + std::string(object.name) + "'");
		if(spheres_set_attribute_bulk(lod.lodHdl, attrHdl, 0u,
									  lod.numSpheres,
									  &attrBulk) == INVALID_SIZE)
			throw std::runtime_error("Failed to set sphere attribute data for object '" + std::string(object.name) + "'");
		// Seek past the attribute
		m_fileStream.seekg(m_attribStateBuffer.bytes, std::ios_base::cur);
	}
	logPedantic("[BinaryLoader::read_uncompressed_sphere_attributes] Object '", object.name, "': Read ",
				lod.numSphereAttribs, " sphere attributes");
}

void BinaryLoader::read_uncompressed_face_materials(const ObjectState& object, const LodState& lod) {
	if(lod.numTriangles == 0 && lod.numQuads == 0)
		return;
	m_fileDescs[0u].seek(m_fileStream.tellg() - m_fileStart, std::ios_base::beg);
	BulkLoader matsBulk{ BulkType::BULK_FILE, { m_fileDescs[0u].get() } };

	const u32 faces = lod.numTriangles + lod.numQuads;
	if(polygon_set_material_idx_bulk(lod.lodHdl, static_cast<FaceHdl>(0u),
									 faces, &matsBulk) == INVALID_SIZE)
		throw std::runtime_error("Failed to set face material for object '"
								 + std::string(object.name) + "'");
	logPedantic("[BinaryLoader::read_uncompressed_face_materials] Object '", object.name, "': Read ",
				faces, " face material indices");
	// Seek to face attributes
	m_fileStream.seekg(sizeof(u16) * faces, std::ios_base::cur);
}

void BinaryLoader::read_uncompressed_sphere_materials(const ObjectState& object, const LodState& lod) {
	if(lod.numSpheres == 0)
		return;
	m_fileDescs[0u].seek(m_fileStream.tellg() - m_fileStart, std::ios_base::beg);
	BulkLoader matsBulk{ BulkType::BULK_FILE, { m_fileDescs[0u].get() } };

	if(spheres_set_material_idx_bulk(lod.lodHdl, static_cast<SphereHdl>(0u),
									 lod.numSpheres, &matsBulk) == INVALID_SIZE)
		throw std::runtime_error("Failed to set face material for object '"
								 + std::string(object.name) + "'");
	logPedantic("[BinaryLoader::read_uncompressed_face_materials] Object '", object.name, "': Read ",
				lod.numSpheres, " sphere material indices");
	// Seek to face attributes
	m_fileStream.seekg(sizeof(u16) * lod.numSpheres, std::ios_base::cur);
}

void BinaryLoader::read_uncompressed_triangles(const ObjectState& object, const LodState& lod) {
	auto scope = Profiler::loader().start<CpuProfileState>("BinaryLoader::read_uncompressed_triangles");
	if(lod.numTriangles == 0)
		return;
	// Read the faces (cannot do that bulk-like)
	for(u32 tri = 0u; tri < lod.numTriangles; ++tri) {
		const ei::UVec3 indices = read<ei::UVec3>();
		if(polygon_add_triangle(lod.lodHdl, util::pun<UVec3>(indices)) == INVALID_INDEX)
			throw std::runtime_error("Failed to add triangle to object '" + std::string(object.name) + "'");
	}
	logPedantic("[BinaryLoader::read_uncompressed_triangles] Object '", object.name, "': Read ",
				lod.numTriangles, " triangles");
}

void BinaryLoader::read_uncompressed_quads(const ObjectState& object, const LodState& lod) {
	auto scope = Profiler::loader().start<CpuProfileState>("BinaryLoader::read_uncompressed_quads");
	if(lod.numQuads == 0)
		return;
	// Read the faces (cannot do that bulk-like)
	for(u32 quad = 0u; quad < lod.numQuads; ++quad) {
		const ei::UVec4 indices = read<ei::UVec4>();
		if(polygon_add_quad(lod.lodHdl, util::pun<UVec4>(indices)) == INVALID_INDEX)
			throw std::runtime_error("Failed to add quad to object '" + std::string(object.name) + "'");
	}
	logPedantic("[BinaryLoader::read_uncompressed_quads] Object '", object.name, "': Read ",
				lod.numQuads, " quads");
}

void BinaryLoader::read_uncompressed_spheres(const ObjectState& object, const LodState& lod) {
	if(lod.numSpheres == 0)
		return;
	m_fileDescs[0u].seek(m_fileStream.tellg() - m_fileStart, std::ios_base::beg);
	BulkLoader spheresBulk{ BulkType::BULK_FILE, { m_fileDescs[0u].get() } };
	AABB aabb{
		util::pun<Vec3>(object.aabb.min),
		util::pun<Vec3>(object.aabb.max)
	};

	std::size_t spheresRead = 0u;
	if(spheres_add_sphere_bulk(lod.lodHdl, lod.numSpheres,
							   &spheresBulk, &aabb, &spheresRead) == INVALID_INDEX)
		throw std::runtime_error("Failed to add spheres to object '" + std::string(object.name) + "'");
	logPedantic("[BinaryLoader::read_uncompressed_spheres] Object '", object.name, "': Read ",
				spheresRead, " spheres bulk");
	if(spheresRead != lod.numSpheres)
		throw std::runtime_error("Not all spheres were fully read for object '" + std::string(object.name) + "'");
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

	BulkLoader pointsBulk{ BulkType::BULK_ARRAY, {} };
	pointsBulk.descriptor.bytes = reinterpret_cast<const char*>(points);
	BulkLoader uvsBulk{ BulkType::BULK_ARRAY, {} };
	uvsBulk.descriptor.bytes = reinterpret_cast<const char*>(uvs);
	std::size_t pointsRead = 0u;
	std::size_t uvsRead = 0u;
	AABB aabb{
		util::pun<Vec3>(object.aabb.min),
		util::pun<Vec3>(object.aabb.max)
	};
	if(polygon_add_vertex_bulk(lod.lodHdl, lod.numVertices, &pointsBulk, nullptr, &uvsBulk,
							   &aabb, &pointsRead, nullptr, &uvsRead) == INVALID_INDEX)
		throw std::runtime_error("Failed to bulk-read vertices for object '" + std::string(object.name) + "'");
	if(pointsRead != lod.numVertices || uvsRead != lod.numVertices)
		throw std::runtime_error("Not all vertices were fully read for object '" + std::string(object.name) + "'");

	for(u32 vertex = 0u; vertex < lod.numVertices; ++vertex) {
		Vec3 unpackedNormal = util::pun<Vec3>(ei::normalize(ei::unpackOctahedral32(packedNormals[vertex])));
		if(!polygon_set_vertex_normal(lod.lodHdl, vertex, unpackedNormal))
			throw std::runtime_error("Failed to set vertex normal for vertex " + std::to_string(vertex)
									 + " to object '" + std::string(object.name) + "'");
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
	BulkLoader pointsBulk{ BulkType::BULK_ARRAY, {} };
	BulkLoader normalsBulk{ BulkType::BULK_ARRAY, {} };
	BulkLoader uvsBulk{ BulkType::BULK_ARRAY, {} };
	pointsBulk.descriptor.bytes = reinterpret_cast<const char*>(points);
	normalsBulk.descriptor.bytes = reinterpret_cast<const char*>(normals);
	uvsBulk.descriptor.bytes = reinterpret_cast<const char*>(uvs);
	AABB aabb{
		util::pun<Vec3>(object.aabb.min),
		util::pun<Vec3>(object.aabb.max)
	};
	if(polygon_add_vertex_bulk(lod.lodHdl, lod.numVertices, &pointsBulk, &normalsBulk, &uvsBulk,
							   &aabb, &pointsRead, &normalsRead, &uvsRead) == INVALID_INDEX)
		throw std::runtime_error("Failed to bulk-read vertices for object '" + std::string(object.name) + "'");
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
			throw std::runtime_error("Failed to add triangle to object '" + std::string(object.name) + "'");
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
			throw std::runtime_error("Failed to add quad to object '" + std::string(object.name) + "'");
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
	BulkLoader spheresBulk{ BulkType::BULK_ARRAY, {} };
	spheresBulk.descriptor.bytes = reinterpret_cast<const char*>(spheres);
	BulkLoader matsBulk{ BulkType::BULK_ARRAY, {} };
	matsBulk.descriptor.bytes = reinterpret_cast<const char*>(matIndices);
	AABB aabb{
		util::pun<Vec3>(object.aabb.min),
		util::pun<Vec3>(object.aabb.max)
	};

	if(spheres_add_sphere_bulk(lod.lodHdl, lod.numSpheres, &spheresBulk, &aabb, &readSpheres) == INVALID_INDEX)
		throw std::runtime_error("Failed to add spheres to object '" + std::string(object.name) + "'");
	if(spheres_set_material_idx_bulk(lod.lodHdl, static_cast<SphereHdl>(0u),
									 lod.numSpheres, &matsBulk) == INVALID_SIZE)
		throw std::runtime_error("Failed to set sphere materials for object '"
								 + std::string(object.name) + "'");
	logPedantic("[BinaryLoader::read_compressed_spheres] Object '", object.name, "': Read ",
				lod.numSpheres, " deflated spheres");
	read_compressed_sphere_attributes(object, lod);
}
void BinaryLoader::read_compressed_vertex_attributes(const ObjectState& object, const LodState& lod) {
	if(lod.numVertices == 0 || lod.numVertAttribs == 0)
		return;

	if(read<u32>() != ATTRIBUTE_MAGIC)
		throw std::runtime_error("Invalid attribute magic constant (object '"
								 + std::string(object.name) + "'");

	std::vector<unsigned char> attributeData = decompress();
	const unsigned char* attributes = attributeData.data();
	BulkLoader attrBulk{ BulkType::BULK_ARRAY, {} };
	attrBulk.descriptor.bytes = reinterpret_cast<const char*>(attributes);

	for(u32 i = 0u; i < lod.numVertAttribs; ++i) {
		read_compressed_attribute(attributes);
		auto attrHdl = polygon_request_vertex_attribute(lod.lodHdl, m_attribStateBuffer.name.c_str(),
														m_attribStateBuffer.type);
		if(attrHdl.name == nullptr)
			throw std::runtime_error("Failed to add vertex attribute to object '" + std::string(object.name) + "'");

		if(polygon_set_vertex_attribute_bulk(lod.lodHdl, attrHdl, 0, lod.numVertices, &attrBulk) == INVALID_SIZE)
			throw std::runtime_error("Failed to set vertex attribute data for object '" + std::string(object.name) + "'");
		
		attrBulk.descriptor.bytes += m_attribStateBuffer.bytes;
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
								 + std::string(object.name) + "'");

	std::vector<unsigned char> attributeData = decompress();
	const unsigned char* attributes = attributeData.data();
	BulkLoader attrBulk{ BulkType::BULK_ARRAY, {} };
	attrBulk.descriptor.bytes = reinterpret_cast<const char*>(attributes);

	for(u32 i = 0u; i < lod.numFaceAttribs; ++i) {
		read_compressed_attribute(attributes);
		auto attrHdl = polygon_request_face_attribute(lod.lodHdl, m_attribStateBuffer.name.c_str(),
													  m_attribStateBuffer.type);
		if(attrHdl.name == nullptr)
			throw std::runtime_error("Failed to add face attribute to object '" + std::string(object.name) + "'");

		if(polygon_set_face_attribute_bulk(lod.lodHdl, attrHdl, 0, lod.numTriangles + lod.numQuads, &attrBulk) == INVALID_SIZE)
			throw std::runtime_error("Failed to set face attribute data for object '" + std::string(object.name) + "'");

		attrBulk.descriptor.bytes += m_attribStateBuffer.bytes;
	}
	logPedantic("[BinaryLoader::read_compressed_face_attributes] Object '", object.name, "': Read ",
				lod.numVertAttribs, " deflated face attributes");
}

void BinaryLoader::read_compressed_face_materials(const ObjectState& object, const LodState& lod) {
	if(lod.numTriangles == 0 && lod.numQuads == 0)
		return;

	std::vector<unsigned char> matData = decompress();
	const u16* matIndices = reinterpret_cast<const u16*>(matData.data());
	BulkLoader matsBulk{ BulkType::BULK_ARRAY, {} };
	matsBulk.descriptor.bytes = reinterpret_cast<const char*>(matIndices);

	if(polygon_set_material_idx_bulk(lod.lodHdl, 0, lod.numTriangles + lod.numQuads, &matsBulk) == INVALID_SIZE)
		throw std::runtime_error("Failed to set face material for object '"
									+ std::string(object.name) + "'");
	logPedantic("[BinaryLoader::read_compressed_face_materials] Object '", object.name, "': Read ",
				lod.numTriangles + lod.numQuads, " deflated face material indices");
}

void BinaryLoader::read_compressed_sphere_attributes(const ObjectState& object, const LodState& lod) {
	if(lod.numSpheres == 0 || lod.numSphereAttribs == 0)
		return;

	if(read<u32>() != ATTRIBUTE_MAGIC)
		throw std::runtime_error("Invalid attribute magic constant (object '"
								 + std::string(object.name) + "'");
	std::vector<unsigned char> attributeData = decompress();
	const unsigned char* attributes = attributeData.data();
	BulkLoader attrBulk{ BulkType::BULK_ARRAY, {} };
	attrBulk.descriptor.bytes = reinterpret_cast<const char*>(attributes);

	for(u32 i = 0u; i < lod.numSphereAttribs; ++i) {
		read_compressed_attribute(attributes);
		auto attrHdl = spheres_request_attribute(lod.lodHdl, m_attribStateBuffer.name.c_str(),
												 m_attribStateBuffer.type);
		if(attrHdl.name == nullptr)
			throw std::runtime_error("Failed to add sphere attribute to object '" + std::string(object.name) + "'");

		if(spheres_set_attribute_bulk(lod.lodHdl, attrHdl, 0, lod.numSpheres, &attrBulk) == INVALID_SIZE)
			throw std::runtime_error("Failed to set sphere attribute data for object '" + std::string(object.name) + "'");

		attrBulk.descriptor.bytes += m_attribStateBuffer.bytes;
	}
	logPedantic("[BinaryLoader::read_compressed_sphere_attributes] Object '", object.name, "': Read ",
				lod.numSpheres, " deflated sphere material indices");
}

u32 BinaryLoader::read_lod(const ObjectState& object, u32 lod, bool asReduced) {
	auto scope = Profiler::loader().start<CpuProfileState>("BinaryLoader::read_lod");

	// Remember where we were in the file
	const std::ifstream::off_type currOffset = m_fileStream.tellg() - m_fileStart;
	// Jump to the object
	m_fileStream.seekg(object.offset + sizeof(u32), std::ifstream::beg);
	// Skip the object name + find the jump table
	m_fileStream.seekg(read<u32>() + sizeof(u32) * 9u, std::ifstream::cur);
	// Jump to the LoD
	const u32 lods = read<u32>();
	const u32 actualLod = std::min(lod, lods - 1u);
	m_fileStream.seekg(sizeof(u64) * actualLod, std::ifstream::cur);
	m_fileStream.seekg(read<u64>(), std::ifstream::beg);

	if(read<u32>() != LOD_MAGIC)
		throw std::runtime_error("Invalid LoD magic constant (object '" + std::string(object.name) + "', LoD "
								 + std::to_string(actualLod) + ")");

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
	lodState.lodHdl = object_add_lod(object.objHdl, actualLod, asReduced);

	logPedantic("[BinaryLoader::read_lod] Loading LoD ", actualLod, " for object '", object.name, "'...");

	// Reserve memory for the current LOD
	if(!polygon_reserve(lodState.lodHdl, lodState.numVertices,
						lodState.numTriangles, lodState.numQuads))
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

		logPedantic("[BinaryLoader::read_lod] Loaded LoD ", actualLod, " of object object '",
				object.name, "' with ", lodState.numVertices, " vertices, ",
				lodState.numTriangles, " triangles, ", lodState.numQuads, " quads, ",
				lodState.numSpheres, " spheres");
	}

	// Restore the file state
	m_fileStream.seekg(currOffset, std::ifstream::beg);
	return actualLod;
}

void BinaryLoader::read_object() {
	m_objects.back().name = m_namePool.insert(read<StringView>());
	logPedantic("[BinaryLoader::read_object] Loading object '", m_objects.back().name, "'");
	const auto objInfo = read<ObjectInfoNoName>();
	m_objects.back().flags = static_cast<ObjectFlags>(objInfo.flags);
	m_objects.back().keyframe = objInfo.keyframe;
	m_objects.back().animObjId = objInfo.prevAnimObjId;
	m_objects.back().aabb = objInfo.aabb;

	logPedantic("[BinaryLoader::read_object] Read object '", m_objects.back().name,
				"' with key frame ", m_objects.back().keyframe, " and animation object ID ",
				m_objects.back().animObjId);

	// Create the object for direct writing
	m_objects.back().objHdl = world_create_object(m_mffInstHdl, m_objects.back().name.data(), m_objects.back().flags);
	if(m_objects.back().objHdl == nullptr)
		throw std::runtime_error("Failed to create object '" + std::string(m_objects.back().name));

	// Reserve space for the LoDs
	const auto lodCount = read<u32>();
	object_allocate_lod_slots(m_objects.back().objHdl, lodCount);
}

void BinaryLoader::read_bone_animation_data() {
	const u32 numBones = read<u32>();
	const u32 frameCount = read<u32>();
	world_reserve_animation(m_mffInstHdl, numBones, frameCount);
	for(u32 f = 0u; f < frameCount; ++f)
		for(u32 i = 0u; i < numBones; ++i) {
			DualQuaternion t = read<DualQuaternion>();
			world_set_bone(m_mffInstHdl, i, f, &t);
		}
}

bool BinaryLoader::read_instances(const u32 globalLod,
								  const util::FixedHashMap<StringView, u32>& objectLods,
								  util::FixedHashMap<StringView, InstanceMapping>& instanceLods,
								  const bool noDefaultInstances) {
	auto scope = Profiler::loader().start<CpuProfileState>("BinaryLoader::read_instances");
	sprintf(m_loadingStage.data(), "Loading instances%c", '\0');
	std::vector<uint8_t> hasInstance(m_objects.size(), false);
	const u32 numInstances = read<u32>();
	const u32 instDispInterval = std::max(1u, numInstances / 200u);
	for(u32 i = 0u; i < numInstances; ++i) {
		if(m_abort)
			return false;
		// Only set the string every x instances to avoid overhead
		if(i % instDispInterval == 0)
			sprintf(m_loadingStage.data(), "Loading instance %u / %u%c",
					i, numInstances, '\0');
		const auto name = read<StringView>();
		const auto info = read<InstanceInfoNoName>();
		// Determine what level-of-detail should be applied for this instance
		u32 lod = globalLod;
		if(auto iter = objectLods.find(m_objects[info.objId].name); iter != objectLods.end())
			lod = iter->second;
		// TODO: this is actually a significant cost, even if few instances are in the list
		const auto instIter = instanceLods.find(name);
		if(instIter != instanceLods.end())
			lod = instIter->second.customLod;

		// Only log below a certain threshold
		logPedantic("[BinaryLoader::read_instances] Creating given instance (keyframe ", info.keyframe,
					", animInstId ", info.prevAnimInstId, ") for object '", name, "\'");
		InstanceHdl instHdl = world_create_instance(m_mffInstHdl, m_objects[info.objId].objHdl, info.keyframe);
		if(instHdl == nullptr)
			throw std::runtime_error("Failed to create instance for object ID "
									 + std::to_string(info.objId));
		if(instIter != instanceLods.end())
			instIter->second.handle = instHdl;

		// Here was the point where we previously would have loaded in the referenced LoD
		// However, since initial reduction may be needed to load in large scenes,
		// it is non-sensical to not fetch them on demand (except for negligibly small
		// scenes, where it doesn't matter anyway).


		if(!instance_set_transformation_matrix(m_mffInstHdl, instHdl, &info.transMat, m_loadWorldToInstTrans))
			throw std::runtime_error("Failed to set transformation matrix for instance of object ID "
									 + std::to_string(info.objId));
		// We manually calculate the bounding box so that we don't have to actually
		// load in the LoD, which we would if we'd use the regular interface
		Mat3x4 instToWorld;
		if(!instance_get_transformation_matrix(m_mffInstHdl, instHdl, &instToWorld))
			throw std::runtime_error("Failed to read back instance-to-world transformation matrix for instance of object ID "
									 + std::to_string(info.objId));
		const auto instanceAabb = ei::transform(m_objects[info.objId].aabb, util::pun<ei::Mat3x4>(instToWorld));
		m_aabb = ei::Box(m_aabb, instanceAabb);

		hasInstance[info.objId] = true;
	}

	logInfo("[BinaryLoader::read_instances] Loaded ", numInstances, " instances");
	if(!noDefaultInstances) {
		sprintf(m_loadingStage.data(), "Creating default instances%c", '\0');
		// Create identity instances for objects not having one yet
		const auto objectCount = hasInstance.size();
		const u32 objDispInterval = std::max(1u, static_cast<u32>(objectCount) / 10u);
		u32 defaultCreatedInstances = 0u;
		for(u32 i = 0u; i < static_cast<u32>(objectCount); ++i) {
			if(m_abort)
				return false;
			// Only set the string every x objects to avoid overhead
			if(i % objDispInterval == 0)
				sprintf(m_loadingStage.data(), "Checking default instance for object %u / %zu%c", i, objectCount, '\0');
			if(!hasInstance[i]) {
				StringView objName = m_objects[i].name;
				logPedantic("[BinaryLoader::read_instances] Creating default instance for object '",
							objName, "\'");
				// Add default instance
				m_nameBuffer.clear();
				m_nameBuffer.append(objName.data(), objName.size());
				m_nameBuffer.append("###defaultInstance");
				// Determine what level-of-detail should be applied for this instance
				u32 lod = globalLod;
				if(auto iter = objectLods.find(objName); iter != objectLods.end())
					lod = iter->second;
				InstanceHdl instHdl = world_create_instance(m_mffInstHdl, m_objects[i].objHdl, 0xFFFFFFFF);
				if(instHdl == nullptr)
					throw std::runtime_error("Failed to create instance for object ID "
											 + std::to_string(i));

				// We now have a valid instance: time to check if we have the required LoD
				if(!object_has_lod(m_objects[i].objHdl, lod)) {
					// We don't -> gotta load it
					read_lod(m_objects[i], lod);
				}

				m_aabb = ei::Box(m_aabb, m_objects[i].aabb);
				++defaultCreatedInstances;
			}
		}
		logInfo("[BinaryLoader::read_instances] Created ", defaultCreatedInstances, " default instances");
	}
	
	return true;
}

void BinaryLoader::deinstance() {
	sprintf(m_loadingStage.data(), "Performing deinstancing%c", '\0');
	// Both animated and not animated instances
	auto applyTranformation = [this](const u32 frame) {
		const u32 numInstances = world_get_instance_count(m_mffInstHdl, frame);
		for(u32 i = 0u; i < numInstances; ++i) {
			InstanceHdl hdl = world_get_instance_by_index(m_mffInstHdl, i, frame);
			world_apply_instance_transformation(m_mffInstHdl, hdl);
		}
	};

	const u32 frames = world_get_highest_instance_frame(m_mffInstHdl);
	applyTranformation(0xFFFFFFFF);
	for(u32 i = 0u; i < frames; ++i)
		applyTranformation(frames);
}

void BinaryLoader::load_lod(const fs::path& file, ObjectHdl obj, mufflon::u32 objId, mufflon::u32 lod,
							const bool asReduced) {
	auto scope = Profiler::loader().start<CpuProfileState>("BinaryLoader::load_lod");
	m_filePath = file;

	for(u32 i = 0u; i < 3u; ++i)
		m_fileDescs[i] = FileDescriptor{ m_filePath };

	if(!fs::exists(m_filePath))
		throw std::runtime_error("Binary file '" + m_filePath.u8string() + "' doesn't exist");

	logPedantic("[BinaryLoader::load_lod] Loading LoD ", lod, " for object ID ", objId,
			" from file '", m_filePath.u8string(), "'");

	try {
		// Open the binary file and enable exception management
		m_fileStream = std::ifstream(m_filePath, std::ios_base::binary);
		if(m_fileStream.bad() || m_fileStream.fail())
			throw std::runtime_error("Failed to open binary file '" + m_filePath.u8string() + "\'");
		m_fileStream.exceptions(std::ifstream::failbit);
		m_fileStart = m_fileStream.tellg();

		// Skip over the materials header
		if(read<u32>() != MATERIALS_HEADER_MAGIC)
			throw std::runtime_error("Invalid materials header magic constant");
		const u64 objectStart = read<u64>();
		m_fileStream.seekg(objectStart, std::ifstream::beg);

		// Skip over animations header (if existing)
		u32 headerMagic = read<u32>();
		if(headerMagic == BONE_ANIMATION_MAGIC) {
			const u64 objectStart = read<u64>();
			m_fileStream.seekg(objectStart, std::ifstream::beg);
			headerMagic = read<u32>();
		}

		// Parse the object header
		if(headerMagic != OBJECTS_HEADER_MAGIC)
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
		object.name = m_namePool.insert(read<StringView>());
		object.flags = static_cast<ObjectFlags>(read<u32>());
		object.keyframe = read<u32>();
		object.animObjId = read<u32>();
		object.aabb = read<ei::Box>();
		if(obj == nullptr)
			object.objHdl = world_get_object(m_mffInstHdl, object.name.data());
		else
			object.objHdl = obj;
		if(object.objHdl == nullptr)
			throw std::runtime_error("Unknown object '" + std::string(object.name) + ")");
		// Read the LoD
		if(!object_has_lod(object.objHdl, lod))
			read_lod(object, lod, asReduced);

		this->clear_state();
	} catch(const std::exception&) {
		// Clean up before leaving throwing
		this->clear_state();
		throw;
	}
}

u32 BinaryLoader::read_unique_object_material_indices(const fs::path& file, const u32 objId, u16* indices) {
	auto scope = Profiler::loader().start<CpuProfileState>("BinaryLoader::read_unique_object_material_indices");
	m_filePath = file;

	for(u32 i = 0u; i < 3u; ++i)
		m_fileDescs[i] = FileDescriptor{ m_filePath };

	if(!fs::exists(m_filePath))
		throw std::runtime_error("Binary file '" + m_filePath.u8string() + "' doesn't exist");

	try {
		// Open the binary file and enable exception management
		m_fileStream = std::ifstream(m_filePath, std::ios_base::binary);
		if(m_fileStream.bad() || m_fileStream.fail())
			throw std::runtime_error("Failed to open binary file '" + m_filePath.u8string() + "\'");
		m_fileStream.exceptions(std::ifstream::failbit);
		m_fileStart = m_fileStream.tellg();

		// Skip over the materials header
		if(read<u32>() != MATERIALS_HEADER_MAGIC)
			throw std::runtime_error("Invalid materials header magic constant");
		const u64 objectStart = read<u64>();
		m_fileStream.seekg(objectStart, std::ifstream::beg);

		// Skip over animations header (if existing)
		u32 headerMagic = read<u32>();
		if(headerMagic == BONE_ANIMATION_MAGIC) {
			const u64 objectStart = read<u64>();
			m_fileStream.seekg(objectStart, std::ifstream::beg);
			headerMagic = read<u32>();
		}

		// Parse the object header
		if(headerMagic != OBJECTS_HEADER_MAGIC)
			throw std::runtime_error("Invalid objects header magic constant");
		(void)read<u64>(); // Instance start
		GlobalFlag compressionFlags = GlobalFlag{ { read<u32>() } };

		// Jump to the desired object
		const u32 jumpCount = read<u32>();
		if(objId >= jumpCount)
			throw std::runtime_error("Object index out of bounds (" + std::to_string(objId)
									 + " >= " + std::to_string(jumpCount) + ")");
		m_fileStream.seekg(sizeof(u64) * objId, std::ifstream::cur);

		const auto offset = read<u64>();
		// Jump to the object
		m_fileStream.seekg(offset + sizeof(u32), std::ifstream::beg);
		// Skip the object name and properties
		m_fileStream.seekg(read<u32>() + sizeof(u32) * 9u, std::ifstream::cur);
		// Skip the LoD jump table
		m_fileStream.seekg(read<u32>() * sizeof(u64), std::ifstream::cur);

		// Read the material indices
		const auto numIndices = read<u32>();
		for(u32 i = 0u; i < numIndices; ++i)
			indices[i] = read<u16>();
		this->clear_state();
		return numIndices;
	} catch(const std::exception&) {
		// Clean up before leaving throwing
		this->clear_state();
		throw;
	}
	return 0u;
}

std::size_t BinaryLoader::read_lods_metadata(const fs::path& file, LodMetadata* buffer) {
	auto scope = Profiler::loader().start<CpuProfileState>("BinaryLoader::read_unique_object_material_indices");
	m_filePath = file;

	for(u32 i = 0u; i < 3u; ++i)
		m_fileDescs[i] = FileDescriptor{ m_filePath };

	if(!fs::exists(m_filePath))
		throw std::runtime_error("Binary file '" + m_filePath.u8string() + "' doesn't exist");

	try {
		// Open the binary file and enable exception management
		m_fileStream = std::ifstream(m_filePath, std::ios_base::binary);
		if(m_fileStream.bad() || m_fileStream.fail())
			throw std::runtime_error("Failed to open binary file '" + m_filePath.u8string() + "\'");
		m_fileStream.exceptions(std::ifstream::failbit);
		m_fileStart = m_fileStream.tellg();

		// Skip over the materials header
		if(read<u32>() != MATERIALS_HEADER_MAGIC)
			throw std::runtime_error("Invalid materials header magic constant");
		const u64 objectStart = read<u64>();
		m_fileStream.seekg(objectStart, std::ifstream::beg);

		// Skip over animations header (if existing)
		u32 headerMagic = read<u32>();
		if(headerMagic == BONE_ANIMATION_MAGIC) {
			const u64 objectStart = read<u64>();
			m_fileStream.seekg(objectStart, std::ifstream::beg);
			headerMagic = read<u32>();
		}

		// Parse the object header
		if(headerMagic != OBJECTS_HEADER_MAGIC)
			throw std::runtime_error("Invalid objects header magic constant");
		(void)read<u64>(); // Instance start
		GlobalFlag compressionFlags = GlobalFlag{ { read<u32>() } };

		// Jump to the desired object
		std::vector<u64> objJumpTable(read<u32>());
		for(std::size_t i = 0u; i < objJumpTable.size(); ++i)
			objJumpTable[i] = read<u64>();

		// For every object and every LoD, we read the metadata
		std::vector<u64> lodJumpTable;
		std::size_t currIndex = 0u;
		for(const auto objOffset : objJumpTable) {
			m_fileStream.seekg(objOffset, std::ifstream::beg);
			if(read<u32>() != OBJECT_MAGIC)
				throw std::runtime_error("Invalid object magic constant");
			// Skip the object name and properties
			m_fileStream.seekg(read<u32>() + sizeof(u32) * 9u, std::ifstream::cur);
			// Read the LoD jump table
			lodJumpTable.resize(read<u32>());
			for(std::size_t i = 0u; i < lodJumpTable.size(); ++i)
				lodJumpTable[i] = read<u64>();

			// Iterate every LoD
			for(const auto lodOffset : lodJumpTable) {
				m_fileStream.seekg(lodOffset, std::ifstream::beg);
				if(read<u32>() != LOD_MAGIC)
					throw std::runtime_error("Invalid LoD magic constant");
				// Read the LoD information
				LodMetadata& data = buffer[currIndex++];
				data.triangles = read<u32>();
				data.quads = read<u32>();
				data.spheres = read<u32>();
				data.vertices = read<u32>();
				data.edges = read<u32>();
			}
		}
		return currIndex;
	} catch(const std::exception&) {
		// Clean up before leaving throwing
		this->clear_state();
		throw;
	}
}

bool BinaryLoader::load_file(fs::path file, const u32 globalLod,
							 const util::FixedHashMap<StringView, mufflon::u32>& objectLods,
							 util::FixedHashMap<StringView, InstanceMapping>& instanceLods,
							 const bool deinstance, const bool loadWorldToInstTrans,
							 const bool noDefaultInstances) {
	auto scope = Profiler::loader().start<CpuProfileState>("BinaryLoader::load_file");
	m_loadWorldToInstTrans = loadWorldToInstTrans;
	m_filePath = std::move(file);
	if(!fs::exists(m_filePath))
		throw std::runtime_error("Binary file '" + m_filePath.u8string() + "' doesn't exist");
	m_aabb.min = ei::Vec3{ 1e30f };
	m_aabb.max = ei::Vec3{ -1e30f };
	sprintf(m_loadingStage.data(), "Loading binary file%c", '\0');
	logInfo("[BinaryLoader::load_file] Loading binary file '", m_filePath.u8string(), "'");
	try {
		// Open the binary file and enable exception management
		m_fileStream = std::ifstream(m_filePath, std::ios_base::binary);
		if(m_fileStream.bad() || m_fileStream.fail())
			throw std::runtime_error("Failed to open binary file '" + m_filePath.u8string() + "\'");
		m_fileStream.exceptions(std::ifstream::failbit);
		// Needed to get a C file descriptor offset
		m_fileStart = m_fileStream.tellg();
		for(u32 i = 0u; i < 3u; ++i)
			m_fileDescs[i] = FileDescriptor{ m_filePath };

		if(m_abort)
			return false;

		// Read the materials header
		if(read<u32>() != MATERIALS_HEADER_MAGIC)
			throw std::runtime_error("Invalid materials header magic constant");
		const u64 nextStart = read<u64>();
		const u32 numMaterials = read<u32>();
		// Read the material names (and implicitly their indices)
		m_materialNames.reserve(numMaterials);
		for(u32 i = 0u; i < numMaterials; ++i) {
			m_materialNames.push_back(read<std::string>());
		}
		if(m_abort)
			return false;

		// Jump to the location of the next section
		m_fileStream.seekg(nextStart, std::ifstream::beg);

		// Parse bone data if excistent
		u32 headerMagic = read<u32>();
		if(headerMagic == BONE_ANIMATION_MAGIC) {
			const u64 objectStart = read<u64>();
			read_bone_animation_data();
			m_fileStream.seekg(objectStart, std::ifstream::beg);
			headerMagic = read<u32>();
		}

		// Parse the object header
		if(headerMagic != OBJECTS_HEADER_MAGIC)
			throw std::runtime_error("Invalid objects header magic constant");
		const u64 instanceStart = read<u64>();
		GlobalFlag compressionFlags = GlobalFlag{ { read<u32>() } };

		// Parse the object jumptable
		m_objJumpTable.resize(read<u32>());
		for(std::size_t i = 0u; i < m_objJumpTable.size(); ++i)
			m_objJumpTable[i] = read<u64>();
		{	// Quickly jump to the instance section to peek at their count
			const auto currPos = m_fileStream.tellg();
			m_fileStream.seekg(instanceStart + sizeof(u32), std::ifstream::beg);
			const auto instanceCount = read<u32>();
			m_fileStream.seekg(currPos, std::ifstream::beg);
			// Since there may be implicit (default) instances, add object count to be sure
			// If we're using deinstancing, the number of objects increases by the number of instances
			// (conservative estimate)
			world_reserve_objects_instances(m_mffInstHdl, static_cast<u32>(m_objJumpTable.size() + (deinstance ? instanceCount : 0u)),
											static_cast<u32>(instanceCount + m_objJumpTable.size()));
		}

		const auto objectCount = m_objJumpTable.size();
		const std::size_t objDispInterval = std::max<std::size_t>(1u, objectCount / 200u);
		// Next come the objects
		for(std::size_t i = 0u; i < objectCount; ++i) {
			if(m_abort)
				return false;
			if(i % objDispInterval == 0u)
				sprintf(m_loadingStage.data(), "Reading object definition %zu / %zu%c",
						i, objectCount, '\0');

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
		logInfo("[BinaryLoader::load_file] Parsed ", objectCount, " objects");

		// Now come instances
		m_fileStream.seekg(instanceStart, std::ios_base::beg);
		if(read<u32>() != INSTANCE_MAGIC)
			throw std::runtime_error("Invalid instance magic constant");
		if(!read_instances(globalLod, objectLods, instanceLods, noDefaultInstances))
			return false;
		if(deinstance)
			this->deinstance();
		this->clear_state();
		for(u32 i = 0u; i < 3u; ++i)
			m_fileDescs[i].close();
	} catch(const std::exception&) {
		// Clean up before leaving throwing
		this->clear_state();
		throw;
	}
	return true;
}

} // namespace mff_loader::binary
