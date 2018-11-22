#pragma once

#include "util/filesystem.hpp"
#include "util/flag.hpp"
#include "util/int_types.hpp"
#include "core/export/interface.h"
#include <ei/3dtypes.hpp>
#include <fstream>
#include <string>
#include <string_view>
#include <vector>
#include <unordered_map>

namespace loader::binary {

class BinaryLoader {
public:
	static constexpr mufflon::u32 MATERIALS_HEADER_MAGIC = ('M' << 24u) | ('a' << 16u) | ('t' << 8u) | 's';
	static constexpr mufflon::u32 OBJECTS_HEADER_MAGIC = ('O' << 24u) | ('b' << 16u) | ('j' << 8u) | 's';
	static constexpr mufflon::u32 OBJECT_MAGIC = ('O' << 24u) | ('b' << 16u) | ('j' << 8u) | '_';
	static constexpr mufflon::u32 INSTANCE_MAGIC = ('I' << 24u) | ('n' << 16u) | ('s' << 8u) | 't';
	static constexpr mufflon::u32 LOD_MAGIC = ('L' << 24u) | ('O' << 16u) | ('D' << 8u) | '_';
	static constexpr mufflon::u32 ATTRIBUTE_MAGIC = ('A' << 24u) | ('t' << 16u) | ('t' << 8u) | 'r';

	BinaryLoader(fs::path filePath) :
		m_filePath(std::move(filePath))
	{
		if(!fs::exists(m_filePath))
			throw std::runtime_error("JSON file '" + m_filePath.string() + "' doesn't exist");
	}

	void load_file(const mufflon::u64 globalLod,
				   const std::unordered_map<std::string_view, mufflon::u64>& localLods);
	void clear_state();

private:
	// Per-object flags
	struct ObjectFlag : public mufflon::util::Flags<mufflon::u32> {
		static constexpr mufflon::u32 NONE = 0;
		static constexpr mufflon::u32 DEFLATE = 1;
		static constexpr mufflon::u32 COMPRESSED_NORMALS = 2;
	};

	struct ObjectState {
		std::string name;
		ObjectFlag flags;
		mufflon::u64 lodLevel;
		ei::Box aabb;
		mufflon::u32 keyframe;
		mufflon::u32 animObjId;
		mufflon::u32 numTriangles;
		mufflon::u32 numQuads;
		mufflon::u32 numSpheres;
		mufflon::u32 numVertices;
		mufflon::u32 numEdges;
		mufflon::u32 numVertAttribs;
		mufflon::u32 numFaceAttribs;
		mufflon::u32 numSphereAttribs;
		ObjectHdl objHdl;
	};

	struct AttribState {
		std::string name;
		std::string meta;
		mufflon::u32 metaFlags;
		AttribDesc type;
		mufflon::u64 bytes;
	};

	enum class AttribType {
		CHAR,
		UCHAR,
		SHORT,
		USHORT,
		INT,
		UINT,
		LONG,
		ULONG,
		FLOAT,
		DOUBLE,
		UCHAR2,
		UCHAR3,
		UCHAR4,
		INT2,
		INT3,
		INT4,
		FLOAT2,
		FLOAT3,
		FLOAT4
	};

	// Reads a typed value from the stream without having to previously declare it
	template < class T >
	T read() {
		T val;
		m_fileStream >> val;
		return val;
	}

	static AttribDesc map_bin_attrib_type(AttribType type);
	AttribState read_uncompressed_attribute();
	void read_normal_compressed_vertices();
	void read_normal_uncompressed_vertices();
	void read_uncompressed_triangles();
	void read_uncompressed_quads();
	void read_uncompressed_spheres();
	void read_uncompressed_vertex_attributes();
	void read_uncompressed_face_attributes();
	void read_uncompressed_sphere_attributes();
	void read_uncompressed_face_materials();
	void read_uncompressed_sphere_materials();

	void read_instances();
	void read_object(const mufflon::u64 globalLod,
					 const std::unordered_map<std::string_view, mufflon::u64> localLods);
	void read_lod();

	const fs::path m_filePath;
	// Parser state
	std::ifstream m_fileStream;
	std::ifstream::pos_type m_fileStart;
	ObjectState m_currObjState;
	// Parsed data
	std::vector<std::string> m_materialNames;
	std::vector<ObjectHdl> m_objectHandles;
};

} // namespace loader::binary
