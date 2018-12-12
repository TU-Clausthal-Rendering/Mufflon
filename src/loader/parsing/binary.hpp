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
	BinaryLoader(fs::path filePath) :
		m_filePath(std::move(filePath))
	{
		if(!fs::exists(m_filePath))
			throw std::runtime_error("JSON file '" + m_filePath.string() + "' doesn't exist");
		m_aabb.min = ei::Vec3{ 1e30f};
		m_aabb.max = ei::Vec3{-1e30f};
	}

	void load_file(const mufflon::u64 globalLod,
				   const std::unordered_map<std::string_view, mufflon::u64>& localLods);

	const std::vector<std::string>& get_material_names() const noexcept {
		return m_materialNames;
	}

	const ei::Box& get_bounding_box() const noexcept {
		return m_aabb;
	}

private:
	static constexpr mufflon::u32 MATERIALS_HEADER_MAGIC = 'M' | ('a' << 8u) | ('t' << 16u) | ('s' << 24u);
	static constexpr mufflon::u32 OBJECTS_HEADER_MAGIC = 'O' | ('b' << 8u) | ('j' << 16u) | ('s' << 24u);
	static constexpr mufflon::u32 OBJECT_MAGIC = 'O' | ('b' << 8u) | ('j' << 16u) | ('_' << 24u);
	static constexpr mufflon::u32 INSTANCE_MAGIC = 'I' | ('n' << 8u) | ('s' << 16u) | ('t' << 24u);
	static constexpr mufflon::u32 LOD_MAGIC = 'L' | ('O' << 8u) | ('D' << 16u) | ('_' << 24u);
	static constexpr mufflon::u32 ATTRIBUTE_MAGIC = 'A' | ('t' << 8u) | ('t' << 16u) | ('r' << 24u);

	struct GlobalFlag : public mufflon::util::Flags<mufflon::u32> {
		static constexpr mufflon::u32 NONE = 0;
		static constexpr mufflon::u32 DEFLATE = 1;
		static constexpr mufflon::u32 COMPRESSED_NORMALS = 2;
	};
	// Per-object flags
	/*struct ObjectFlag : public mufflon::util::Flags<mufflon::u32> {
		static constexpr mufflon::u32 NONE = 0;
		static constexpr mufflon::u32 EMISSIVE = 1;		// At least one emitting primitive
	};*/

	struct ObjectState {
		std::string name;
		GlobalFlag globalFlags;
		ObjectFlags flags;
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
		m_fileStream.read(reinterpret_cast<char*>(&val), sizeof(T));
		return val;
	}

	// Reads a value from a character stream
	template < class T >
	T read(const unsigned char*& data) {
		T val = *reinterpret_cast<const T*>(data);
		data += sizeof(T);
		return val;
	}

	// Cleans up the internal data structures
	void clear_state();

	static AttribDesc map_bin_attrib_type(AttribType type);
	// Uncompressed data
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
	// Deflated data
	std::vector<unsigned char> decompress();
	AttribState read_compressed_attribute(const unsigned char*& data);
	void read_compressed_normal_compressed_vertices();
	void read_compressed_normal_uncompressed_vertices();
	void read_compressed_triangles();
	void read_compressed_quads();
	void read_compressed_spheres();
	void read_compressed_vertex_attributes();
	void read_compressed_face_attributes();
	void read_compressed_face_materials();
	void read_compressed_sphere_attributes(const unsigned char* data);

	void read_instances();
	void read_object(const mufflon::u64 globalLod,
					 const std::unordered_map<std::string_view, mufflon::u64>& localLods);
	void read_lod();

	const fs::path m_filePath;
	// Parser state
	std::ifstream m_fileStream;
	std::ifstream::pos_type m_fileStart;
	ObjectState m_currObjState;
	// Parsed data
	// The material names are wrapped in [mat:...] for ease of use in the JSON parser
	std::vector<std::string> m_materialNames;
	std::vector<ObjectHdl> m_objectHandles;
	ei::Box m_aabb;
};

} // namespace loader::binary
