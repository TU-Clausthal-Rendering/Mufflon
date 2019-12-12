#pragma once

#include "util/filesystem.hpp"
#include "util/flag.hpp"
#include "util/int_types.hpp"
#include "util/fixed_hashmap.hpp"
#include "core_interface.h"
#include <ei/3dtypes.hpp>
#include <atomic>
#include <fstream>
#include <string>
#include "util/string_pool.hpp"
#include "util/string_view.hpp"
#include <vector>

namespace mff_loader::binary {

struct InstanceMapping {
	mufflon::u32 customLod;
	InstanceHdl handle;
};

class BinaryLoader {
public:
	BinaryLoader(std::string& stage) :
		m_loadingStage{ stage }
	{}

	/* Loads the specified file with the given global LoD level and optionally
	 * specific LoDs for given objects (by name). Returns false if loading was
	 * unsuccessful. May throw.
	 * After a successful load, properties of the loaded file may be queried via
	 * this object; calling this function again will overwrite them, however.
	 */
	bool load_file(fs::path file, const mufflon::u32 globalLod,
				   const mufflon::util::FixedHashMap<mufflon::StringView, mufflon::u32>& objectLods,
				   mufflon::util::FixedHashMap<mufflon::StringView, InstanceMapping>& instanceLods,
				   const bool deinstance, const bool loadWorldToInstTrans,
				   const bool keepTrackOfAabb, const bool noDefaultInstances);

	void load_lod(const fs::path& file, mufflon::u32 objId, mufflon::u32 lod);

	const std::vector<std::string>& get_material_names() const noexcept {
		return m_materialNames;
	}

	const ei::Box& get_bounding_box() const noexcept {
		return m_aabb;
	}

	// This may be called from a different thread and leads to the current load being cancelled
	void abort_load() { m_abort = true; }
	bool was_aborted() { return m_abort; }

private:
	static constexpr mufflon::u32 MATERIALS_HEADER_MAGIC = 'M' | ('a' << 8u) | ('t' << 16u) | ('s' << 24u);
	static constexpr mufflon::u32 OBJECTS_HEADER_MAGIC = 'O' | ('b' << 8u) | ('j' << 16u) | ('s' << 24u);
	static constexpr mufflon::u32 OBJECT_MAGIC = 'O' | ('b' << 8u) | ('j' << 16u) | ('_' << 24u);
	static constexpr mufflon::u32 INSTANCE_MAGIC = 'I' | ('n' << 8u) | ('s' << 16u) | ('t' << 24u);
	static constexpr mufflon::u32 LOD_MAGIC = 'L' | ('O' << 8u) | ('D' << 16u) | ('_' << 24u);
	static constexpr mufflon::u32 ATTRIBUTE_MAGIC = 'A' | ('t' << 8u) | ('t' << 16u) | ('r' << 24u);

	// RAII wrapper around C file descriptor
	class FileDescriptor {
	public:
		FileDescriptor() = default;
		FileDescriptor(fs::path file, const char* mode);
		FileDescriptor(const FileDescriptor&) = delete;
		FileDescriptor(FileDescriptor&&);
		FileDescriptor& operator=(const FileDescriptor&) = delete;
		FileDescriptor& operator=(FileDescriptor&&);
		~FileDescriptor();
		void close() noexcept;
		FILE* get() const noexcept { return m_desc; }
		// Advances a C file descriptor (possibly in multiple steps)
		FileDescriptor& seek(mufflon::u64 offset, std::ios_base::seekdir dir = std::ios_base::cur);

	private:
		FILE* m_desc = nullptr;
	};

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
		mufflon::StringView name;
		GlobalFlag globalFlags;
		ObjectFlags flags;
		ei::Box aabb;
		mufflon::u32 keyframe;
		mufflon::u32 animObjId;
		ObjectHdl objHdl;
		mufflon::u64 offset;
	};

	struct LodState {
		mufflon::u32 numTriangles;
		mufflon::u32 numQuads;
		mufflon::u32 numSpheres;
		mufflon::u32 numVertices;
		mufflon::u32 numEdges;
		mufflon::u32 numVertAttribs;
		mufflon::u32 numFaceAttribs;
		mufflon::u32 numSphereAttribs;
		LodHdl lodHdl;
	};

	struct AttribState {
		std::string name;
		std::string meta;
		mufflon::u32 metaFlags;
		GeomAttributeType type;
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
	void read(std::string& str) {
		str.clear();
		str.resize(read<mufflon::u32>());
		m_fileStream.read(str.data(), str.length());
	}
	void read(std::string& str, const unsigned char*& data) {
		str.clear();
		mufflon::u32 size = read<mufflon::u32>(data);
		str.resize(size);
		for(mufflon::u32 i = 0u; i < size; ++i)
			str[i] = read<char>(data);
	}

	// Cleans up the internal data structures
	void clear_state();

	static GeomAttributeType map_bin_attrib_type(AttribType type);
	// Uncompressed data
	void read_uncompressed_attribute();
	void read_normal_compressed_vertices(const ObjectState& object, const LodState& lod);
	void read_normal_uncompressed_vertices(const ObjectState& object, const LodState& lod);
	void read_uncompressed_triangles(const ObjectState& object, const LodState& lod);
	void read_uncompressed_quads(const ObjectState& object, const LodState& lod);
	void read_uncompressed_spheres(const ObjectState& object, const LodState& lod);
	void read_uncompressed_vertex_attributes(const ObjectState& object, const LodState& lod);
	void read_uncompressed_face_attributes(const ObjectState& object, const LodState& lod);
	void read_uncompressed_sphere_attributes(const ObjectState& object, const LodState& lod);
	void read_uncompressed_face_materials(const ObjectState& object, const LodState& lod);
	void read_uncompressed_sphere_materials(const ObjectState& object, const LodState& lod);
	// Deflated data
	std::vector<unsigned char> decompress();
	void read_compressed_attribute(const unsigned char*& data);
	void read_compressed_normal_compressed_vertices(const ObjectState& object, const LodState& lod);
	void read_compressed_normal_uncompressed_vertices(const ObjectState& object, const LodState& lod);
	void read_compressed_triangles(const ObjectState& object, const LodState& lod);
	void read_compressed_quads(const ObjectState& object, const LodState& lod);
	void read_compressed_spheres(const ObjectState& object, const LodState& lod);
	void read_compressed_vertex_attributes(const ObjectState& object, const LodState& lod);
	void read_compressed_face_attributes(const ObjectState& object, const LodState& lod);
	void read_compressed_face_materials(const ObjectState& object, const LodState& lod);
	void read_compressed_sphere_attributes(const ObjectState& object, const LodState& lod);

	bool read_instances(const mufflon::u32 globalLod,
						const mufflon::util::FixedHashMap<mufflon::StringView, mufflon::u32>& objectLods,
						mufflon::util::FixedHashMap<mufflon::StringView, InstanceMapping>& instanceLods,
						const bool noDefaultInstances);
	void deinstance();
	void read_object();
	mufflon::u32 read_lod(const ObjectState& object, mufflon::u32 lod);

	fs::path m_filePath;
	// Parser state
	std::ifstream m_fileStream;
	std::ifstream::pos_type m_fileStart;
	mufflon::util::StringPool m_namePool;
	AttribState m_attribStateBuffer;
	std::string m_nameBuffer;
	// We need at most three file descriptors at any time, so we keep them around
	FileDescriptor m_fileDescs[3u];
	// Parsed data
	// The material names are wrapped in [mat:...] for ease of use in the JSON parser
	std::vector<std::string> m_materialNames;
	std::vector<ObjectState> m_objects;
	std::vector<mufflon::u64> m_objJumpTable;
	ei::Box m_aabb;

	// These are for aborting a load and keeping track of progress
	bool m_loadWorldToInstTrans = false;
	bool m_keepTrackOfAabb = true;
	std::atomic_bool m_abort = false;
	std::string& m_loadingStage;
};

} // namespace mff_loader::binary
