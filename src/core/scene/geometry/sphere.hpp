#pragma once

#include "core/scene/attribute.hpp"
#include "core/scene/types.hpp"
#include "core/scene/tessellation/tessellater.hpp"
#include <ei/3dtypes.hpp>
#include <ei/vector.hpp>
#include <tuple>
#include <vector>
#include <unordered_set>

namespace mufflon {
enum class Device : unsigned char;
namespace util {
class IByteReader;
} // namespace util
} // namespace mufflon

namespace mufflon { namespace scene {

template < Device dev >
struct SpheresDescriptor;

class Scenario;

namespace tessellation {
class TessLevelOracle;
} // namespace tessellation

namespace geometry {

/**
 * Instantiation of geometry class.
 * Can store spheres only.
 */
class Spheres {
public:
	// Basic type definitions
	using Index = u32;
	using SphereHandle = std::size_t;

	// Struct communicating the number of bulk-read spheres
	struct BulkReturn {
		SphereHandle handle;
		std::size_t readSpheres;
	};

	// Associates an attribute name with a type
	template < class T >
	struct AttrDesc {
		using Type = T;
		std::string name;
	};

	// Default construction, creates material-index attribute.
	Spheres();
	Spheres(const Spheres&);
	Spheres(Spheres&&);
	Spheres& operator=(const Spheres&) = delete;
	Spheres& operator=(Spheres&&) = delete;
	~Spheres();

	void reserve(std::size_t count) {
		m_attributes.reserve(count);
	}

	template < class T >
	SphereAttributeHandle add_attribute(std::string name) {
		return m_attributes.add_attribute<T>(std::move(name));
	}

	void remove_attribute(StringView name) {
		throw std::runtime_error("Operation not implemented yet");
	}

	// Synchronizes the default attributes position, normal, uv, matindex 
	template < Device dev >
	void synchronize() {
		m_attributes.synchronize<dev>();
	}
	template < Device dev >
	void synchronize(StringView name) {
		m_attributes.synchronize<dev>(name);
	}
	template < Device dev >
	void synchronize(SphereAttributeHandle hdl) {
		m_attributes.synchronize<dev>(hdl);
	}

	template < Device dev >
	void unload() {
		m_attributes.unload<dev>();
	}

	template < Device dev, class T >
	ArrayDevHandle_t<dev, T> acquire(SphereAttributeHandle hdl) {
		return m_attributes.acquire<dev, T>(hdl);
	}
	template < Device dev, class T >
	ConstArrayDevHandle_t<dev, T> acquire_const(SphereAttributeHandle hdl) {
		return m_attributes.acquire_const<dev, T>(hdl);
	}
	template < Device dev, class T >
	ArrayDevHandle_t<dev, T> acquire(StringView name) {
		return m_attributes.acquire<dev, T>(name);
	}
	template < Device dev, class T >
	ConstArrayDevHandle_t<dev, T> acquire_const(StringView name) {
		return m_attributes.acquire_const<dev, T>(name);
	}

	void mark_changed(Device dev) {
		m_attributes.mark_changed(dev);
	}

	// Adds a sphere.
	SphereHandle add(const Point& point, float radius);
	SphereHandle add(const Point& point, float radius, MaterialIndex idx);
	/**
	 * Adds a bulk of spheres.
	 * Returns both a handle to the first added sphere as well as the number of
	 * read spheres.
	 */
	BulkReturn add_bulk(std::size_t count, util::IByteReader& radPosStream);
	BulkReturn add_bulk(std::size_t count, util::IByteReader& radPosStream,
						const ei::Box& boundingBox);
	/**
	 * Bulk-loads the given attribute starting at the given sphere.
	 * The number of read values will be capped by the number of spheres present
	 * after the starting position.
	 */
	std::size_t add_bulk(StringView name, const SphereHandle& startSphere,
						 std::size_t count, util::IByteReader& attrStream);
	std::size_t add_bulk(SphereAttributeHandle hdl, const SphereHandle& startSphere,
						 std::size_t count, util::IByteReader& attrStream);

	SphereAttributeHandle get_spheres_hdl() const noexcept {
		return m_spheresHdl;
	}
	SphereAttributeHandle get_material_indices_hdl() const noexcept {
		return m_matIndicesHdl;
	}
	// Transforms sphere data
	void transform(const ei::Mat3x4& transMat);

	// Gets the descriptor with only default attributes (position etc)
	template < Device dev >
	SpheresDescriptor<dev> get_descriptor();
	// Updates the descriptor with the given set of attributes
	template < Device dev >
	void update_attribute_descriptor(SpheresDescriptor<dev>& descriptor,
									 const std::vector<const char*>& attribs);

	const ei::Box& get_bounding_box() const noexcept {
		return m_boundingBox;
	}

	std::size_t get_sphere_count() const noexcept {
		return m_attributes.get_attribute_elem_count();
	}

	// Get a list of all materials which are referenced by any primitive
	const std::unordered_set<MaterialIndex>& get_unique_materials() const {
		return m_uniqueMaterials;
	}

	// Returns whether any polygon has a displacement map associated with the given material assignment
	bool has_displacement_mapping(const Scenario& scenario) const noexcept {
		return false;
	}

	bool was_displacement_mapping_applied() const noexcept {
		return true;
	}

	void displace(tessellation::TessLevelOracle& oracle, const Scenario& scenario);
	void tessellate(tessellation::TessLevelOracle& oracle, const Scenario* scenario,
					const bool usePhong);

private:
	template < Device dev >
	struct AttribBuffer {
		static constexpr Device DEVICE = dev;
		ArrayDevHandle_t<dev, ArrayDevHandle_t<dev, void>> buffer;
		std::size_t size = 0u;
	};

	using AttribBuffers = util::TaggedTuple<AttribBuffer<
		Device::CPU>,
		AttribBuffer<Device::CUDA>,
		AttribBuffer<Device::OPENGL>>;

	// Make sure that spheres are tightly packed
	static_assert(sizeof(ei::Sphere) == 4u * sizeof(float));

	AttributePool m_attributes;
	SphereAttributeHandle m_spheresHdl;
	SphereAttributeHandle m_matIndicesHdl;
	// Array for aquired attribute descriptors
	AttribBuffers m_attribBuffer;
	ei::Box m_boundingBox;
	// Whenever a primitive is added the table of all referenced
	// materials will be updated. Assumption: a material reference
	// will not change afterwards.
	std::unordered_set<MaterialIndex> m_uniqueMaterials;
};

}}} // namespace mufflon::scene::geometry
