#pragma once

#include "core/scene/attributes/attribute.hpp"
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

	SphereAttributeHandle add_attribute(StringView name, AttributeType type) {
		return m_attributes.add_attribute(AttributeIdentifier{ type, name });
	}
	template < class T >
	SphereAttributeHandle add_attribute(StringView name) {
		return m_attributes.add_attribute(AttributeIdentifier{ get_attribute_type<T>(), name });
	}

	std::optional<SphereAttributeHandle> find_attribute(StringView name, const AttributeType& type) const {
		const AttributeIdentifier ident{ type, name };
		return m_attributes.find_attribute(ident);
	}

	template < class T >
	void remove_attribute(StringView name) {
		const AttributeIdentifier identifier{ get_attribute_type<T>(), name };
		if(const auto handle = m_attributes.find_attribute(identifier); handle.has_value())
			m_attributes.remove(handle.value());
	}

	// Synchronizes the default attributes position, normal, uv, matindex 
	template < Device dev >
	void synchronize() {
		m_attributes.synchronize<dev>();
	}
	/*template < Device dev >
	void synchronize(StringView name) {
		m_attributes.synchronize<dev>(name);
	}
	template < Device dev >
	void synchronize(SphereAttributeHandle hdl) {
		m_attributes.synchronize<dev>(hdl);
	}*/

	template < Device dev >
	void unload() {
		m_attributes.unload<dev>();
	}

	template < Device dev, class T >
	ArrayDevHandle_t<dev, T> acquire(const SphereAttributeHandle& hdl) {
		return m_attributes.template acquire<dev, T>(hdl);
	}
	template < Device dev, class T >
	ConstArrayDevHandle_t<dev, T> acquire_const(const SphereAttributeHandle& hdl) {
		return m_attributes.template acquire_const<dev, T>(hdl);
	}
	template < Device dev, class T >
	ArrayDevHandle_t<dev, T> acquire(const AttributeIdentifier& ident) {
		if(const auto handle = m_attributes.find_attribute(ident); handle.has_value())
			return this->template acquire<dev, T>(handle.value());
		return {};
	}
	template < Device dev, class T >
	ConstArrayDevHandle_t<dev, T> acquire_const(const AttributeIdentifier& ident) {
		if(const auto handle = m_attributes.find_attribute(ident); handle.has_value())
			return this->template acquire_const<dev, T>(handle.value());
		return {};
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
									 const std::vector<AttributeIdentifier>& attribs);

	const ei::Box& get_bounding_box() const noexcept {
		return m_boundingBox;
	}

	std::size_t get_sphere_count() const noexcept {
		return m_attributes.get_attribute_elem_count();
	}

	bool was_displacement_mapping_applied() const noexcept {
		return true;
	}

	void displace(tessellation::TessLevelOracle& oracle, const Scenario& scenario);
	void tessellate(tessellation::TessLevelOracle& oracle, const Scenario* scenario,
					const bool usePhong);
	bool apply_animation(u32 frame, const Bone* bones);

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
};

}}} // namespace mufflon::scene::geometry
