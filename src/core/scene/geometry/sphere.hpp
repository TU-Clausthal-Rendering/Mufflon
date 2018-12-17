#pragma once

#include "core/scene/attribute.hpp"
#include "core/scene/types.hpp"
#include <ei/3dtypes.hpp>
#include <ei/vector.hpp>
#include <tuple>
#include <vector>

namespace mufflon {
enum class Device : unsigned char;
namespace util {
class IByteReader;
} // namespace util
} // namespace mufflon

namespace mufflon { namespace scene {

template < Device dev >
struct SpheresDescriptor;

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
	Spheres(const Spheres&) = delete;
	Spheres(Spheres&&);
	Spheres& operator=(const Spheres&) = delete;
	Spheres& operator=(Spheres&&) = delete;
	~Spheres();

	void reserve(std::size_t count) {
		m_attributes.reserve(count);
	}

	template < class T >
	AttributePool::AttributeHandle add_attribute(std::string name) {
		return m_attributes.add_attribute<T>(std::move(name));
	}

	void remove_attribute(std::string_view name) {
		throw std::runtime_error("Operation not implemented yet");
	}

	// Synchronizes the default attributes position, normal, uv, matindex 
	template < Device dev >
	void synchronize() {
		m_attributes.synchronize<dev>();
	}
	template < Device dev >
	void synchronize(std::string_view name) {
		m_attributes.synchronize<dev>(name);
	}
	template < Device dev >
	void synchronize(AttributePool::AttributeHandle hdl) {
		m_attributes.synchronize<dev>(hdl);
	}

	template < Device dev >
	void unload() {
		m_attributes.unload<dev>();
	}

	template < Device dev, class T >
	T* acquire(typename AttributePool::AttributeHandle hdl) {
		return m_attributes.acquire<dev, T>(hdl);
	}
	template < Device dev, class T >
	const T* acquire_const(typename AttributePool::AttributeHandle hdl) {
		return m_attributes.acquire_const<dev, T>(hdl);
	}
	template < Device dev, class T >
	T* acquire(std::string_view name) {
		return m_attributes.acquire<dev, T>(name);
	}
	template < Device dev, class T >
	const T* acquire_const(std::string_view name) {
		return m_attributes.acquire_const<dev, T>(name);
	}

	void mark_changed(Device dev) {
		m_attributes.mark_changed(dev);
	}
	void mark_changed(Device dev, AttributePool::AttributeHandle hdl) {
		m_attributes.mark_changed(dev, hdl);
	}
	void mark_changed(Device dev, std::string_view name) {
		m_attributes.mark_changed(dev, name);
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
	std::size_t add_bulk(std::string_view name, const SphereHandle& startSphere,
						 std::size_t count, util::IByteReader& attrStream);
	std::size_t add_bulk(AttributePool::AttributeHandle hdl, const SphereHandle& startSphere,
						 std::size_t count, util::IByteReader& attrStream);

	AttributePool::AttributeHandle get_spheres_hdl() const noexcept {
		return m_spheresHdl;
	}
	AttributePool::AttributeHandle get_material_indices_hdl() const noexcept {
		return m_matIndicesHdl;
	}

	/**
	 * Returns a descriptor (on CPU side) with pointers to resources (on Device side).
	 * Takes two tuples: they must each contain the name and type of attributes which the
	 * renderer wants to have access to. If an attribute gets written to, it is the
	 * renderer's task to aquire it once more after that, since we cannot hand out
	 * Accessors to the concrete device.
	 */
	template < Device dev >
	SpheresDescriptor<dev> get_descriptor(const std::vector<const char*>& attribs);

	const ei::Box& get_bounding_box() const noexcept {
		return m_boundingBox;
	}

	std::size_t get_sphere_count() const noexcept {
		return m_attributes.get_attribute_elem_count();
	}

	std::size_t get_attribute_count() const noexcept {
		return m_attributes.get_attribute_count();
	}

private:
	template < Device dev >
	struct AttribBuffer {
		static constexpr Device DEVICE = dev;
		ArrayDevHandle_t<dev, ArrayDevHandle_t<dev, void>> buffer;
		std::size_t size = 0u;
	};

	using AttribBuffers = util::TaggedTuple<AttribBuffer<Device::CPU>,
		AttribBuffer<Device::CUDA>>;

	// Make sure that spheres are tightly packed
	static_assert(sizeof(ei::Sphere) == 4u * sizeof(float));

	AttributePool m_attributes;
	AttributePool::AttributeHandle m_spheresHdl;
	AttributePool::AttributeHandle m_matIndicesHdl;
	// Array for aquired attribute descriptors
	AttribBuffers m_attribBuffer;
	ei::Box m_boundingBox;
};

}}} // namespace mufflon::scene::geometry
