#pragma once

#include "core/scene/descriptors.hpp"
#include "core/scene/types.hpp"
#include "core/scene/attribute_list.hpp"
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

namespace mufflon::scene::geometry {

/**
 * Instantiation of geometry class.
 * Can store spheres only.
 */
class Spheres {
public:
	// Basic type definitions
	using Index = u32;
	using SphereHandle = std::size_t;
	template < class T >
	using AttributeHandle = typename AttributeList::template AttributeHandle<T>;
	template < class T >
	using Attribute = typename AttributeList::template BaseAttribute<T>;

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

	void resize(std::size_t count) {
		m_attributes.resize(count);
	}

	// Requests a new per-sphere attribute.
	template < class T >
	AttributeHandle<T> request(const std::string& name) {
		return m_attributes.add<T>(name);
	}

	// Removes a per-sphere attribute.
	template < class T >
	void remove(const AttributeHandle<T> &attr) {
		m_attributes.remove<T>(attr);
	}

	// Finds a per-sphere attribute by name.
	template < class T >
	std::optional<AttributeHandle<T>> find(const std::string& name) {
		return m_attributes.find<T>(name);
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
	template < class T >
	std::size_t add_bulk(Attribute<T>& attribute, const SphereHandle& startSphere,
						 std::size_t count, util::IByteReader& attrStream) {
		std::size_t start = startSphere;
		if(start >= m_attributes.get_size())
			return 0u;
		if(start + count > m_attributes.get_size())
			m_attributes.resize(start + count);
		return attribute.restore(attrStream, start, count);
	}
	// Also performs bulk-load for an attribute, but aquires it first.
	template < class Type >
	std::size_t add_bulk(const AttributeHandle<Type>& attrHandle,
						 const SphereHandle& startSphere, std::size_t count,
						 util::IByteReader& attrStream) {
		return this->add_bulk(m_attributes.aquire(attrHandle), startSphere, count, attrStream);
	}

	template < class T >
	Attribute<T>& aquire(const AttributeHandle<T>& attrHandle) {
		return m_attributes.aquire(attrHandle);
	}

	template < class T >
	const Attribute<T>& aquire(const AttributeHandle<T>& attrHandle) const {
		return m_attributes.aquire(attrHandle);
	}

	Attribute<ei::Sphere>& get_spheres() {
		return this->aquire(m_sphereData);
	}
	const Attribute<ei::Sphere>& get_spheres() const {
		return this->aquire(m_sphereData);
	}

	Attribute<MaterialIndex>& get_mat_indices() {
		return this->aquire(m_matIndex);
	}
	const Attribute<MaterialIndex>& get_mat_indices() const {
		return this->aquire(m_matIndex);
	}

	/**
	 * Returns a descriptor (on CPU side) with pointers to resources (on Device side).
	 * Takes two tuples: they must each contain the name and type of attributes which the
	 * renderer wants to have access to. If an attribute gets written to, it is the
	 * renderer's task to aquire it once more after that, since we cannot hand out
	 * Accessors to the concrete device.
	 */
	template < Device dev, class... Args >
	SpheresDescriptor<dev> get_descriptor(const std::tuple<AttrDesc<Args...>>& attribs) {
		this->synchronize<dev>();
		constexpr std::size_t numAttribs = sizeof...(Args);
		// Collect the attributes; for that, we iterate the given Attributes and
		// gather them on CPU side (or rather, their device pointers); then
		// we copy it to the actual device
		AttribBuffer<dev>& attribBuffer = m_attribBuffer.get<AttribBuffer<dev>>();
		if(numAttribs > 0) {
			// Resize the attribute array if necessary
			if(attribBuffer.size < numAttribs) {
				if(attribBuffer.size == 0)
					attribBuffer.buffer = Allocator<dev>::template alloc_array<ArrayDevHandle_t<dev, void>>(numAttribs);
				else
					attribBuffer.buffer = Allocator<dev>::template realloc(attribBuffer.buffer, attribBuffer.size,
																		   numAttribs);
				attribBuffer.size = numAttribs;
			}

			std::vector<void*> cpuAttribs(numAttribs);
			push_back_attrib<0u, dev>(cpuAttribs, attribs);
			Allocator<Device::CPU>::template copy<void*, dev>(attribBuffer.buffer, cpuAttribs.data(),
															  numAttribs);
		}

		return SpheresDescriptor<dev>{
			static_cast<u32>(this->get_sphere_count()),
			static_cast<u32>(numAttribs),
			*this->get_spheres().aquireConst<dev>(),
			*this->get_mat_indices().aquireConst<dev>(),
			attribBuffer.buffer
		};
	}
	template < Device dev >
	SpheresDescriptor<dev> get_descriptor() {
		this->synchronize<dev>();
		
		return SpheresDescriptor<dev>{
			static_cast<u32>(this->get_sphere_count()),
				0u,
				this->get_spheres().aquireConst<dev>(),
				this->get_mat_indices().aquireConst<dev>(),
				ArrayDevHandle_t<dev, ArrayDevHandle_t<dev, void>>{}
		};
	}
	
	// Synchronizes the default attributes position, normal, uv, matindex 
	template < Device dev >
	void synchronize() {
		m_attributes.synchronize<dev>();
	}

	template < Device dev >
	void unload() {
		m_attributes.unload<dev>();
	}

	const ei::Box& get_bounding_box() const noexcept {
		return m_boundingBox;
	}

	std::size_t get_sphere_count() const noexcept {
		return m_attributes.get_size();
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

	// Helper for iterating a tuple
	template < std::size_t I, Device dev, class... Args >
	void push_back_attrib(std::vector<void*>& vec, const std::tuple<Args...>& attribs) {
		if constexpr(I < sizeof...(Args)) {
			// First extract the underlying attribute type...
			using Type = std::tuple_element_t<I, std::tuple<Args...>>;
			// ...then mix it with the presumed name to find it
			const std::string& name = std::get<I>(attribs).name;
			std::optional<AttributeHandle<Type>> attribHdl = this->find<Type>(name);

			if(!attribHdl.has_value()) {
				logWarning("[Polygons::push_back_attrib] Could not find attribute '",
						   name, '\'');
				vec.push_back(nullptr);
			} else {
				vec.push_back(*this->aquire(attribHdl.value()).aquire<dev>());
			}
			push_back_attrib<I + 1u, dev>(vec, attribs);
		}
	}

	// Make sure that spheres are tightly packed
	static_assert(sizeof(ei::Sphere) == 4u * sizeof(float));

	AttributeList m_attributes;
	AttributeHandle<ei::Sphere> m_sphereData;
	AttributeHandle<MaterialIndex> m_matIndex;
	// Array for aquired attribute descriptors
	AttribBuffers m_attribBuffer;
	ei::Box m_boundingBox;
};

} // namespace mufflon::scene::geometry
