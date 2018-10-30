#pragma once

#include "util/types.hpp"
#include "util/flag.hpp"
#include "ei/vector.hpp"
#include "../attribute_list.hpp"
#include "../residency.hpp"
#include "core/scene/types.hpp"
#include <istream>
#include <tuple>
#include <vector>

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
	using AttributeListType = AttributeList<false>;
	template < class Attr >
	using AttributeHandle = typename AttributeListType::template AttributeHandle<Attr>;
	template < class T >
	using Attribute = typename AttributeListType::template Attribute<T>;

	// Struct communicating the number of bulk-read spheres
	struct BulkReturn {
		SphereHandle handle;
		std::size_t readSpheres;
	};

	/**
	 * Sphere class.
	 * As a speciality, spheres store their position and radius packed in a float4.
	 */
	struct Sphere {
		// Note: this is (in theory) ILLEGAL C++, leading to undefined behaviour.
		// However, MSVC seems to understand that not having type punning through unions
		// is stupid and thus supports this, same with GCC. In the same vein, we use
		// anonymous unions, which is part of C11 but only available as an extension in
		// MSVC and GCC.
		union {
			struct {
				ei::Vec3 position;
				float radius;
			};
			float arr[4u];
		} m_radPos;
	};

	// Default construction, creates material-index attribute.
	Spheres() :
		m_attributes(),
		m_sphereData(m_attributes.aquire(m_attributes.add<Sphere>("radius-position"))),
		m_matIndex(m_attributes.aquire(m_attributes.add<MaterialIndex>("materialIdx")))
	{}
	Spheres(const Spheres&) = default;
	Spheres(Spheres&&) = default;
	Spheres& operator=(const Spheres&) = delete;
	Spheres& operator=(Spheres&&) = delete;
	~Spheres() = default;

	void resize(std::size_t count) {
		m_attributes.resize(count);
	}

	// Requests a new per-sphere attribute.
	template < class Attr >
	AttributeHandle<Attr> request(const std::string& name) {
		return m_attributes.add<Attr>(name);
	}

	// Removes a per-sphere attribute.
	template < class Attr >
	void remove(const AttributeHandle<Attr> &attr) {
		m_attributes.remove(attr);
	}

	// Finds a per-sphere attribute by name.
	template < class Attr >
	std::optional<AttributeHandle<Attr>> find(const std::string& name) {
		return m_attributes.find(name);
	}

	// Adds a sphere.
	SphereHandle add(const Point& point, float radius, MaterialIndex idx);
	/**
	 * Adds a bulk of spheres.
	 * Returns both a handle to the first added sphere as well as the number of
	 * read spheres.
	 */
	BulkReturn add_bulk(std::size_t count, std::istream& radPosStream);
	/**
	 * Bulk-loads the given attribute starting at the given sphere.
	 * The number of read values will be capped by the number of spheres present
	 * after the starting position.
	 */
	template < class Type >
	std::size_t add_bulk(Attribute<Type>& attribute, const SphereHandle& startSphere,
						 std::size_t count, std::istream& attrStream) {
		std::size_t start = startSphere;
		if(start >= m_attributes.get_size())
			return 0u;
		if(start + count >= m_attributes.get_size())
			m_attributes.resize(start + count);
		return attribute.restore(attrStream, start, count);
	}
	// Also performs bulk-load for an attribute, but aquires it first.
	template < class Attr >
	std::size_t add_bulk(const AttributeHandle<Attr>& attrHandle,
						 const SphereHandle& startSphere, std::size_t count,
						 std::istream& attrStream) {
		Attr& attribute = m_attributes.aquire(attrHandle);
		return this->add_bulk(attribute, startSphere, count, attrStream);
	}

	template < class Attr >
	Attr &aquire(const AttributeHandle<Attr>& attrHandle) {
		m_attributes.aquire(attrHandle);
	}

	template < class Attr >
	const Attr &aquire(const AttributeHandle<Attr>& attrHandle) const {
		m_attributes.aquire(attrHandle);
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

private:
	AttributeListType m_attributes;
	Attribute<Sphere>& m_sphereData;
	Attribute<MaterialIndex>& m_matIndex;
};

} // namespace mufflon::scene::geometry
