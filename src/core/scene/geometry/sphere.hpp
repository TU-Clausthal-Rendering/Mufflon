#pragma once

#include "util/types.hpp"
#include "ei/vector.hpp"
#include "core/scene/attributes.hpp"
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
	template < class Attr >
	using AttributeHandle = AttributeList::AttributeHandle<Attr>;
	using BulkReturn = std::tuple<SphereHandle, std::size_t>;

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

	/// Default construction, creates material-index attribute.
	Spheres() :
		m_attributes(),
		m_sphereData(m_attributes.aquire(m_attributes.add<ArrayAttribute<Sphere>>("radius-position"))),
		m_matIndex(m_attributes.aquire(m_attributes.add<ArrayAttribute<MaterialIndex>>("materialIdx")))
	{}
	Spheres(const Spheres&) = default;
	Spheres(Spheres&&) = default;
	Spheres& operator=(const Spheres&) = default;
	Spheres& operator=(Spheres&&) = default;
	~Spheres() = default;

	void reserve(std::size_t count) {
		m_sphereData.reserve(count);
		m_attributes.reserve(count);
	}

	void resize(std::size_t count) {
		m_sphereData.resize(count);
		m_attributes.resize(count);
	}

	void clear() {
		m_sphereData.clear();
		m_attributes.clear();
	}

	/// Requests a new per-sphere attribute.
	template < class Attr >
	AttributeHandle<Attr> request(const std::string& name) {
		return m_attributes.add<Attr>(name);
	}

	/// Removes a per-sphere attribute.
	template < class Attr >
	void remove(const AttributeHandle<Attr> &attr) {
		m_attributes.remove(attr);
	}

	/// Finds a per-sphere attribute by name.
	template < class Attr >
	std::optional<AttributeHandle<Attr>> find(const std::string& name) {
		return m_attributes.find(name);
	}

	/// Adds a sphere.
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
	template < class Attribute >
	std::size_t add_bulk(const Attribute& attribute, const SphereHandle& startSphere,
						 std::size_t count, std::istream& attrStream) {
		std::size_t actualCount = std::min(m_sphereData.size() - startSphere, count);
		// Read the attribute from the stream
		attrStream.read(attribute.as_bytes(), actualCount * attribute.elem_size());
		std::size_t actuallyRead = static_cast<std::size_t>(attrStream.gcount()) / attribute.elem_size();
		return actuallyRead;
	}
	/// Also performs bulk-load for an attribute, but aquires it first.
	template < class Attr >
	std::size_t add_bulk(const AttributeHandle<Attr>& attrHandle,
						 const SphereHandle& startSphere, std::size_t count,
						 std::istream& attrStream) {
		Attr& attribute = m_attributes.aquire(attrHandle);
		return add_bulk(attribute, startSphere, count, attrStream);
	}

	template < class Attribute >
	Attribute &aquire(const AttributeHandle<Attribute>& attrHandle) {
		return m_attributes.aquire(attrHandle);
	}

	template < class Attribute >
	const Attribute &aquire(const AttributeHandle<Attribute>& attrHandle) const {
		// TODO: lock attributes?
		return m_attributes.aquire(attrHandle);
	}

	template < class Attribute >
	const typename Attribute::Type& get(const SphereHandle& sphereHandle,
										const AttributeHandle<Attribute>& attrHandle) const {
		this->aquire(attrHandle)[sphereHandle];
	}

	template < class Attribute >
	void set(const SphereHandle& sphereHandle, const AttributeHandle<Attribute>& attrHandle,
			 const typename Attribute::Type& val) const {
		this->aquire(attrHandle)[sphereHandle] = val;
	}

	/// Gets a constant handle to the underlying sphere data.
	const ArrayAttribute<Sphere>& native() const {
		return m_sphereData;
	}

private:
	AttributeList m_attributes;
	ArrayAttribute<Sphere>& m_sphereData;
	ArrayAttribute<MaterialIndex>& m_matIndex;
};

} // namespace mufflon::scene::geometry