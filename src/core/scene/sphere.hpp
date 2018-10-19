#pragma once

//#include "attributes.hpp"
#include "ei/vector.hpp"
#include "util/assert.hpp"
#include "util/types.hpp"
#include <cstdint>
#include <vector>

namespace mufflon::scene {

using MaterialIndex = u16;

/**
 * Object-level representation of a sphere.
 * TODO: are more attributes necessary? Dynamic attributes?
 */
struct Sphere {
	ei::Vec3 position;
	Real radius;
	MaterialIndex matIndex;
};

/**
 * 
 */
class SphereArray {
public:
	SphereArray() = default;
	SphereArray(const SphereArray&) = default;
	SphereArray(SphereArray&&) = default;
	SphereArray& operator=(const SphereArray&) = default;
	SphereArray& operator=(SphereArray&&) = default;
	~SphereArray() = default;
	SphereArray(std::size_t count) : SphereArray() {
		this->reserve(count);
	}

	void reserve(std::size_t count) {
		m_positions.reserve(count);
	}

	void push_back(const Sphere &sphere);

	std::size_t elements() const noexcept {
		mAssert(m_positions.size() == m_radii.size()
				&& m_positions.size() == m_matIndices.size());
		return m_positions.size();
	}

private:
	std::vector<ei::Vec3> m_positions;
	std::vector<Real> m_radii;
	std::vector<MaterialIndex> m_matIndices;
	//AttributeList m_customAttributes;
};

} // namespace mufflon::scene