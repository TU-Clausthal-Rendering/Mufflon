#pragma once

#include "attribute.hpp"
#include <unordered_map>
#include <optional>
#include <vector>

namespace mufflon::scene {

/**
 * Base class for attribute list.
 * Specialization has to be used to differentiate between OpenMesh and our custom attributes.
 * OpenMesh store their properties internally, thus we need to take it by reference instead 
 * of allocating it ourselves.
 */
class BaseAttributeList {
public:
	using ListType = std::vector<std::unique_ptr<IBaseAttribute>>;

	/**
	 * Generic class serving as a handle to a given attribute.
	 * It should only be used as a means to communicate to an attribute list
	 * which attribute is supposed to be adressed.
	 */
	template < class Attr >
	class AttributeHandle {
	public:
		using AttributeType = Attr;


	private:
		using IteratorType = typename std::vector<std::unique_ptr<IBaseAttribute>>::iterator;
		friend class BaseAttributeList;

		AttributeHandle(std::size_t idx) :
			m_index(idx) {}

		constexpr std::size_t index() const noexcept {
			return m_index;
		}

		std::size_t m_index;
	};

	/// Iterator class for all attribute handles
	class Iterator {
	public:
		static Iterator begin(ListType &attribs) {
			return Iterator(attribs, attribs.begin());
		}

		static Iterator end(ListType &attribs) {
			return Iterator(attribs, attribs.end());
		}

		Iterator& operator++() {
			do {
				++m_curr;
			} while(m_curr != m_attribs.end() && *m_curr != nullptr);
			return *this;
		}
		Iterator operator++(int) {
			Iterator temp(*this);
			++(*this);
			return temp;
		}

		Iterator operator+(std::size_t i) const {
			Iterator temp(m_attribs, m_curr + i);
			while(temp.m_curr != temp.m_attribs.end() && *temp.m_curr == nullptr)
				++temp.m_curr;
		}

		IBaseAttribute& operator*() {
			mAssert(m_curr != m_attribs.end());
			return **m_curr;
		}
		const IBaseAttribute& operator*() const {
			mAssert(m_curr != m_attribs.end());
			return **m_curr;
		}

		IBaseAttribute* operator->() {
			return m_curr->get();
		}
		const IBaseAttribute* operator->() const {
			return m_curr->get();
		}

		bool operator==(const Iterator& iter) const {
			return m_curr == iter.m_curr;
		}

		bool operator!=(const Iterator& iter) const {
			return !((*this) == iter);
		}

	private:
		Iterator(ListType &attribs, typename ListType::iterator iter) :
			m_attribs(attribs),
			m_curr(iter) {}

		ListType& m_attribs;
		typename ListType::iterator m_curr;
	};

	BaseAttributeList() = default;
	BaseAttributeList(const BaseAttributeList&) = default;
	BaseAttributeList(BaseAttributeList&&) = default;
	BaseAttributeList& operator=(const BaseAttributeList&) = default;
	BaseAttributeList& operator=(BaseAttributeList&&) = default;
	~BaseAttributeList() = default;

	template < class Attr >
	void remove(const AttributeHandle<Attr>& handle) {
		auto iter = m_mapping.find(m_attributes[handle.index()]->name());
		if(iter != m_mapping.end()) {
			// Remove element from both mapping and attribute list
			m_mapping.erase(iter);
			m_attributes[handle.index()].reset();
		}
	}

	/**
	 * Finds an attribute handle by the attribute's name.
	 * If the attribute name can't be found std::nullopt is returned.
	 */
	template < class Attr >
	std::optional<AttributeHandle<Attr>> find(const std::string_view& name) {
		auto attrIter = m_mapping.find(name);
		if(attrIter == m_mapping.end())
			return std::nullopt;
		// Attribute "cast" to later allow us to correctly determine attribute type
		return AttributeHandle<Attr>(attrIter->second.index());
	}

	/**
	 * Attempts to aquire a valid attribute reference from an attribute handle.
	 * If the attribute handle has a different type than the originally inserted attribute
	 * handle, a bad_cast-exception is thrown.
	 */
	template < class Attr >
	Attr &aquire(const AttributeHandle<Attr>& handle) {
		mAssert(m_attributes[handle.index()] != nullptr);
		return dynamic_cast<Attr&>(*m_attributes[handle.index()]);
	}
	template < class Attr >
	const Attr &aquire(const AttributeHandle<Attr>& handle) const {
		mAssert(m_attributes[handle.index()] != nullptr);
		return dynamic_cast<Attr&>(*m_attributes[handle.index()]);
	}

	/// Reserves space for all attributes
	void reserve(std::size_t count) {
		for(auto &attr : m_attributes) {
			if(attr != nullptr)
				attr->reserve(count);
		}
	}

	/// Resizes all attributes
	void resize(std::size_t count) {
		for(auto &attr : m_attributes) {
			if(attr != nullptr)
				attr->resize(count);
		}
	}

	/// Clears all attributes
	void clear() {
		for(auto &attr : m_attributes) {
			if(attr != nullptr)
				attr->clear();
		}
	}

	/// Gets the number of attributes.
	std::size_t size() const noexcept {
		return m_mapping.size();
	}

	Iterator begin() {
		return Iterator::begin(m_attributes);
	}

	Iterator end() {
		return Iterator::end(m_attributes);
	}

protected:
	// Adds a new, already constructed attribute to the list
	template < class Attr >
	AttributeHandle<Attr> add(std::unique_ptr<Attr>&& attribute) {
		// Since we leave deleted attributes in the vector, we need to manually iterate it
		// Default-index is end of vector
		AttributeHandle<Attr> newAttr(m_attributes.size());
		for(std::size_t i = 0u; i < m_attributes.size(); ++i) {
			if(m_attributes[i] == nullptr) {
				// Found a hole to place attribute in
				newAttr = AttributeHandle<Attr>(i);
				break;
			}
		}

		// Create the new attribute and mapping to handle
		m_attributes.push_back(std::move(attribute));
		m_mapping.insert(std::make_pair(m_attributes.back()->name(), AttributeHandle<IBaseAttribute>(newAttr.index())));
		return newAttr;
	}

private:
	std::unordered_map<std::string_view, AttributeHandle<IBaseAttribute>> m_mapping;
	ListType m_attributes;
};

template < bool storeCpu = true >
class AttributeList : public BaseAttributeList {
public:
	/**
	 * Adds a new attribute to the attribute list.
	 * If the name is already in use, this returns std::nullopt.
	 */
	template < template < class, bool > class Attr, class T >
	AttributeHandle<Attr<T, storeCpu>> add(std::string name) {
		// Check if attribute already exists
		auto attrIter = this->find<Attr<T, storeCpu>>(name);
		if(attrIter.has_value())
			return attrIter.value();

		return BaseAttributeList::add<Attr<T, storeCpu>>(std::make_unique<Attr<T, storeCpu>>(std::move(name)));
	}
};

// Attribute specialization for OpenMesh-esque scenario
template <>
class AttributeList<false> : public BaseAttributeList {
public:
	/**
	 * Adds a new attribute to the attribute list.
	 * If the name is already in use, this returns std::nullopt.
	 */
	template < template < class, bool > class Attr, class T, class Ref >
	AttributeHandle<Attr<T, false>> add(std::string name, Ref& ref) {
		// Check if attribute already exists
		auto attrIter = this->find<Attr<T, false>>(name);
		if(attrIter.has_value())
			return attrIter.value();

		return BaseAttributeList::add<Attr<T, false>>(std::make_unique<Attr<T, false>>(std::move(name), ref));
	}
};

} // namespace mufflon::scene