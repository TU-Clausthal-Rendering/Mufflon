#pragma once

#include "util/assert.hpp"
#include <cstddef>
#include <map>
#include <optional>
#include <vector>

namespace mufflon::scene {

/**
 * Base class for all attribtues.
 * Any attribute which should be used in an attribute array needs to inherit
 * from this.
 */
class IBaseAttribute {
public:
	virtual ~IBaseAttribute() {}
	virtual void reserve(std::size_t count) = 0;
	virtual void resize(std::size_t count) = 0;
	virtual void clear() = 0;
	virtual std::string_view name() const noexcept = 0;
	virtual std::size_t size() const noexcept = 0;
	virtual std::size_t elem_size() const noexcept = 0;
};

template < class T >
class ArrayAttribute : public IBaseAttribute {
public:
	using Type = T;

	ArrayAttribute(std::string name) :
		IBaseAttribute(),
		m_name(std::move(name))
	{}

	ArrayAttribute(std::string name, std::size_t n_res) :
		IBaseAttribute(),
		m_name(std::move(name))
	{
		this->reserve(n_res);
	}

	virtual void reserve(std::size_t count) override {
		m_data.reserve(count);
	}

	virtual void resize(std::size_t count) override {
		m_data.resize(count);
	}

	virtual void clear() override {
		m_data.clear();
	}

	virtual std::string_view name() const noexcept override {
		return std::string_view(m_name);
	}

	virtual std::size_t size() const noexcept override {
		return m_data.size();
	}

	virtual std::size_t elem_size() const noexcept override {
		return sizeof(Type);
	}

	T &operator[](std::size_t index) noexcept {
		mAssert(index < m_data.size());
		return m_data[index];
	}

	const T &operator[](std::size_t index) const noexcept {
		mAssert(index < m_data.size());
		return m_data[index];
	}

	T &at(std::size_t index) {
		if(index >= m_data.size())
			throw std::out_of_range("Attribute index out of range!");
		return m_data[index];
	}

	const T &at(std::size_t index) const {
		if(index >= m_data.size())
			throw std::out_of_range("Attribute index out of range!");
		return m_data[index];
	}

private:
	std::string m_name;
	std::vector<T> m_data;
};

/**
 * Represents a list of attributes.
 * While it accepts any kind of attribute derived from IBaseAttribute, 
 */
class AttributeList {
public:
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

		AttributeHandle(std::size_t idx) :
			m_index(idx)
		{}
		
		std::size_t m_index;
	};

	/**
	 * Adds a new attribute to the attribute list.
	 * If the name is already in use, this returns std::nullopt.
	 */
	template < class Attr >
	std::optional<AttributeHandle<Attr>> add(std::string name) {
		// Check if attribute already exists
		// TODO: would it be better if we threw here?
		auto attrIter = m_mapping.find(name);
		// Attribute "cast" to later allow us to correctly determine attribute type
		if(attrIter != m_mapping.end())
			return std::nullopt;
		// Create new attribute and a handle for it
		m_attributes.push_back(std::make_unique<Attr>(std::move(name)));
		AttributeHandle<Attr> newAttrIter(m_attributes.size() - 1u);
		m_mapping.insert({m_attributes.back()->name(), newAttrIter});
		return newAttrIter;
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
		return AttributeHandle<Attr>(attrIter->second.m_iter);
	}

	/**
	 * Attempts to aquire a valid attribute reference from an attribute handle.
	 * If the attribute handle has a different type than the originally inserted attribute
	 * handle, a bad_cast-exception is thrown.
	 */
	template < class Attr >
	Attr &aquire(const AttributeHandle<Attr>& handle) {
		return dynamic_cast<Attr&>(m_attributes[handle.m_index]);
	}
	template < class Attr >
	const Attr &aquire(const AttributeHandle<Attr>& handle) const {
		return dynamic_cast<Attr&>(m_attributes[handle.m_index]);
	}

	/// Reserves space for all attributes
	void reserve(std::size_t count) {
		for(auto &attr : m_attributes)
			attr->reserve(count);
	}

	/// Resizes all attributes
	void resize(std::size_t count) {
		for(auto &attr : m_attributes)
			attr->resize(count);
	}

	/// Clears all attributes
	void clear() {
		for(auto &attr : m_attributes)
			attr->clear();
	}

private:
	std::map<std::string_view, AttributeHandle<IBaseAttribute>> m_mapping;
	std::vector<std::unique_ptr<IBaseAttribute>> m_attributes;
};

} // namespace mufflon::scene