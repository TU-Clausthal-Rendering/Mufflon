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
	IBaseAttribute() = default;
	IBaseAttribute(const IBaseAttribute&) = default;
	IBaseAttribute(IBaseAttribute&&) = default;
	IBaseAttribute& operator=(const IBaseAttribute&) = default;
	IBaseAttribute& operator=(IBaseAttribute&&) = default;
	virtual ~IBaseAttribute() = default;
	virtual void reserve(std::size_t count) = 0;
	virtual void resize(std::size_t count) = 0;
	virtual void clear() = 0;
	virtual std::string_view name() const noexcept = 0;
	virtual bool empty() const noexcept = 0;
	virtual std::size_t size() const noexcept = 0;
	virtual std::size_t elem_size() const noexcept = 0;
	virtual std::size_t capacity() const noexcept = 0;
	virtual char* as_bytes() noexcept = 0;
	virtual const char* as_bytes() const noexcept = 0;
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

	virtual bool empty() const noexcept {
		return m_data.empty();
	}

	virtual std::size_t size() const noexcept override {
		return m_data.size();
	}

	virtual std::size_t elem_size() const noexcept override {
		return sizeof(Type);
	}

	virtual std::size_t capacity() const noexcept override {
		return m_data.capacity();
	}

	T& operator[](std::size_t index) noexcept {
		mAssert(index < m_data.size());
		return m_data[index];
	}

	const T& operator[](std::size_t index) const noexcept {
		mAssert(index < m_data.size());
		return m_data[index];
	}

	T& at(std::size_t index) {
		if(index >= m_data.size())
			throw std::out_of_range("Attribute index out of range!");
		return m_data[index];
	}

	const T& at(std::size_t index) const {
		if(index >= m_data.size())
			throw std::out_of_range("Attribute index out of range!");
		return m_data[index];
	}

	T* data() noexcept {
		return m_data.data();
	}

	const T* data() const noexcept {
		return m_data.data();
	}

	char* as_bytes() noexcept override {
		return reinterpret_cast<char*>(m_data.data());
	}

	const char* as_bytes() const noexcept override {
		return reinterpret_cast<const char*>(m_data.data());
	}

	void push_back(const Type& val) {
		m_data.push_back(val);
	}

	void push_back(Type&& val) {
		m_data.push_back(val);
	}

	template < class... Args >
	T &emplace_back(Args&& ...args) {
		return m_data.emplace_back(std::forward<Args>(args)...);
	}

	void pop_back() {
		m_data.pop_back();
	}

	T& front() {
		return m_data.front();
	}

	const T& front() const {
		return m_data.front();
	}

	T& back() {
		return m_data.back();
	}

	const T& back() const {
		return m_data.back();
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
		friend class AttributeList;

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
			do
				++m_curr;
			while(m_curr != m_attribs.end() && *m_curr != nullptr);
			return *this;
		}
		Iterator operator++(int) {
			Iterator temp(*this);
			++(*this);
			return temp;
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

	AttributeList() = default;
	AttributeList(const AttributeList&) = default;
	AttributeList(AttributeList&&) = default;
	AttributeList& operator=(const AttributeList&) = default;
	AttributeList& operator=(AttributeList&&) = default;
	~AttributeList() = default;

	/**
	 * Adds a new attribute to the attribute list.
	 * If the name is already in use, this returns std::nullopt.
	 */
	template < class Attr >
	AttributeHandle<Attr> add(std::string name) {
		// Check if attribute already exists
		auto attrIter = m_mapping.find(name);
		// Attribute "cast" to later allow us to correctly determine attribute type
		if(attrIter != m_mapping.end())
			return AttributeHandle<Attr>(attrIter->second.index());
		
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
		m_attributes.push_back(std::make_unique<Attr>(std::move(name)));
		m_mapping.insert(std::make_pair(m_attributes.back()->name(), AttributeHandle<IBaseAttribute>(newAttr.index())));
		return newAttr;
	}

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

	Iterator begin() {
		return Iterator::begin(m_attributes);
	}

	Iterator end() {
		return Iterator::end(m_attributes);
	}

private:
	std::map<std::string_view, AttributeHandle<IBaseAttribute>> m_mapping;
	ListType m_attributes;
};

} // namespace mufflon::scene