#pragma once

#include "util/filesystem.hpp"
#include <fstream>

namespace mufflon::util {

template < class T >
class SwappedVector {
public:
	using Type = T;

	SwappedVector(const fs::path& path, std::size_t slots, std::size_t slotSize) :
		m_swapFile(path, std::fstream::in | std::fstream::out | std::fstream::binary | std::fstream::trunc),
		m_activeSlot(slotSize),
		m_activeSlotIndex{ 0u }
	{
		// Create the swap file of desired size
		m_swapFile.exceptions(std::fstream::badbit | std::fstream::failbit);
		if(m_swapFile.is_open())
			m_swapFile.close();
		fs::resize_file(path, slots * slotSize * sizeof(Type));
		// Reopen the file
		m_swapFile.open(path, std::fstream::in | std::fstream::out | std::fstream::binary | std::fstream::trunc);
	}

	void change_slot(std::size_t newSlot, const bool saveCurrSlot, const bool loadNewSlot) {
		// First save the active slot to the swap file
		if(saveCurrSlot) {
			m_swapFile.seekp(m_activeSlotIndex * m_activeSlot.size() * sizeof(Type), std::fstream::beg);
			m_swapFile.write(reinterpret_cast<const char*>(m_activeSlot.data()), m_activeSlot.size() * sizeof(Type));
		}
		// Load the data from the file
		if(loadNewSlot) {
			m_swapFile.seekp(newSlot * m_activeSlot.size() * sizeof(Type), std::fstream::beg);
			m_swapFile.read(reinterpret_cast<char*>(m_activeSlot.data()), m_activeSlot.size() * sizeof(Type));
		}
		// Change the active slot
		m_activeSlotIndex = newSlot;
	}
	std::vector<Type>& active_slot() noexcept { return m_activeSlot; }
	const std::vector<Type>& active_slot() const noexcept { return m_activeSlot; }

private:
	std::fstream m_swapFile;
	std::vector<Type> m_activeSlot;
	std::size_t m_activeSlotIndex;
};

} // namespace mufflon::util