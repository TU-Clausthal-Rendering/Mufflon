#include "byte_io.hpp"
#include "assert.hpp"

namespace mufflon::util {

ArrayStreamBuffer::ArrayStreamBuffer(const char *begin, const std::size_t bytes) :
	m_begin(begin),
	m_end(begin + bytes),
	m_current(m_begin) {
	mAssert(std::less_equal<const char*>()(m_begin, m_end));
}

ArrayStreamBuffer::int_type ArrayStreamBuffer::underflow() {
	if(m_current == m_end)
		return traits_type::eof();

	return traits_type::to_int_type(*m_current);
}

ArrayStreamBuffer::int_type ArrayStreamBuffer::uflow() {
	if(m_current == m_end)
		return traits_type::eof();

	return traits_type::to_int_type(*m_current++);
}

ArrayStreamBuffer::int_type ArrayStreamBuffer::pbackfail(int_type ch) {
	if(m_current == m_begin || (ch != traits_type::eof() && ch != m_current[-1]))
		return traits_type::eof();

	return traits_type::to_int_type(*--m_current);
}

std::streamsize ArrayStreamBuffer::showmanyc() {
	mAssert(std::less_equal<const char *>()(m_current, m_end));
	return m_end - m_current;
}


std::streampos ArrayStreamBuffer::seekoff(std::streamoff off, std::ios_base::seekdir way,
										  std::ios_base::openmode which) {
	if(way == std::ios_base::beg) {
		m_current = m_begin + off;
	} else if(way == std::ios_base::cur) {
		m_current += off;
	} else if(way == std::ios_base::end) {
		m_current = m_end;
	}

	if(m_current < m_begin || m_current > m_end)
		return -1;


	return m_current - m_begin;
}

std::streampos ArrayStreamBuffer::seekpos(std::streampos sp,
										  std::ios_base::openmode which) {
	m_current = m_begin + sp;

	if(m_current < m_begin || m_current > m_end)
		return -1;

	return m_current - m_begin;
}

} // namespace mufflon::util