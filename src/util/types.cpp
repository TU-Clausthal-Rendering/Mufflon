#include "types.hpp"

namespace mufflon {

AreaPdf& AreaPdf::operator+=(AreaPdf pdf) noexcept {
	m_pdf += pdf.m_pdf;
	return *this;
}
AreaPdf& AreaPdf::operator-=(AreaPdf pdf) noexcept {
	m_pdf -= pdf.m_pdf;
	return *this;
}
AreaPdf& AreaPdf::operator*=(AreaPdf pdf) noexcept {
	m_pdf *= pdf.m_pdf;
	return *this;
}
AreaPdf& AreaPdf::operator/=(AreaPdf pdf) noexcept {
	m_pdf /= pdf.m_pdf;
	return *this;
}

AngularPdf& AngularPdf::operator+=(AngularPdf pdf) noexcept {
	m_pdf += pdf.m_pdf;
	return *this;
}
AngularPdf& AngularPdf::operator-=(AngularPdf pdf) noexcept {
	m_pdf -= pdf.m_pdf;
	return *this;
}
AngularPdf& AngularPdf::operator*=(AngularPdf pdf) noexcept {
	m_pdf *= pdf.m_pdf;
	return *this;
}
AngularPdf& AngularPdf::operator/=(AngularPdf pdf) noexcept {
	m_pdf /= pdf.m_pdf;
	return *this;
}

} // namespace mufflon