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
AngularPdf AreaPdf::to_angular_pdf(Real cos, Real distSqr) const noexcept {
	return AngularPdf(m_pdf * distSqr / cos);
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
AreaPdf AngularPdf::to_area_pdf(Real cos, Real distSqr) const noexcept {
	return AreaPdf(m_pdf * cos / distSqr);
}

} // namespace mufflon