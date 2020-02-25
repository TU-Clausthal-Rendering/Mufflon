#include "string_view.hpp"

namespace mufflon {

template class BasicStringView<char>;
template class BasicStringView<wchar_t>;
template class BasicStringView<char16_t>;
template class BasicStringView<char32_t>;

} // namespace mufflon