#pragma once

namespace mufflon { namespace renderer {

/*
 * The random walk routine is a basic primitive of most Monte-Carlo based renderers.
 * It is meant to be used as a subfunction of any renderer and summariezes effects of
 * sampling and roussian roulette
 */
bool walk();

}} // namespace mufflon::renderer