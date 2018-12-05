#pragma once

#include "parameter.hpp"

namespace mufflon { namespace renderer {

class OutputHandler; // TODO: implement an output handler for various configurations (variance, guides, ...)
class IParameterHandler;

class IRenderer {
public:
	virtual void iterate(OutputHandler& output) = 0;
	virtual void reset() = 0;
	virtual IParameterHandler& get_parameters() = 0;
};

}} // namespace mufflon::renderer