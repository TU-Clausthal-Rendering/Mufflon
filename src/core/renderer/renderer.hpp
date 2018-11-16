#pragma once

namespace mufflon { namespace renderer {

class OutputHandler; // TODO: implement an output handler for various configurations (variance, guides, ...)

class IRenderer {
public:
	virtual void iterate(OutputHandler& output) const = 0;
	virtual void reset() = 0;
};

}} // namespace mufflon::renderer