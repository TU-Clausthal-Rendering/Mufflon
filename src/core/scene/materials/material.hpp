#pragma once

#include "util/types.hpp"
#include "core/scene/residency.hpp"
#include "core/scene/types.hpp"
#include "medium.hpp"

namespace mufflon { namespace scene { namespace material {

/**
 * List of all implemented materials. These materials may form a hierarchy through
 * BLEND or FRESNEL. This enum is used to dynamically dispatch sampler, evaluation
 * and fetch kernels.
 */
enum class Materials: i32 {
	LAMBERT,		// Lambert diffuse
	TORRANCE,		// Torrance-Sparrow microfacet reflection
	WALTER,			// Walter microfacet refraction
	EMISSIVE,		// Emitting surface
	ORENNAYAR,		// Oren-Nayar diffuse
	BLEND,			// Mix two other materials
	FRESNEL,		// Mix two other materials using Fresnel equations
	GLASS,			// Mix of FRESNEL [TORRANCE, WALTER]

	NUM				// How many materials are there?
};
#ifndef __CUDA_ARCH__
const std::string& to_string(Materials type);
#endif

struct HandlePack {
	Materials type;
};

struct ParameterPack {
	Materials type;
	MediumHandle outerMedium;
	MediumHandle innerMedium;
};

/**
 * High level material abstraction. A material manages a set of texture handles for
 * its own parametrization.
 * Besides some meta information, the material provides a get_parameter_pack() method which
 * instanciates a certain parameter pack. This is similar to a texture fetch of all
 * associated textures. The parameter packs are used directly by the sampling and
 * evaluation routines of the BxDFs which are specific kernels matching the material.
 */
class IMaterial {
public:
	virtual ~IMaterial() = default;

	/* 
	 * Size of the material descriptor itself (mainly texture handles)
	 * The size may vary per device.
	 */
	virtual std::size_t get_handle_pack_size(Residency device) const = 0;

	/*
	 * Size of a fetched parameter instanciation from this material.
	 * The size may vary per device.
	 */
	virtual std::size_t get_parameter_pack_size(Residency device) const = 0;

	/*
	 * Get the handles which are required to fetch the current material.
	 * A handle pack consits of the material type (see Materials) followed
	 * by the two media handles and model dependent handles and parameters.
	 * device: structure the output for the target device
	 * outBuffer: pointer to a writeable buffer with at least get
	 *		get_handle_pack_size(device) memory.
	 */
	virtual void get_handle_pack(Residency device, HandlePack* outBuffer) const = 0;

	/*
	 * Get the instanciated parameters for the evaluation of the material.
	 * A parameter pack consits of the material type (see Materials) followed
	 * by the two media handles and and specific parameters used in the
	 * sampling/evaluation routines.
	 * device: structure the output for the target device
	 * texCoord: surface texture coordinate for fetching the textures.
	 * outBuffer: pointer to a writeable buffer with at least get
	 *		get_parameter_pack_size(device) memory.
	 */
	virtual void get_parameter_pack(Residency device, const UvCoordinate& uvCoordinate, ParameterPack* outBuffer) const = 0;

	// Get the medium on the side of the normal.
	MediumHandle get_outer_medium() const {
		return m_outerMedium;
	}
	void set_outer_medium(MediumHandle medium) {
		m_outerMedium = medium;
	}
	// Get the medium on opposite side of the normal.
	MediumHandle get_inner_medium() const {
		return m_innerMedium;
	}

	// Is any component of this material able to emit light?
	virtual bool is_emissive() const = 0;
	// Is there any contribution from reflections? (contribution for incident and excident on the same side)
	virtual bool is_brdf() const = 0;
	// Is there any contribution from refractions? (contribution for incident and excident on opposite sides)
	virtual bool is_btdf() const = 0;
	// Does this material need a half vector for evaluations?
	virtual bool is_halfvector_based() const = 0;

	// TODO: move to sample
	/*
	 * Get the average color of the material (integral over all view direction in
	 * a white furnace environment. Not necessarily the correct value - approximations
	 * suffice.
	 */
	//virtual Spectrum get_albedo() const = 0;

	// Would be necessary for regularization
	//virtual Spectrum get_maximum() const = 0;

protected:
	MediumHandle m_outerMedium;
	MediumHandle m_innerMedium;
};

}}} // namespace mufflon::scene::material