#pragma once

#include "util/types.hpp"
#include "util/flag.hpp"
#include "core/memory/residency.hpp"
#include "core/scene/types.hpp"
#include "core/scene/handles.hpp"
#include "medium.hpp"

namespace mufflon { namespace scene { namespace materials {

/**
 * List of all implemented materials. These materials may form a hierarchy through
 * BLEND or FRESNEL. This enum is used to dynamically dispatch sampler, evaluation
 * and fetch kernels.
 */
enum class Materials: u16 {
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

struct MaterialPropertyFlags : public util::Flags<u16> {
	static constexpr u16 EMISSIVE = 1u;		// Is any component of this material able to emit light?
	static constexpr u16 REFLECTIVE = 2u;	// BRDF = Is there any contribution from reflections? (contribution for incident and excident on the same side)
	static constexpr u16 REFRACTIVE = 4u;	// BTDF = Is there any contribution from refractions? (contribution for incident and excident on opposite sides)
	static constexpr u16 HALFVECTOR_BASED = 8u;	// Does this material need a half vector for evaluations?

	MaterialPropertyFlags() = default;
	MaterialPropertyFlags(u16 m) { mask = m; }

	bool is_emissive() const noexcept { return is_set(EMISSIVE); }
	bool is_reflective() const noexcept { return is_set(REFLECTIVE); }
	bool is_refractive() const noexcept { return is_set(REFRACTIVE); }
	bool is_halfv_based() const noexcept { return is_set(HALFVECTOR_BASED); }
};

// Base definition of material descriptors.
// Each implementation of a material adds some additional information to this descriptor.
struct MaterialDescriptorBase {
	Materials type;
	MaterialPropertyFlags flags;
	MediumHandle innerMedium;
	MediumHandle outerMedium;
};

struct ParameterPack {
	Materials type;
	MaterialPropertyFlags flags;
	MediumHandle innerMedium;
	MediumHandle outerMedium;

	// Get the medium handle dependent on the sign of a direction x
	// with respect to the normal.
	__host__ __device__ MediumHandle get_medium(float xDotN) const {
		return xDotN < 0.0f ? innerMedium : outerMedium;
	}
};

// TODO remove this and provide a scene based (detect which material needs the most at runtime) max size instead
constexpr std::size_t MAX_MATERIAL_PARAMETER_SIZE = 256;


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
	IMaterial(Materials mat) : m_type(mat) {}
	virtual ~IMaterial() = default;

	// A name of the material for mental recognition (no program logic depends on this name)
	const std::string& get_name() const noexcept { return m_name; }
	void set_name(std::string name) { m_name = move(name); }

	virtual MaterialPropertyFlags get_properties() const noexcept = 0;

	/* 
	 * Size of the material descriptor itself (mainly texture handles)
	 * The size may vary per device.
	 */
	/*template< Device dev >
	std::size_t get_handle_pack_size() const {
		switch(m_type) {
			case Materials::LAMBERT: return sizeof(LambertHandlePack<dev>);
		}
		mAssertMsg(false, "Material not (fully) implemented!");
	}*/

	virtual std::size_t get_descriptor_size(Device device) const = 0;

	/*
	 * Size of a fetched parameter instanciation from this material.
	 */
	virtual std::size_t get_parameter_pack_size() const = 0;

	/*
	 * Get the handles which are required to fetch the current material.
	 * A handle pack consits of the material type (see Materials) followed
	 * by the two media handles and model dependent handles and parameters.
	 * device: structure the output for the target device
	 * outBuffer: pointer to a writeable buffer with at least get
	 *		get_handle_pack_size(device) memory.
	 * returns: a pointer to the end of the written data. I.e. outBuffer - return
	 *		is the size of the written descriptor.
	 */
	virtual char* get_descriptor(Device device, char* outBuffer) const;

	// Get only the texture for emissive materials
	virtual TextureHandle get_emissive_texture() const = 0;

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
	void set_inner_medium(MediumHandle medium) {
		m_innerMedium = medium;
	}

	virtual Medium compute_medium() const = 0;

protected:
	MediumHandle m_innerMedium;
	MediumHandle m_outerMedium;
private:
	std::string m_name;
	Materials m_type;
};

}}} // namespace mufflon::scene::materials