#pragma once

#include "util/types.hpp"
#include "util/flag.hpp"
#include "core/memory/dyntype_memory.hpp"
#include "core/memory/residency.hpp"
#include "core/scene/types.hpp"
#include "core/scene/handles.hpp"
#include "medium.hpp"
#include "material_definitions.hpp"
#include <string>
#include <array>

namespace mufflon::scene::materials {

const std::string& to_string(Materials type);


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
	void set_name(std::string name) {
		m_name = move(name);
		// A name change is not flagged as dirty, because it does not influence
		// any property on renderer side.
	}

	virtual MaterialPropertyFlags get_properties() const noexcept = 0;

	/* 
	 * Size of the material descriptor itself (mainly texture handles)
	 * The size may vary per device.
	 */
	virtual std::size_t get_descriptor_size(Device device) const {
		return sizeof(MaterialDescriptorBase);
	}

	/*
	 * Size of a fetched parameter instanciation from this material.
	 */
	virtual std::size_t get_parameter_pack_size() const {
		return sizeof(ParameterPack);
	}

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
	virtual char* get_descriptor(Device device, char* outBuffer) const = 0;

	// Get the medium on the side of the normal.
	MediumHandle get_outer_medium() const {
		return m_outerMedium;
	}
	void set_outer_medium(MediumHandle medium) {
		m_outerMedium = medium;
		//m_dirty = true;
	}
	// Get the medium on opposite side of the normal.
	MediumHandle get_inner_medium() const {
		return m_innerMedium;
	}
	void set_inner_medium(MediumHandle medium) {
		m_innerMedium = medium;
		//m_dirty = true;
	}

	// Get the alpha texture
	TextureHandle get_alpha_texture() const noexcept {
		return m_alpha;
	}
	void set_alpha_texture(TextureHandle alpha) noexcept {
		m_alpha = alpha;
		//m_dirty = true;
	}

	// Get the displacement mat (a height map for now)
	TextureHandle get_displacement_map() const noexcept {
		return m_displacement;
	}
	TextureHandle get_displacement_max_mips() const noexcept {
		return m_displacementMaxMips;
	}
	float get_displacement_bias() const noexcept {
		return m_displacementBias;
	}
	float get_displacement_scale() const noexcept {
		return m_displacementScale;
	}
	void set_displacement(TextureHandle map, TextureHandle maxMips, const float scale = 1.f, const float bias = 0.f) noexcept {
		m_displacement = map;
		m_displacementMaxMips = maxMips;
		m_displacementBias = bias;
		m_displacementScale = scale;
	}

	virtual Medium compute_medium(const Medium& outerMedium) const = 0;

	Materials get_type() const { return m_type; }

	// Query if the material changed since last request and reset the flag.
	/*bool dirty_reset() const {
		bool dirty = m_dirty;
		m_dirty = false;
		return dirty;
	}*/

protected:
	MediumHandle m_innerMedium;
	MediumHandle m_outerMedium;
	TextureHandle m_alpha = nullptr;		// This is not part of the material descriptor, but rather
											// separately stored in the scene descriptor

	float m_displacementBias = 0.f;
	float m_displacementScale = 1.f;
	TextureHandle m_displacement = nullptr;
	TextureHandle m_displacementMaxMips = nullptr;
	//mutable bool m_dirty = true;			// Any property of the material changed

private:
	std::string m_name;
	Materials m_type;
};

static_assert(sizeof(MaterialDescriptorBase) % 8 == 0, "Size must be a multiple of 8byte, such that the appended texture handles are aligned porperly.");

// Automatic implementation of the IMaterial interface.
// To create a new material instance use 'new Material<Materials::TYPE>'.
template<Materials M>
class Material : public IMaterial {
public:
	using SubMaterial = mat_type<M>;

	template<typename... Args>
	Material(Args&&... args) : IMaterial{M},
		m_material{m_textures, 0, std::forward<Args&&>(args)...}
	{}

	MaterialPropertyFlags get_properties() const noexcept final;
	std::size_t get_descriptor_size(Device device) const final {
		return (device == Device::CPU) ? get_material_descriptor_size<Device::CPU, M>()
									   : get_material_descriptor_size<Device::CUDA, M>();
	}
	std::size_t get_parameter_pack_size() const final;
	char* get_descriptor(Device device, char* outBuffer) const final;
	Medium compute_medium(const Medium& outerMedium) const final;

private:
	TextureHandle m_textures[int(SubMaterial::Textures::TEX_COUNT)];
	SubMaterial m_material;
};


} // namespace mufflon::scene::materials
