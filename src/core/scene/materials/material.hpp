#pragma once

#include "util/types.hpp"
#include "util/flag.hpp"
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

	// Get only the texture for emissive materials
	virtual Emission get_emission() const { return {nullptr, Spectrum{0.0f}}; }

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

	virtual Medium compute_medium() const = 0;

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
	static constexpr std::size_t _get_descriptor_size(Device device) {
		size_t texDescSize = device == Device::CPU ? sizeof(textures::ConstTextureDevHandle_t<Device::CPU>)
												   : sizeof(textures::ConstTextureDevHandle_t<Device::CUDA>);
		return sizeof(MaterialDescriptorBase) + sizeof(SubMaterial::NonTexParams)
			+ int(SubMaterial::Textures::TEX_COUNT) * texDescSize;
	}
	std::size_t get_descriptor_size(Device device) const final { return _get_descriptor_size(device); }
	std::size_t get_parameter_pack_size() const final;
	char* get_descriptor(Device device, char* outBuffer) const final;
	Emission get_emission() const final;
	Medium compute_medium() const final;

private:
	TextureHandle m_textures[int(SubMaterial::Textures::TEX_COUNT)];
	SubMaterial m_material;
};

// Automatic detection of the maximum possible descriptor size
template<int... Is>
constexpr std::array<int, sizeof...(Is)> enumerate_desc_sizes(
    std::integer_sequence<int, Is...>) {
	return {{int(ei::max(Material<Materials(Is)>::_get_descriptor_size(Device::CPU),
						 Material<Materials(Is)>::_get_descriptor_size(Device::CUDA)))...}};
}
constexpr std::size_t MAX_MATERIAL_DESCRIPTOR_SIZE() {
	int maxSize = 0;
	for(int size : enumerate_desc_sizes(std::make_integer_sequence<int, int(Materials::NUM)>{}))
		if(size > maxSize) maxSize = size;
	return sizeof(MaterialDescriptorBase) + maxSize;
}


} // namespace mufflon::scene::materials
