#pragma once

/*
 * File for the definition of all Material-management classes
 * MaterialConcept and MaterialSampleConcept.
 * The implementation of sampling and evaluation of the respective
 * materials can be found in files 'matname.hpp'.
 */

#include "material_concepts.hpp"
#include "microfacet_base.hpp"
#include "core/scene/handles.hpp"

namespace mufflon { namespace scene { namespace materials {

/*
 * List of all implemented materials. These materials may form a hierarchy through
 * BLEND or FRESNEL. This enum is used to dynamically dispatch sampler, evaluation
 * and fetch kernels.
 *
 * Each blended material should sort the sub-layers in descending order.
 * This simplifies loading, because there will be much less possible combinations.
 */
enum class Materials: u16 {
	EMISSIVE,					// Emitting surface
	LAMBERT,					// Lambert diffuse
	ORENNAYAR,					// Oren-Nayar diffuse
	TORRANCE,					// Torrance-Sparrow microfacet reflection
	WALTER,						// Walter microfacet refraction
	LAMBERT_EMISSIVE,			// BLEND [ LAMBERT, EMISSIVE ]
	//ORENNAYAR_EMISSIVE,		// BLEND [ ORENNAYAR, EMISSIVE ]
	TORRANCE_LAMBERT,			// BLEND [ TORRANCE, LAMBERT ]
	//FRESNEL_TORRANCE_LAMBERT,	// FRESNEL [ TORRANCE, LAMBERT ]
	//TORRANCE_ORENNAYAR,		// BLEND [ TORRANCE, ORENNAYAR ]
	//FRESNEL_TORRANCE_ORENNAYAR,	// FRESNEL [ TORRANCE, ORENNAYAR ]
	WALTER_TORRANCE,			// BLEND [ WALTER, TORRANCE ]
	//FRESNEL_WALTER_TORRANCE	// FRESNEL [ WALETR, TORRANCE ] = Specialization for glass

	NUM				// How many materials are there?
};

// Base definition of material descriptors.
// Each implementation of a material adds some additional information to this descriptor.
struct alignas(8) MaterialDescriptorBase {
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

struct ParameterPack: public MaterialDescriptorBase {};

struct Emission {
	TextureHandle texture;
	Spectrum scale;
};



// ************************************************************************* //
// LAMBERT																	 //
// ************************************************************************* //
// Sample of Lambertian diffuse material
struct MatSampleLambert {
	Spectrum albedo;
};

// Management layer of the Lambertian material
struct MatLambert {
	static constexpr MaterialPropertyFlags PROPERTIES =
		MaterialPropertyFlags::REFLECTIVE;

	enum Textures {
		ALBEDO,
		TEX_COUNT
	};

	using SampleType = MatSampleLambert;

	MatLambert(TextureHandle* texTable, int texOffset, TextureHandle albedo) {
		texTable[ALBEDO+texOffset] = albedo;
	}

	struct NonTexParams {
	} nonTexParams;
};

// ************************************************************************* //
// MICROFACET REFLECTION: TORRANCE											 //
// ************************************************************************* //
struct MatSampleTorrance {
	Spectrum albedo;
	float angle;
	ei::Vec2 roughness;
	NDF ndf;
};

// Class for the handling of the Torrance-Sparrow microfacet reflection model.
struct MatTorrance {
	static constexpr MaterialPropertyFlags PROPERTIES =
		MaterialPropertyFlags::REFLECTIVE | MaterialPropertyFlags::HALFVECTOR_BASED;

	enum Textures {
		ALBEDO,
		ROUGHNESS,
		TEX_COUNT
	};

	using SampleType = MatSampleTorrance;

	MatTorrance(TextureHandle* texTable, int texOffset, TextureHandle albedo, TextureHandle roughness, NDF ndf) {
		texTable[ALBEDO+texOffset] = albedo;
		texTable[ROUGHNESS+texOffset] = roughness;
		nonTexParams.ndf = ndf;
	}

	struct NonTexParams {
		NDF ndf;
	} nonTexParams;
};

// ************************************************************************* //
// MICROFACET REFRACTION: WALTER											 //
// ************************************************************************* //
struct MatSampleWalter {
	Spectrum absorption; // Absorption λ per meter (transmission = exp(-λ*d))
	float angle;
	ei::Vec2 roughness;
	NDF ndf;
};

// Class for the handling of the Walter microfacet refraction model.
struct MatWalter {
	static constexpr MaterialPropertyFlags PROPERTIES =
		MaterialPropertyFlags::REFLECTIVE | MaterialPropertyFlags::REFRACTIVE | MaterialPropertyFlags::HALFVECTOR_BASED;

	enum Textures {
		ROUGHNESS,
		TEX_COUNT
	};

	using SampleType = MatSampleWalter;

	MatWalter(TextureHandle* texTable, int texOffset, Spectrum absorption, float refractionIndex, TextureHandle roughness, NDF ndf) :
		refractionIndex(refractionIndex)
	{
		texTable[ROUGHNESS+texOffset] = roughness;
		nonTexParams.absorption = absorption;
		nonTexParams.ndf = ndf;
	}

	struct NonTexParams {
		Spectrum absorption;
		NDF ndf;
	} nonTexParams;
	float refractionIndex;

	Medium compute_medium() const {
		return Medium{ei::Vec2{refractionIndex, 0.0f}, nonTexParams.absorption};
	}
};

// ************************************************************************* //
// EMISSIVE																	 //
// ************************************************************************* //
struct MatSampleEmissive {
	Spectrum radiance;
};

// Lambertian self-emitting surface (uniform radiance in all view directions).
// This class is a special case - it has an BSDF value of 0 and is not sampleable
// instead it is the only class providing a non-zero output on get_emission().
struct MatEmissive {
	static constexpr MaterialPropertyFlags PROPERTIES =
		MaterialPropertyFlags::EMISSIVE;

	enum Textures {
		EMISSION,
		TEX_COUNT
	};

	using SampleType = MatSampleEmissive;

	MatEmissive(TextureHandle* texTable, int texOffset, TextureHandle emission, Spectrum scale) {
		texTable[EMISSION+texOffset] = emission;
		nonTexParams.scale = scale;
	}

	Emission get_emission(const TextureHandle* texTable, int texOffset) const {
		return {texTable[EMISSION+texOffset], nonTexParams.scale};
	}

	struct NonTexParams {
		Spectrum scale;
	} nonTexParams;
};

// ************************************************************************* //
// BLEND																	 //
// ************************************************************************* //
template<class LayerASample, class LayerBSample>
struct MatSampleBlend {
	LayerASample a;
	LayerBSample b;
	float factorA;
	float factorB;
};

// Definition of NonTexParams as non-dependent type. Otherwise the overload
// resolution for fetch does not work.
template<class LayerA, class LayerB>
struct MatNTPBlend {
	typename LayerA::NonTexParams a;
	typename LayerB::NonTexParams b;
	float factorA;
	float factorB;
};

namespace details {
	// Inspired from https://en.cppreference.com/w/cpp/utility/make_from_tuple
	// Cannot use make_from_tuple for two reasons: C++17 (non CUDA) and additional
	// arguments.
	template<class T, class Tuple, std::size_t... Is>
	constexpr T construct_layer(TextureHandle* texTable, int texOffset, Tuple&& t, std::index_sequence<Is...>) {
		return T(texTable, texOffset, std::get<Is>(std::forward<Tuple>(t))...);
	}
}

// Additive blending of two other layers using constant factors.
template<class LayerA, class LayerB>
struct MatBlend {
	static constexpr MaterialPropertyFlags PROPERTIES =
		LayerA::PROPERTIES | LayerB::PROPERTIES;

	enum Textures {
		TEX_COUNT = LayerA::TEX_COUNT + LayerB::TEX_COUNT
	};

	using SampleType = MatSampleBlend<typename LayerA::SampleType, typename LayerB::SampleType>;

	template<typename... AArgs, typename... BArgs>
	MatBlend(TextureHandle* texTable, int texOffset, float factorA, float factorB,
			 std::tuple<AArgs...>&& aArgs, std::tuple<BArgs...>&& bArgs) :
		layerA{ details::construct_layer<LayerA>(texTable, texOffset, std::forward<std::tuple<AArgs...>>(aArgs), std::make_index_sequence<sizeof...(AArgs)>()) },
		layerB{ details::construct_layer<LayerB>(texTable, texOffset+LayerA::TEX_COUNT, std::forward<std::tuple<BArgs...>>(bArgs), std::make_index_sequence<sizeof...(BArgs)>()) }
	{
		nonTexParams.a = layerA.nonTexParams;
		nonTexParams.b = layerB.nonTexParams;
		nonTexParams.factorA = factorA;
		nonTexParams.factorB = factorB;
	}

	Emission get_emission(const TextureHandle* texTable, int texOffset) const;

	Medium compute_medium() const;

	using NonTexParams = MatNTPBlend<LayerA, LayerB>;
	NonTexParams nonTexParams;
private:
	LayerA layerA;
	LayerB layerB;
};






// Type trait to convert enum -> type
template<Materials M> struct mat_info {};
template<> struct mat_info<Materials::EMISSIVE> { using type = MatEmissive; };
template<> struct mat_info<Materials::LAMBERT> { using type = MatLambert; };
template<> struct mat_info<Materials::ORENNAYAR> { using type = MatLambert; };
template<> struct mat_info<Materials::TORRANCE> { using type = MatTorrance; };
template<> struct mat_info<Materials::WALTER> { using type = MatWalter; };
template<> struct mat_info<Materials::LAMBERT_EMISSIVE> { using type = MatBlend<MatLambert,MatEmissive>; };
template<> struct mat_info<Materials::TORRANCE_LAMBERT> { using type = MatBlend<MatTorrance, MatLambert>; };
template<> struct mat_info<Materials::WALTER_TORRANCE> { using type = MatBlend<MatWalter, MatTorrance>; };
template<Materials M>
using mat_type = typename mat_info<M>::type;


// Get the full descriptor size of a material
template < Device dev, Materials M >
constexpr std::size_t get_material_descriptor_size() {
	return round_to_align<8u>(sizeof(MaterialDescriptorBase)
		+ int(mat_type<M>::Textures::TEX_COUNT) * sizeof(textures::ConstTextureDevHandle_t<dev>)
		+ (std::is_empty<typename mat_type<M>::NonTexParams>::value ? 0 : sizeof(typename mat_type<M>::NonTexParams)));
}


namespace details {
	template<typename T, int N>
	struct Array { // As CUDA replacement for std::array
		T a[N];
		constexpr T operator[] (int i) const {
			return a[i];
		}
	};
	// Automatic creation of a constant array of the number of required
	// textures for each material.
	template<int... Is>
	constexpr Array<i16, sizeof...(Is)> enumerate_tex_counts(
		std::integer_sequence<int, Is...>) {
		return {{mat_type<Materials(Is)>::Textures::TEX_COUNT...}};
	}

	// Automatic detection of the maximum possible descriptor size
	template<std::size_t... Is>
	constexpr Array<std::size_t, sizeof...(Is)> enumerate_desc_sizes(
		std::integer_sequence<std::size_t, Is...>) {
		return {{ei::max(get_material_descriptor_size<Device::CPU, Materials(Is)>(),
						 get_material_descriptor_size<Device::CUDA, Materials(Is)>())...}};
	}
}
//constexpr auto MAT_TEX_COUNT = details::enumerate_tex_counts( std::make_integer_sequence<int, int(Materials::NUM)>{} );*/

constexpr int MAT_MAX_TEX_COUNT() { // Must be a function for CUDA C++14 compatibility, otherwise a Lambda could initialize the constant
	auto MAT_TEX_COUNT = details::enumerate_tex_counts( std::make_integer_sequence<int, int(Materials::NUM)>{} );
	int maxCount = 0;
	for(int i = 0; i < int(Materials::NUM); ++i)
		if(MAT_TEX_COUNT[i] > maxCount) maxCount = MAT_TEX_COUNT[i];
	return maxCount;
};


constexpr std::size_t MAX_MATERIAL_DESCRIPTOR_SIZE() {
	auto DESC_SIZES = details::enumerate_desc_sizes( std::make_integer_sequence<std::size_t, int(Materials::NUM)>{} );
	std::size_t maxSize = 0;
	for(int i = 0; i < int(Materials::NUM); ++i)
		if(DESC_SIZES[i] > maxSize) maxSize = DESC_SIZES[i];
	return maxSize;
}

// Conversion of a runtime parameter 'mat' into a consexpr 'MatType'.
// WARNING: this is a switch -> need to return or to break; at the end
// of 'expr'.
#define material_switch(mat, expr)							\
	switch(mat) {											\
		case Materials::EMISSIVE: {							\
			using MatType = mat_type<Materials::EMISSIVE>;	\
			expr;											\
		}													\
		case Materials::LAMBERT: {							\
			using MatType = mat_type<Materials::LAMBERT>;	\
			expr;											\
		}													\
		case Materials::ORENNAYAR: {						\
			using MatType = mat_type<Materials::ORENNAYAR>;	\
			expr;											\
		}													\
		case Materials::TORRANCE: {							\
			using MatType = mat_type<Materials::TORRANCE>;	\
			expr;											\
		}													\
		case Materials::WALTER: {							\
			using MatType = mat_type<Materials::WALTER>;	\
			expr;											\
		}													\
		case Materials::LAMBERT_EMISSIVE: {					\
			using MatType = mat_type<Materials::LAMBERT_EMISSIVE>;	\
			expr;											\
		}													\
		case Materials::TORRANCE_LAMBERT: {					\
			using MatType = mat_type<Materials::TORRANCE_LAMBERT>;	\
			expr;											\
		}													\
		case Materials::WALTER_TORRANCE: {					\
			using MatType = mat_type<Materials::WALTER_TORRANCE>;	\
			expr;											\
		}													\
		default:											\
			mAssertMsg(false, "Unknown material type.");	\
	}

}}} // namespace mufflon::scene::materials
