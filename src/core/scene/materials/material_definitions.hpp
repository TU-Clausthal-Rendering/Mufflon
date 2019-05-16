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
#include "core/math/tabulated_function.hpp"

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
	FRESNEL_TORRANCE_LAMBERT,	// FRESNEL [ TORRANCE, LAMBERT ]
	//TORRANCE_ORENNAYAR,		// BLEND [ TORRANCE, ORENNAYAR ]
	//FRESNEL_TORRANCE_ORENNAYAR,	// FRESNEL [ TORRANCE, ORENNAYAR ]
	WALTER_TORRANCE,			// BLEND [ WALTER, TORRANCE ]
	FRESNEL_TORRANCE_WALTER,	// FRESNEL [ TORRANCE, WALTER ]
	MICROFACET,					// MICROFACET (effecient FRESNEL [ TORRANCE, WALTER ])

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
// OREN-NAYAR																 //
// ************************************************************************* //
// Sample of Oren-Nayar diffuse material
struct MatSampleOrenNayar {
	Spectrum albedo;
	float a, b;
};

// Management layer of the Oren-Nayar material
struct MatOrenNayar {
	static constexpr MaterialPropertyFlags PROPERTIES =
		MaterialPropertyFlags::REFLECTIVE;

	enum Textures {
		ALBEDO,
		TEX_COUNT
	};

	using SampleType = MatSampleOrenNayar;

	MatOrenNayar(TextureHandle* texTable, int texOffset, TextureHandle albedo, float roughness) {
		texTable[ALBEDO+texOffset] = albedo;
		float ssq = roughness * roughness;
		nonTexParams.a = 1.0f - ssq / (2*ssq + 0.66f);
		nonTexParams.b = 0.45f * ssq / (ssq + 0.09f);
		// Rescale to decrease energy loss
		nonTexParams.a /= ALBEDO_AVG(roughness);
		nonTexParams.b /= ALBEDO_AVG(roughness);
	}

	struct NonTexParams {
		float a;
		float b;
	} nonTexParams;

	// Lookup tables for energy compensation
	// Table1: maximum directional albedo.
	//		Use for a physical plausible model with respect to all directions.
	static constexpr mufflon::math::TabulatedFunction<33> ALBEDO_MAX { 0.0f, ei::PI,
		{ 1.0f, 1.0075758798f, 1.0151743809f, 1.0063538621f, 0.9828398736f,
		  0.9528022529f, 0.9223891535f, 0.8946505486f, 0.8706137643f, 0.8502976999f,
		  0.8333128438f, 0.8191573529f, 0.8073450181f, 0.7974512353f, 0.7891224362f,
		  0.7820709339f, 0.7760651278f, 0.7709192682f, 0.76648434f, 0.7626404831f,
		  0.7592908962f, 0.7563570202f, 0.7537747609f, 0.7514915367f, 0.7494639705f,
		  0.747656081f, 0.7460378603f, 0.7445841511f, 0.7432737558f, 0.7420887262f,
		  0.7410137942f, 0.7400359135f, 0.7391438884f
		}
	};
	// Table2: average albedo.
	//		Prefer for artistic reasons.
	//		This model is energy preserving in total, but not for individual directions.
	static constexpr mufflon::math::TabulatedFunction<33> ALBEDO_AVG { 0.0f, ei::PI,
		{ 1.0f, 0.9984515433f, 0.9868863928f, 0.9600609238f, 0.9232777349f,
		  0.8841291491f, 0.8474927155f, 0.8154250323f, 0.7883002593f, 0.7657241714f,
		  0.7470450606f, 0.7315916234f, 0.7187656605f, 0.7080666669f, 0.6990884661f,
		  0.6915061377f, 0.6850612128f, 0.6795481399f, 0.6748032291f, 0.6706953859f,
		  0.6671192049f, 0.6639894329f, 0.6612367028f, 0.658804242f, 0.6566453051f,
		  0.6547211818f, 0.6529996354f, 0.6514536737f, 0.6500605773f, 0.648802979f,
		  0.6476608432f, 0.6466202214f, 0.6456728566f
		}
	};
};

// ************************************************************************* //
// MICROFACET REFLECTION: TORRANCE											 //
// ************************************************************************* //
struct MatSampleTorrance {
	Spectrum albedo;
	NDF ndf;
	ei::Vec2 roughness;
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
	NDF ndf;
	ei::Vec2 roughness;
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
// MICROFACET REFRACTION: TORRANCE + WALTER									 //
// ************************************************************************* //
struct MatSampleMicrofacet {
	Spectrum absorption; // Absorption λ per meter (transmission = exp(-λ*d))
	NDF ndf;
	ei::Vec2 roughness;
};

// Class for the handling of the Walter microfacet refraction model.
struct MatMicrofacet {
	static constexpr MaterialPropertyFlags PROPERTIES =
		MaterialPropertyFlags::REFLECTIVE | MaterialPropertyFlags::REFRACTIVE | MaterialPropertyFlags::HALFVECTOR_BASED;

	enum Textures {
		ROUGHNESS,
		TEX_COUNT
	};

	using SampleType = MatSampleMicrofacet;

	MatMicrofacet(TextureHandle* texTable, int texOffset, Spectrum absorption, float refractionIndex, TextureHandle roughness, NDF ndf) :
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

	Medium compute_medium(const Medium& outerMedium) const;

	using NonTexParams = MatNTPBlend<LayerA, LayerB>;
	NonTexParams nonTexParams;
private:
	LayerA layerA;
	LayerB layerB;
};


// ************************************************************************* //
// BLEND - FRESNEL															 //
// ************************************************************************* //
template<class LayerASample, class LayerBSample>
struct MatSampleBlendFresnel {
	LayerASample a;
	LayerBSample b;
	//float pReflect;	// Reflection probability constant for sampling
};

// Definition of NonTexParams as non-dependent type. Otherwise the overload
// resolution for fetch does not work.
template<class LayerA, class LayerB>
struct MatNTPBlendFresnel {
	typename LayerA::NonTexParams a;
	typename LayerB::NonTexParams b;
};

// Additive blending of two other layers using Dielectric Fresnel equations.
// TODO: Conductor
template<class LayerA, class LayerB>
struct MatBlendFresnel {
	static constexpr MaterialPropertyFlags PROPERTIES =
		LayerA::PROPERTIES | LayerB::PROPERTIES;

	enum Textures {
		TEX_COUNT = LayerA::TEX_COUNT + LayerB::TEX_COUNT
	};

	using SampleType = MatSampleBlendFresnel<typename LayerA::SampleType, typename LayerB::SampleType>;

	template<typename... AArgs, typename... BArgs>
	MatBlendFresnel(TextureHandle* texTable, int texOffset, ei::Vec2 ior,
			 std::tuple<AArgs...>&& aArgs, std::tuple<BArgs...>&& bArgs) :
		ior{ ior },
		layerA{ details::construct_layer<LayerA>(texTable, texOffset, std::forward<std::tuple<AArgs...>>(aArgs), std::make_index_sequence<sizeof...(AArgs)>()) },
		layerB{ details::construct_layer<LayerB>(texTable, texOffset+LayerA::TEX_COUNT, std::forward<std::tuple<BArgs...>>(bArgs), std::make_index_sequence<sizeof...(BArgs)>()) }
	{
		nonTexParams.a = layerA.nonTexParams;
		nonTexParams.b = layerB.nonTexParams;
		// Compute a scalar average reflection factor.
		// This is required for sampling (see sampling method).
		//float f0 = ei::sq( (ior - 1.0f) / (ior + 1.0f) );
		//nonTexParams.pReflect = 0.89176122288449f * f0 + 0.10823877711551f;
	}

	Medium compute_medium(const Medium& outerMedium) const;

	using NonTexParams = MatNTPBlendFresnel<LayerA, LayerB>;
	NonTexParams nonTexParams;
	ei::Vec2 ior;
private:
	LayerA layerA;	// Reflection layer
	LayerB layerB;	// Refraction layer
};






// Type trait to convert enum -> type
template<Materials M> struct mat_info {};
template<> struct mat_info<Materials::EMISSIVE> { using type = MatEmissive; };
template<> struct mat_info<Materials::LAMBERT> { using type = MatLambert; };
template<> struct mat_info<Materials::ORENNAYAR> { using type = MatOrenNayar; };
template<> struct mat_info<Materials::TORRANCE> { using type = MatTorrance; };
template<> struct mat_info<Materials::WALTER> { using type = MatWalter; };
template<> struct mat_info<Materials::LAMBERT_EMISSIVE> { using type = MatBlend<MatLambert,MatEmissive>; };
template<> struct mat_info<Materials::TORRANCE_LAMBERT> { using type = MatBlend<MatTorrance, MatLambert>; };
template<> struct mat_info<Materials::FRESNEL_TORRANCE_LAMBERT> { using type = MatBlendFresnel<MatTorrance, MatLambert>; };
template<> struct mat_info<Materials::WALTER_TORRANCE> { using type = MatBlend<MatWalter, MatTorrance>; };
template<> struct mat_info<Materials::FRESNEL_TORRANCE_WALTER> { using type = MatBlendFresnel<MatTorrance, MatWalter>; };
template<> struct mat_info<Materials::MICROFACET> { using type = MatMicrofacet; };
template<Materials M>
using mat_type = typename mat_info<M>::type;


// Get the full descriptor size of a material
template < Device dev, Materials M >
constexpr std::size_t get_material_descriptor_size() {
	return round_to_align<8u>(sizeof(MaterialDescriptorBase)
		+ int(mat_type<M>::Textures::TEX_COUNT) * sizeof(textures::ConstTextureDevHandle_t<dev>)
		+ (std::is_empty<typename mat_type<M>::NonTexParams>::value ? 0 : sizeof(typename mat_type<M>::NonTexParams)));
}

// Get the full param size of a material
template < Materials M >
constexpr std::size_t get_material_param_size() {
	return sizeof(MaterialDescriptorBase) + sizeof(typename mat_type<M>::SampleType);
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

	// Automatic detection of the maximum possible param size
	template<std::size_t... Is>
	constexpr Array<std::size_t, sizeof...(Is)> enumerate_param_sizes(
		std::integer_sequence<std::size_t, Is...>) {
		return {{get_material_param_size<Materials(Is)>()...}};
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

constexpr std::size_t MAX_MATERIAL_PARAM_SIZE() {
	auto PARAM_SIZES = details::enumerate_param_sizes( std::make_integer_sequence<std::size_t, int(Materials::NUM)>{} );
	std::size_t maxSize = 0;
	for(int i = 0; i < int(Materials::NUM); ++i)
		if(PARAM_SIZES[i] > maxSize) maxSize = PARAM_SIZES[i];
	return maxSize;
}


struct ParameterPack: public MaterialDescriptorBase {
	u8 subParams[MAX_MATERIAL_PARAM_SIZE() - sizeof(MaterialDescriptorBase)];
};


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
		case Materials::FRESNEL_TORRANCE_LAMBERT: {			\
			using MatType = mat_type<Materials::FRESNEL_TORRANCE_LAMBERT>;	\
			expr;											\
		}													\
		case Materials::WALTER_TORRANCE: {					\
			using MatType = mat_type<Materials::WALTER_TORRANCE>;	\
			expr;											\
		}													\
		case Materials::FRESNEL_TORRANCE_WALTER: {			\
			using MatType = mat_type<Materials::FRESNEL_TORRANCE_WALTER>;	\
			expr;											\
		}													\
		case Materials::MICROFACET: {						\
			using MatType = mat_type<Materials::MICROFACET>;	\
			expr;											\
		}													\
		default:											\
			mAssertMsg(false, "Unknown material type.");	\
	}

}}} // namespace mufflon::scene::materials
