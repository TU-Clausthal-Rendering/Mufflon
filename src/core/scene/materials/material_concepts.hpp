#include "util/flag.hpp"
#include "core/math/sample_types.hpp"
#include "core/scene/handles.hpp"
#include "core/scene/textures/texture.hpp"

namespace mufflon { namespace scene { namespace materials {

struct Boundary;

/* The MaterialSample concept quarantees that a material can be managed from
 * CUDA without alignment failure or similar issues. It also enforces, that
 * all methods required for materials are existent.
 *
 * Enforced members:
 *		math::EvalValue evaluate(const Direction& incidentTS, const Direction& excidentTS, Boundary& boundary) const
 *		math::PathSample sample(const Direction& incidentTS, Boundary& boundary, const math::RndSet2_1& rndSet, bool adjoint) const
 *		Spectrum get_albedo() const
 * Other conditions:
 *		alignof() <= 4
 *		trivially copyable
 */
template<class T>
class MaterialSampleConcept {
private:
	template<class T, typename = typename std::is_same< math::EvalValue,
		decltype(evaluate(std::declval<const T>(),
						  std::declval<Direction>(),
						  std::declval<Direction>(),
						  std::declval<Boundary&>()))>::type >
		static constexpr bool has_evaluate(int) { return true; }
	template<class T> static constexpr bool has_evaluate(...) { return false; }

	template<class T, typename = typename std::is_same< math::PathSample,
		decltype(sample(std::declval<const T>(),
						std::declval<Direction>(),
						std::declval<Boundary&>(),
						std::declval<math::RndSet2_1>(),
						false ))>::type >
		static constexpr bool has_sample(int) { return true; }
	template<class T> static constexpr bool has_sample(...) { return false; }

	template<class T, typename = typename std::is_same< Spectrum,
		decltype(albedo(std::declval<const T>()))>::type >
		static constexpr bool has_albedo(int) { return true; }
	template<class T> static constexpr bool has_albedo(...) { return false; }
public:
	static_assert(has_evaluate<T>(0), "Must have a function evaluate(T, Direction, Direction, Boundary&) -> math::EvalValue.");
	static_assert(has_sample<T>(0), "Must have a function sample(T, Direction, Boundary&, math::RndSet2_1, bool) -> math::PathSample.");
	static_assert(has_albedo<T>(0), "Must have a function get_albedo(T) -> Spectrum.");
	static_assert(std::is_trivially_copyable<T>::value, "Material samples must be trivially copyable.");
	static_assert(alignof(T) <= 4, "Too large alignment required.");
};

/* Material management layer.
 * A material must define a data layout to the outside world with some specific
 * types to help automated Descriptor construction.
 * Also, it must convert a set of values from a descriptor to a MaterialSample.
 *
 * Optional:
 *		A member compute_medium() which defines the inner medium.
 */
template<class T>
class MaterialConcept {
	template<class T, typename = typename std::is_same< T::NonTexParams,
		decltype(std::declval<const T>().nonTexParams)>::type >
		static constexpr bool has_params(int) { return true; }
	template<class T> static constexpr bool has_params(...) { return false; }

	template<class T, typename = MaterialSampleConcept<
		decltype( fetch(std::declval<const textures::ConstTextureDevHandle_t<CURRENT_DEV>*>(),
						std::declval<const ei::Vec4*>(),
						0,
						std::declval<const typename T::NonTexParams&>()) ) >>
		static constexpr bool has_fetch(int) { return true; }
	template<class T> static constexpr bool has_fetch(...) { return false; }
public:
	// Must define non-texture parameters
	static_assert(std::is_trivially_copyable<T::NonTexParams>::value, "Must contain an internal type 'NonTexParams' which is trivially copyable.");
	static_assert(alignof(typename T::NonTexParams) <= 4, "Must contain an internal type 'NonTexParams' which is at most 4 byte alignable.");
	static_assert(alignof(typename T::SampleType) <= 4, "Must contain an internal type 'SampleType' which is at most 4 byte alignable.");
	static constexpr MaterialSampleConcept<typename T::SampleType> SampleTypeOK {}; // The internal SampleType must fulfil the MaterialSampleConcept
	static_assert(has_params<T>(0), "Must have a member get_non_texture_parameters() const -> NonTexParams.");

	// Must define meta infomation for texture parameters
	static_assert(std::is_enum<T::Textures>::value, "Must define a Textures enum.");
	static_assert(T::Textures::TEX_COUNT >= 0, "Must define an enum member NUM for the number of textures");
	//static_assert(T::Textures::LAST >= T::Textures::NUM, "Must define an enum member LAST for the offset of textures from the next layer.");

	// Must define a conversion function
	static_assert(has_fetch<T>(0), "Must have a function fetch(const ConstTextureDevHandle_t*, const ei::Vec4*, int, NonTexParams) whose return value fulfils the MaterialSampleConcept.");
};



struct MaterialPropertyFlags : public util::Flags<u16> {
	static constexpr u16 EMISSIVE = 1u;		// Is any component of this material able to emit light?
	static constexpr u16 REFLECTIVE = 2u;	// BRDF = Is there any contribution from reflections? (contribution for incident and excident on the same side)
	static constexpr u16 REFRACTIVE = 4u;	// BTDF = Is there any contribution from refractions? (contribution for incident and excident on opposite sides)
	static constexpr u16 HALFVECTOR_BASED = 8u;	// Does this material need a half vector for evaluations?

	MaterialPropertyFlags() = default;
	constexpr MaterialPropertyFlags(u16 m) { mask = m; }

	bool is_emissive() const noexcept { return is_set(EMISSIVE); }
	bool is_reflective() const noexcept { return is_set(REFLECTIVE); }
	bool is_refractive() const noexcept { return is_set(REFRACTIVE); }
	bool is_halfv_based() const noexcept { return is_set(HALFVECTOR_BASED); }
};

namespace details {
	// Helper to compiletime check for a parameter dependent medium
	template<class T, typename = std::is_same< Medium,
		decltype( std::declval<const T>().compute_medium() ) >::type >
		static constexpr bool has_dependent_medium(int) { return true; }
	template<class T> static constexpr bool has_dependent_medium(...) { return false; }

	// Helper to check for an emission method
	template<class T, typename = 
		decltype( std::declval<const T>().get_emission(std::declval<TextureHandle*>(), 1) ) >
		static constexpr bool has_emission(int) { return true; }
	template<class T> static constexpr bool has_emission(...) { return false; }
}

}}} // namespace mufflon::scene::materials
