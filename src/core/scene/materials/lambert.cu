#include "lambert.hpp"

namespace mufflon { namespace scene { namespace material {

__host__ __device__ Sample
lambert_sample(const LambertParameterPack& params,
			   const Direction& incidentTS,
			   const RndSet& rndSet) {
	return Sample{};
}

}}} // namespace mufflon::scene::material