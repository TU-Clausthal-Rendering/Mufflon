/*
This source is published under the following 3-clause BSD license.

Copyright (c) 2012 - 2013, Lukas Hosek and Alexander Wilkie
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

	* Redistributions of source code must retain the above copyright
	  notice, this list of conditions and the following disclaimer.
	* Redistributions in binary form must reproduce the above copyright
	  notice, this list of conditions and the following disclaimer in the
	  documentation and/or other materials provided with the distribution.
	* None of the names of the contributors may be used to endorse or promote
	  products derived from this software without specific prior written
	  permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#pragma once

#include "util/degrad.hpp"
#include "core/export/api.h"
#include <ei/vector.hpp>
#include <cuda_runtime.h>


namespace mufflon { namespace scene { namespace lights {

// RGB version of Hosek-Wilkie-skydome model (https://cgg.mff.cuni.cz/projects/SkylightModelling/)
struct HosekSkyModel {
	ei::Vec3 sunDir;
	float solarRadius;
	ei::Vec3 zenithDir;
	float turbidity;
	ei::Vec3 radiances;
	float albedo;
	ei::Vec3 configs[9u];
	Radians elevation;
};

// This data is taken from the sample code at https://cgg.mff.cuni.cz/projects/SkylightModelling/
__host__ void bake_hosek_sky_configuration(HosekSkyModel& model);

CUDA_FUNCTION ei::Vec3 get_hosek_sky_rgb_radiance(const HosekSkyModel& model, const ei::Vec3& direction) {
	// Get angle between view dir and sun
	const auto cosGamma = ei::max(0.f, ei::min(1.f, ei::dot(model.sunDir, direction)));
	const auto gamma = std::acos(cosGamma);
	// Clamp theta to the horizon
	const float cosTheta = ei::max(0.f, direction.y);
	const auto theta = ei::min(std::acos(cosTheta), ei::PI / 2.f - 0.001f);

	const ei::Vec3 expM = ei::exp(model.configs[4u] * gamma);
	const float rayM = cosGamma * cosGamma;
	const ei::Vec3 mieM = (1.f + rayM) / ei::pow(1.f + model.configs[8u] * model.configs[8u] - 2.f * model.configs[8u] * cosGamma, 1.5f);
	const float zenith = std::sqrt(cosTheta);

	const auto radiance = (1.f + model.configs[0u] * ei::exp(model.configs[1u] / (cosTheta + 0.01f)))
		* (model.configs[2u] + model.configs[3u] * expM + model.configs[5u] * rayM
		   + model.configs[6u] * mieM + model.configs[7u] * zenith);
	// Scaling to normalize brightness
	const auto skyRadiance = radiance * model.radiances * (ei::PI * ei::PI / 683.f);
	// TODO: sun radiance!
	return skyRadiance;
}

// TODO: add importance sampling

}}} // namespace mufflon::scene::lights