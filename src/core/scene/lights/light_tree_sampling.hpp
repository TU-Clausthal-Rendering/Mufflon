#pragma once

#include "light_sampling.hpp"
#include "light_tree.hpp"
#include "core/cuda/cuda_utils.hpp"

namespace mufflon { namespace scene { namespace lights {

namespace lighttree_detail {

// Helper to adjust PDF by the chance to pick light type
CUDA_FUNCTION __forceinline__ Emitter adjustPdf(Emitter&& sample, float chance) {
	sample.pChoice = chance;
	sample.intensity /= chance;
	return sample;
}
CUDA_FUNCTION __forceinline__ NextEventEstimation adjustPdf(NextEventEstimation&& sample, float chance) {
	sample.creationPdf *= chance;
	sample.diffIrradiance /= chance;
	return sample;
}


// Converts the typeless memory into the given light type and samples it
inline CUDA_FUNCTION Emitter sample_light(const SceneDescriptor<CURRENT_DEV>& scene,
								   LightType type, const char* light,
								   const math::RndSet2& rnd) {
	mAssert(static_cast<u16>(type) < static_cast<u16>(LightType::NUM_LIGHTS));
	switch(type) {
		case LightType::POINT_LIGHT: return sample_light_pos(*reinterpret_cast<const PointLight*>(light), rnd);
		case LightType::SPOT_LIGHT: return sample_light_pos(*reinterpret_cast<const SpotLight*>(light), rnd);
		case LightType::AREA_LIGHT_TRIANGLE: return sample_light_pos(scene, *reinterpret_cast<const AreaLightTriangle<CURRENT_DEV>*>(light), rnd);
		case LightType::AREA_LIGHT_QUAD: return sample_light_pos(scene, *reinterpret_cast<const AreaLightQuad<CURRENT_DEV>*>(light), rnd);
		case LightType::AREA_LIGHT_SPHERE: return sample_light_pos(scene, *reinterpret_cast<const AreaLightSphere<CURRENT_DEV>*>(light), rnd);
		case LightType::DIRECTIONAL_LIGHT: return sample_light_pos(*reinterpret_cast<const DirectionalLight*>(light), scene.aabb);
		default: mAssert(false); return {};
	}
}

// Converts the typeless memory into the given light type and samples it
inline CUDA_FUNCTION NextEventEstimation connect_light(const SceneDescriptor<CURRENT_DEV>& scene,
												LightType type, const char* light,
												const ei::Vec3& position,
												const math::RndSet2& rnd) {
	mAssert(static_cast<u16>(type) < static_cast<u16>(LightType::NUM_LIGHTS));
	switch(type) {
		case LightType::POINT_LIGHT: return connect_light(scene, *reinterpret_cast<const PointLight*>(light), position, rnd);
		case LightType::SPOT_LIGHT: return connect_light(scene, *reinterpret_cast<const SpotLight*>(light), position, rnd);
		case LightType::AREA_LIGHT_TRIANGLE: return connect_light(scene, *reinterpret_cast<const AreaLightTriangle<CURRENT_DEV>*>(light), position, rnd);
		case LightType::AREA_LIGHT_QUAD: return connect_light(scene, *reinterpret_cast<const AreaLightQuad<CURRENT_DEV>*>(light), position, rnd);
		case LightType::AREA_LIGHT_SPHERE: return connect_light(scene, *reinterpret_cast<const AreaLightSphere<CURRENT_DEV>*>(light), position, rnd);
		case LightType::DIRECTIONAL_LIGHT: return connect_light(*reinterpret_cast<const DirectionalLight*>(light), position, scene.aabb);
		default: mAssert(false); return {};
	}
}

// Guide the light tree traversal based on flux only
inline CUDA_FUNCTION float guide_flux(const scene::Point&, const scene::Point&, const scene::Point&,
							   float leftFlux, float rightFlux) {
	return leftFlux / (leftFlux + rightFlux);
}

// Guide the light tree traversal based on expected contribution
inline CUDA_FUNCTION float guide_flux_pos(const scene::Point& refPosition,
								   const scene::Point& leftPosition,
								   const scene::Point& rightPosition,
								   float leftFlux, float rightFlux) {
	leftFlux /= lensq(leftPosition - refPosition);
	rightFlux /= lensq(rightPosition - refPosition);
	return leftFlux / (leftFlux + rightFlux);
}

} // namespace lighttree_detail

/** Shared code for emitting a single photon from the tree.
 * Takes the light tree, initial interval limits, and RNG number as inputs.
 * Also takes an index, which is initially used to distribute the photon
 * until it cannot uniquely identify a subtree (ie. index 1 for interval [0,2]
 * and flux distribution of 50/50).
 */
inline CUDA_FUNCTION Emitter emit(const SceneDescriptor<CURRENT_DEV>& scene,
						   const LightSubTree& tree, u64 left, u64 right,
						   u64 rndChoice, float treeProb,
						   const math::RndSet2& rnd, u32* lightIndex = nullptr) {
	using namespace lighttree_detail;

	// Traverse the tree to split chance between lights
	const LightSubTree::Node* currentNode = as<LightSubTree::Node>(tree.memory);
	u16 type = tree.root.type;
	u32 offset = 0u;
	u64 intervalLeft = left;
	u64 intervalRight = right;
	float lightProb = treeProb;
	u32 treeIndex = 1u;

	// Iterate until we hit a leaf
	while(type == LightSubTree::Node::INTERNAL_NODE_TYPE) {
		mAssert(currentNode != nullptr);
		mAssert(intervalLeft <= intervalRight);

		// Scale the flux up
		float probLeft = currentNode->left.flux / (currentNode->left.flux + currentNode->right.flux);
		// Compute the integer bounds
		u64 intervalBoundary = intervalLeft + math::percentage_of(intervalRight - intervalLeft, probLeft);
		treeIndex = treeIndex << 1u;
		if(rndChoice < intervalBoundary) {
			type = currentNode->left.type;
			offset = currentNode->left.offset;
			intervalRight = intervalBoundary;
			lightProb *= probLeft;
		} else {
			type = currentNode->right.type;
			offset = currentNode->right.offset;
			intervalLeft = intervalBoundary;
			lightProb *= (1.0f-probLeft);
			++treeIndex;
		}
		currentNode = tree.get_node(offset);
	}

	mAssert(type != LightSubTree::Node::INTERNAL_NODE_TYPE);
	// If we want to know the index we need to compute the proper one
	// (tracking left/right is not enough since the tree isn't fully balanced)
	if(lightIndex != nullptr)
		*lightIndex += (treeIndex - tree.internalNodeCount) - 1u;

	// We got a light source! Sample it
	return adjustPdf(sample_light(scene, static_cast<LightType>(type),
								  tree.memory + offset, rnd), lightProb);
}

/**
 * Emits a single photon from a light source.
 * To ensure a good distribution, we also take an index, which is used to guide
 * the descent into the tree when it is possible to do so without using RNG.
 * index: Some arbitrary index. The events are evenly distributed among the indices.
 * numIndices: Range (number) of the indices.
 * seed: A random seed to randomize the dicision. All events (enumerated by indices)
 *		must use the same number.
 */
inline CUDA_FUNCTION Emitter emit(const SceneDescriptor<CURRENT_DEV>& scene,
						   u64 index, u64 numIndices, u64 seed,
						   const math::RndSet2& rnd, u32* lightIndex = nullptr) {
	using namespace lighttree_detail;
	const LightTree<CURRENT_DEV>& tree = scene.lightTree;
	// See connect() for details on the rndChoice
	u64 rndChoice = numIndices > 0 ? seed + index * (std::numeric_limits<u64>::max() / numIndices)
		: seed;
	if(rndChoice == ~0ull) --rndChoice;

	float fluxSum = ei::sum(tree.background.flux) + tree.dirLights.root.flux + tree.posLights.root.flux;
	float envProb = ei::sum(tree.background.flux) / fluxSum;
	float dirProb = tree.dirLights.root.flux / fluxSum;
	float posProb = tree.posLights.root.flux / fluxSum;

	// Now split up based on flux
	// First is envmap...
	u64 rightEnv = math::percentage_of(std::numeric_limits<u64>::max(), envProb);
	if(rndChoice < rightEnv) {
		// Sample background
		auto photon = sample_light_pos(tree.background, scene.aabb, rnd);
		// Adjust pdf (but apply the probability to the directional pdf and NOT the pos.pdf)
		photon.intensity /= envProb;
		photon.pChoice = envProb;
		if(lightIndex != nullptr)
			*lightIndex = 0u;
		return photon;
	}
	// ...then the directional lights come...
	u64 right = math::percentage_of(std::numeric_limits<u64>::max(), envProb + dirProb);
	u64 left = rightEnv;
	float p = dirProb;	// TODO: the correct probability would be (right-left) / <64>max, but the differenze might not even noticable in a 23bit float mantissa
	const LightSubTree* subTree = &tree.dirLights;
	if(rndChoice < right) {
		mAssert(tree.dirLights.lightCount > 0u);
		if(lightIndex != nullptr)
			*lightIndex = 1u; // Background
	} else {
		mAssert(tree.posLights.lightCount > 0u);
		left = right;
		right = std::numeric_limits<u64>::max();
		subTree = &tree.posLights;
		p = posProb;
		if(lightIndex != nullptr)
			*lightIndex = 1u + static_cast<u32>(tree.dirLights.lightCount); // Background and dir lights
	}
	return emit(scene, *subTree, left, right, rndChoice, p, rnd, lightIndex);
}


/*
 * Shared code for connecting to a subtree.
 * Takes the light tree, initial interval limits, and RNG number as inputs.
 */
inline CUDA_FUNCTION NextEventEstimation connect(const SceneDescriptor<CURRENT_DEV>& scene,
										  const LightSubTree& tree, u64 left, u64 right,
										  u64 rndChoice, float treeProb, const ei::Vec3& position,
										  const math::RndSet2& rnd,
										  bool posGuide, u32* lightIndex = nullptr,
										  LightType* lightType = nullptr, u32* lightOffset = nullptr) {
	using namespace lighttree_detail;

	// Traverse the tree to split chance between lights
	const LightSubTree::Node* currentNode = as<LightSubTree::Node>(tree.memory);
	u16 type = tree.root.type;
	u32 offset = 0u;
	u64 intervalLeft = left;
	u64 intervalRight = right;
	float lightProb = treeProb;
	u32 treeIndex = 1u;

	// Iterate until we hit a leaf
	while(type == LightSubTree::Node::INTERNAL_NODE_TYPE) {
		mAssert(currentNode != nullptr);
		mAssert(intervalLeft <= intervalRight);
		
		float probLeft = 1.0f;
		if(posGuide) {
			// Find out the two cluster centers
			const ei::Vec3 leftCenter = get_center(tree.memory + currentNode->left.offset, currentNode->left.type);
			const ei::Vec3 rightCenter = get_center(tree.memory + currentNode->right.offset, currentNode->right.type);
			probLeft = guide_flux_pos(position, leftCenter, rightCenter, currentNode->left.flux, currentNode->right.flux);
		} else {
			const ei::Vec3 leftCenter, rightCenter; // Unused dummies
			probLeft = guide_flux(position, leftCenter, rightCenter, currentNode->left.flux, currentNode->right.flux);
		}
		// Compute the integer bounds
		u64 intervalBoundary = intervalLeft + math::percentage_of(intervalRight - intervalLeft, probLeft);
		treeIndex = treeIndex << 1u;
		if(rndChoice < intervalBoundary) {
			type = currentNode->left.type;
			offset = currentNode->left.offset;
			intervalRight = intervalBoundary;
			lightProb *= probLeft;
		} else {
			type = currentNode->right.type;
			offset = currentNode->right.offset;
			intervalLeft = intervalBoundary;
			lightProb *= (1.0f-probLeft);
			++treeIndex;
		}

		currentNode = tree.get_node(offset);
	}

	// If we want to know the index we need to compute the proper one
	// (tracking left/right is not enough since the tree isn't fully balanced)
	if(lightIndex != nullptr)
		*lightIndex += (treeIndex - tree.internalNodeCount) - 1u;

	mAssert(type != LightSubTree::Node::INTERNAL_NODE_TYPE);
	if(lightType != nullptr)
		*lightType = static_cast<LightType>(type);
	if(lightOffset != nullptr)
		*lightOffset = offset;
	// We got a light source! Sample it
	return adjustPdf(connect_light(scene, static_cast<LightType>(type), tree.memory + offset,
							 position, rnd), lightProb);
}

/*
 * Performs next-event estimation.
 * For selecting the light source we want to connect to, we try to maximize
 * the irradiance. Also, this method is able to stratify samples if index ranges are used.
 * Stratification in this method increases the correllation.
 *
 * index: Some arbitrary index. The events are evenly distributed among the indices.
 * numIndices: Range (number) of the indices.
 * seed: A random seed to randomize the dicision. All events (enumerated by indices)
 *		must use the same number.
 * position: A reference position to estimate the expected irradiance.
 * bounds: The scenes bounding box.
 * rnd: A randset used to sample the position on the light source
 * guide: A function to get a cheap prediction of irradiance.
 *		Ready to use implementations: guide_flux (ignores the reference position)
 *		or guide_flux_pos
 */
inline CUDA_FUNCTION NextEventEstimation connect(const SceneDescriptor<CURRENT_DEV>& scene, u64 index,
										  u64 numIndices, u64 seed, const ei::Vec3& position,
										  const math::RndSet2& rnd, u32* lightIndex = nullptr,
										  LightType* lightType = nullptr, u32* lightOffset = nullptr) {
	using namespace lighttree_detail;
	const LightTree<CURRENT_DEV>& tree = scene.lightTree;
	// Scale the indices such that they sample the u64-intervall equally.
	// The (modulu) addition with the seed randomizes the choice.
	// Since the distance between samples is constant this will lead to a
	// correllation, but also a maximized equal distribution.
	u64 rndChoice = numIndices > 0 ? seed + index * (std::numeric_limits<u64>::max() / numIndices)
								   : seed;
	// Ensure that rndChoice != u64::max(). Otherwise it is possible to sample
	// right trees with 0 probability.
	if(rndChoice == ~0ull) --rndChoice; // TODO: better strategy?

	float fluxSum = ei::sum(tree.background.flux) + tree.dirLights.root.flux + tree.posLights.root.flux;
	float envProb = ei::sum(tree.background.flux) / fluxSum;
	float dirProb = tree.dirLights.root.flux / fluxSum;
	float posProb = tree.posLights.root.flux / fluxSum;

	// Now split up based on flux
	// First is envmap...
	u64 rightEnv = math::percentage_of(std::numeric_limits<u64>::max(), envProb);
	if(rndChoice < rightEnv) {
		if(lightIndex != nullptr)
			*lightIndex = 0u;
		if(lightType != nullptr)
			*lightType = LightType::ENVMAP_LIGHT;
		return adjustPdf(connect_light(tree.background, position, scene.aabb, rnd), envProb);
	}
	// ...then the directional lights come...
	u64 right = math::percentage_of(std::numeric_limits<u64>::max(), envProb + dirProb);
	u64 left = rightEnv;
	float p = dirProb;	// TODO: the correct probability would be (right-left) / <64>max, but the differenze might not even noticable in a 23bit float mantissa
	const LightSubTree* subTree = &tree.dirLights;
	if(rndChoice < right) {
		mAssert(tree.dirLights.lightCount > 0u);
		if(lightIndex != nullptr)
			*lightIndex = 1u; // Background
	} else {
		mAssert(tree.posLights.lightCount > 0u);
		left = right;
		right = std::numeric_limits<u64>::max();
		subTree = &tree.posLights;
		p = posProb;
		if(lightIndex != nullptr)
			*lightIndex = 1u + static_cast<u32>(tree.dirLights.lightCount); // Background and dir lights
	}
	return connect(scene, *subTree, left, right, rndChoice, p, position, rnd, tree.posGuide,
				   lightIndex, lightType, lightOffset);
}

/*
 * Hitable light source (area lights) must provide MIS helpers which are
 * called if a surface is hit randomly. This method computes the area pdf
 * which would be produced by the above connect_light() samplers.
 *
 * connect: Get the pdf of connect() [true] or that of emit() [false].
 *		For both variants there is an alias function called connect_pdf()
 *		and emit_pdf() respectively.
 */
inline CUDA_FUNCTION LightPdfs light_pdf(const LightTree<CURRENT_DEV>& tree,
								PrimitiveHandle primitive, ei::Vec2 surfaceParams,
								const ei::Vec3& refPosition) {
	mAssert(primitive.instanceId != -1);
	using namespace lighttree_detail;

	float p = tree.posLights.root.flux / tree.get_flux();
	u32 code = *tree.primToNodePath.find(primitive); // If crash here, you have hit an emissive surface which is not in the light tree. This is a fundamental problem and not only an access violation.

	// Travers through the tree to compute the complete, guide dependent pdf
	u32 offset = 0u;
	u16 type = tree.posLights.root.type;
	// Iterate until we hit a leaf
	while(type == LightSubTree::Node::INTERNAL_NODE_TYPE) {
		const LightSubTree::Node* currentNode = tree.posLights.get_node(offset);
		float pLeft = 1.0f;
		if(tree.posGuide) {
			// Find out the two cluster centers
			const ei::Vec3 leftCenter = get_center(tree.posLights.memory + currentNode->left.offset, currentNode->left.type);
			const ei::Vec3 rightCenter = get_center(tree.posLights.memory + currentNode->right.offset, currentNode->right.type);
			pLeft = guide_flux_pos(refPosition, leftCenter, rightCenter, currentNode->left.flux, currentNode->right.flux);
		} else {
			const ei::Vec3 leftCenter, rightCenter; // Dummies
			pLeft = guide_flux(refPosition, leftCenter, rightCenter, currentNode->left.flux, currentNode->right.flux);
		}

		// Go right? The code has stored the path to the primitive (beginning with the most significant bit).
		if(code & 0x80000000) {
			p *= (1.0f - pLeft);
			type = currentNode->right.type;
			offset = currentNode->right.offset;
		} else {
			p *= pLeft;
			type = currentNode->left.type;
			offset = currentNode->left.offset;
		}
		code <<= 1;
	}

	// Now, p is the choice probability, but we also need the surface area
	switch(static_cast<LightType>(type)) {
		case LightType::AREA_LIGHT_TRIANGLE: {
			auto& a = *as<AreaLightTriangle<CURRENT_DEV>>(tree.posLights.memory + offset);
			float area = len(cross(a.posV[1u], a.posV[2u])) / 2.0f;
			AreaPdf pdf { p / area };
			return { pdf, pdf };
		}
		case LightType::AREA_LIGHT_QUAD: {
			auto& light = *as<AreaLightQuad<CURRENT_DEV>>(tree.posLights.memory + offset);
			// Compute the local density at the point of the surface
			const ei::Vec3 tangentX = light.posV[1u] + surfaceParams.x * light.posV[3u];
			const ei::Vec3 tangentY = light.posV[2u] + surfaceParams.y * light.posV[3u];
			AreaPdf pdf { p / len(cross(tangentY, tangentX)) };
			return { pdf, pdf };
		}
		case LightType::AREA_LIGHT_SPHERE: {
			auto& a = *as<AreaLightSphere<CURRENT_DEV>>(tree.posLights.memory + offset);
			// Only the visible part of the sphere is sampled in connections.
			// Therefore, the area depends on the distance between reference and sphere.
			float cosSphere = a.radius / len(refPosition - a.position);
			mAssert(cosSphere >= 0.0f && cosSphere <= 1.0f);
			float solidAngle = 2 * ei::PI * (1.0f - cosSphere);
			float rsq = a.radius * a.radius;
			AreaPdf connectPdf { p / (solidAngle * rsq) };
			// Emit pdf is uniform on the surface
			AreaPdf emitPdf { p / (4 * ei::PI * rsq) };
			return { emitPdf, connectPdf };
		}
		default:
			mAssertMsg(false, "Decoded node must be some hitable area light.");
	}
	return { AreaPdf{0.0f}, AreaPdf{0.0f} };
}

/*
 * Analogous to the area light hit-pdf there is an environment hit-pdf.
 * While the pdf is an AngularPdf it is reinterpreted as AreaPdf for compatibility reasons.
 */
inline CUDA_FUNCTION AreaPdf background_pdf(const LightTree<CURRENT_DEV>& tree, const math::EvalValue& value) {
	float backgroundFlux = ei::sum(tree.background.flux);
	float p = backgroundFlux / (tree.dirLights.root.flux + tree.posLights.root.flux + backgroundFlux);
	return AreaPdf{ float(value.pdf.back) * p };
}

}}} // namespace mufflon::scene::lights
