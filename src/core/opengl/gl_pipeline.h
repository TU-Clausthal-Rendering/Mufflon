#pragma once
#include "gl_object.h"

namespace mufflon::gl {

enum class CullMode
{
	None = 0,
	Front = 0x0404,
	Back = 0x0405,
};

enum class FillMode {
    Solid = 0x1B02,
    Wireframe = 0x1B01,
    Points = 0x1B00
};

enum class Winding
{
	CW = 0x0900,
	CCW = 0x0901
};

struct RasterizerState
{
	CullMode cullMode = CullMode::None;
	Winding frontFaceWinding = Winding::CCW;
	FillMode fillMode = FillMode::Solid;
	float lineWidth = 1.0f;
	// Discard primitives after transform feedback.
	bool discard = false;
	// Uses glColorMask to enable/disable writes to the current frame buffer.
	bool colorWrite = true;			
	// Dither color components or indices. This has an effect only if output target has a low bit-depth.
	bool dithering = true;			
};

// Comparison functions for depth and stencil buffer
enum class CmpFunc {
	Less = 0x0201,
	LessEqual = 0x0203,
	Greater = 0x0204,
	GreaterEqual = 0x0206,
	Equal = 0x0202,
	NotEqual = 0x0205,
	Never = 0x0200,
	Always = 0x0207,
};

enum class StencilOp {
	Keep = 0x1E00,
	Zero = 0,
	Replace = 0x1E01,
	Increment = 0x8507,	// Increment. Restart from zero on overflow
	Decrement = 0x8508,	// Decrement. Restart from maximum on underflow
	IncSat = 0x1E02,			// Increment. Clamp to maximum on overflow
	DecSat = 0x1E03,			// Decrement. Clamp to 0 on underflow
	Invert = 0x150A,			// Flip all bits
};

struct DepthStencilState
{
	// Enable z-test.
	bool depthTest = false;
	// Set the comparison function for z-tests.
	CmpFunc depthCmpFunc = CmpFunc::Less;
	// Enable depth writes. Otherwise z-buffer is used read only.
	bool depthWrite = true;
	// Disable both: stencil testing and updates
	bool stencilTest = false;
	// Stencil buffer comparison for front and back faces.
	CmpFunc stencilCmpFuncFront = CmpFunc::Always;
	CmpFunc stencilCmpFuncBack = CmpFunc::Always;
	// Reference values for front and back faces
	uint32_t stencilRefFront = 0;
	uint32_t stencilRefBack = 0;
	// Stencil buffer operations for front and back faces.
	// stencilFail: Did not pass stencil test.
	// zfailOpFront: Passed stencil test, but failed in z-test.
	// passOpFront: Passed both tests.
	StencilOp stencilFailOpFront = StencilOp::Keep, zfailOpFront = StencilOp::Keep, passOpFront = StencilOp::Keep;
	StencilOp stencilFailOpBack = StencilOp::Keep, zfailOpBack = StencilOp::Keep, passOpBack = StencilOp::Keep;
};

enum class BlendOp {
	Add = 0x8006,
	Subtract = 0x800A,
	ReverseSubtract = 0x800B,
	Min = 0x8007,
	Max = 0x8008
};

enum class LogicOp {
	Clear = 0x1500,
	SetOne = 0x150F,
	Copy = 0x1503,
	CopyInv = 0x150C,
	Noop = 0x1505,
	Invert = 0x150A,
	And = 0x1501,
	Nand = 0x150E,
	Or = 0x1507,
	Nor = 0x1508,
	Xor = 0x1506,
	Equiv = 0x1509,
	AndInvDst = 0x1502,
	AndInvSrc = 0x1504,
	OrInvDst = 0x150B,
	OrInvSrc = 0x150D
};

enum class BlendFactor {
	Zero = 0,
	One = 1,
	SrcAlpha = 0x0302,
	InvSrcAlpha = 0x0303,
	DstAlpha = 0x0304,
	InvDstAlpha = 0x0305,
	SrcColor = 0x0300,
	InvSrcColor = 0x0301,
	DstColor = 0x0306,
	InvDstColor = 0x0307,
};

// Logic and alphablending exclude each other globally for all buffers.
enum class BlendMode {
	Disable,
	Blend,
	Logic
};

struct BlendState {
    struct RenderTarget {
		BlendFactor srcColorFactor = BlendFactor::One, srcAlphaFactor = BlendFactor::One;
		BlendFactor dstColorFactor = BlendFactor::Zero, dstAlphaFactor = BlendFactor::Zero;
		BlendOp colorBlendOp = BlendOp::Add;
		BlendOp alphaBlendOp = BlendOp::Add;
    } renderTarget[8];

	BlendMode enableBlending = BlendMode::Disable;
	LogicOp logicOp = LogicOp::Copy;
	bool alphaToCoverage = false;
};

struct Pipeline {
	RasterizerState rasterizer;
	DepthStencilState depthStencil;
	BlendState blend;

	gl::Handle program = 0;
	gl::Handle vertexArray = 0;
	gl::Handle framebuffer = 0;
};
}
