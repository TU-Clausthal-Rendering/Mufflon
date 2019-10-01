const float LUT_SIZE = 64.0;
const float LUT_SCALE = (LUT_SIZE - 1.0) / LUT_SIZE;
const float LUT_BIAS = 0.5 / LUT_SIZE;

vec2 LTC_Coords(float cosTheta, float roughness)
{
	vec2 coords = vec2(roughness, sqrt(1.0 - cosTheta));
	coords = coords * LUT_SCALE + LUT_BIAS;

	return coords;
}

mat3 LTC_Matrix(sampler2DArray texLSDMat, vec2 coord)
{
	// load inverse matrix
	vec4 t = textureLod(texLSDMat, vec3(coord, 0.0), 0.0);
	mat3 Minv = mat3(
		vec3(t.x, 0.0, t.y), // first column
		vec3(0.0, 1.0, 0.0), // second column
		vec3(t.z, 0.0, t.w) // third column
	);

	return Minv;
}

vec3 LTC_IntegrateEdge(vec3 v1, vec3 v2)
{
	float x = dot(v1, v2);
	float y = abs(x);

	float a = 0.8543985 + (0.4965155 + 0.0145206*y)*y;
	float b = 3.4175940 + (4.1616724 + y)*y;
	float v = a / b;

	float theta_sintheta = (x > 0.0) ? v : 0.5*inversesqrt(max(1.0 - x * x, 1e-7)) - v;

	return cross(v1, v2)*theta_sintheta;
}

vec3 LTC_Evaluate(
	vec3 N, vec3 V, vec3 P, mat3 Minv, vec3 points[4], int nPoints, sampler2DArray helpTexture)
{
	// do early behind test
	{
		bool isHidden = true;
		for (uint i = 0; i < nPoints && i < 4; ++i) 
		{
			if (dot(normalize(P - points[i]), N) <= 0.4)
				isHidden = false;
		}

		//if (isHidden) return vec3(1.0, 0.0, 0.0);
	}


	if (nPoints > 3)
	{
		// swap 1 and 3
		vec3 tmp = points[3];
		points[3] = points[1];
		points[1] = tmp;
	}
	else
	{
		// swap 1 and 2
		vec3 tmp = points[2];
		points[2] = points[1];
		points[1] = tmp;
	}

	// construct orthonormal basis around N
	vec3 T1, T2;
	T1 = normalize(V - N * dot(V, N));
	T2 = cross(N, T1);

	// rotate area light in (T1, T2, R) basis
	Minv = Minv * transpose(mat3(T1, T2, N));

	// polygon (allocate 5 vertices for clipping)
	vec3 L[4];
	L[0] = Minv * (points[0] - P);
	L[1] = Minv * (points[1] - P);
	L[2] = Minv * (points[2] - P);
	if(nPoints > 3)	L[3] = Minv * (points[3] - P);
	else L[3] = L[0]; // edge length zero

	// clipless approximation
	vec3 dir = points[0] - P;
	vec3 lightNormal = cross(points[1] - points[0], points[2] - points[0]);
	bool behind = (dot(dir, lightNormal) <= 0.000);
	if (behind) return vec3(0.0, 0.0, 0.0);
	
	//return vec3(0.0, 1.0, 0.0);

	// project onto sphere
	L[0] = normalize(L[0]);
	L[1] = normalize(L[1]);
	L[2] = normalize(L[2]);
	L[3] = normalize(L[3]);

	// calc form factor integral
	vec3 vsum = vec3(0.0);
	vsum += LTC_IntegrateEdge(L[0], L[1]);
	vsum += LTC_IntegrateEdge(L[1], L[2]);
	vsum += LTC_IntegrateEdge(L[2], L[3]);
	if(nPoints > 3)
		vsum += LTC_IntegrateEdge(L[3], L[0]);

	float len = length(vsum);
	float z = vsum.z / len;

	vec2 uv = vec2(z*0.5 + 0.5, len);
	uv = uv * LUT_SCALE + LUT_BIAS;
	float scale = textureLod(helpTexture, vec3(uv, 0.0), 0.0).w;

	float sum = len * scale;
	//float sum = vsum.z; // ignore clipping for now
	
	vec3 Lo_i = vec3(sum, sum, sum);

	// scale by filtered light color
	//Lo_i *= textureLight;

	return Lo_i;
}

// An extended version of the implementation from
// "How to solve a cubic equation, revisited"
// http://momentsingraphics.de/?p=105
vec3 SolveCubic(vec4 Coefficient)
{
	const float pi = 3.14159265359;
	// Normalize the polynomial
	Coefficient.xyz /= Coefficient.w;
	// Divide middle coefficients by three
	Coefficient.yz /= 3.0;

	float A = Coefficient.w;
	float B = Coefficient.z;
	float C = Coefficient.y;
	float D = Coefficient.x;

	// Compute the Hessian and the discriminant
	vec3 Delta = vec3(
		-Coefficient.z*Coefficient.z + Coefficient.y,
		-Coefficient.y*Coefficient.z + Coefficient.x,
		dot(vec2(Coefficient.z, -Coefficient.y), Coefficient.xy)
	);

	float Discriminant = dot(vec2(4.0*Delta.x, -Delta.y), Delta.zy);

	vec3 RootsA, RootsD;

	vec2 xlc, xsc;

	// Algorithm A
	{
		float A_a = 1.0;
		float C_a = Delta.x;
		float D_a = -2.0*B*Delta.x + Delta.y;

		// Take the cubic root of a normalized complex number
		float Theta = atan(sqrt(Discriminant), -D_a) / 3.0;

		float x_1a = 2.0*sqrt(-C_a)*cos(Theta);
		float x_3a = 2.0*sqrt(-C_a)*cos(Theta + (2.0 / 3.0)*pi);

		float xl;
		if ((x_1a + x_3a) > 2.0*B)
			xl = x_1a;
		else
			xl = x_3a;

		xlc = vec2(xl - B, A);
	}

	// Algorithm D
	{
		float A_d = D;
		float C_d = Delta.z;
		float D_d = -D * Delta.y + 2.0*C*Delta.z;

		// Take the cubic root of a normalized complex number
		float Theta = atan(D*sqrt(Discriminant), -D_d) / 3.0;

		float x_1d = 2.0*sqrt(-C_d)*cos(Theta);
		float x_3d = 2.0*sqrt(-C_d)*cos(Theta + (2.0 / 3.0)*pi);

		float xs;
		if (x_1d + x_3d < 2.0*C)
			xs = x_1d;
		else
			xs = x_3d;

		xsc = vec2(-D, xs + C);
	}

	float E = xlc.y*xsc.y;
	float F = -xlc.x*xsc.y - xlc.y*xsc.x;
	float G = xlc.x*xsc.x;

	vec2 xmc = vec2(C*F - B * G, -B * F + C * E);

	vec3 Root = vec3(xsc.x / xsc.y, xmc.x / xmc.y, xlc.x / xlc.y);

	if (Root.x < Root.y && Root.x < Root.z)
		Root.xyz = Root.yxz;
	else if (Root.z < Root.x && Root.z < Root.y)
		Root.xyz = Root.xzy;

	return Root;
}


vec3 LTC_Evaluate(
	vec3 N, vec3 V, vec3 P, mat3 Minv, vec3 center, float radius, sampler2DArray helpTexture)
{
	// init points for the disc (quad enclosing the sphere from the materials point of view P)
	vec3 toPoint = center - P;
	vec3 x = normalize(cross(toPoint, vec3(1.0, 0.0, 0.0)));
	vec3 y = normalize(cross(toPoint, x));

	vec3 points[4];
	points[0] = center - x * radius - y * radius;
	points[1] = center + x * radius - y * radius;
	points[2] = center + x * radius + y * radius;
	points[3] = center - x * radius + y * radius;

	 // construct orthonormal basis around N
	vec3 T1, T2;
	T1 = normalize(V - N * dot(V, N));
	T2 = cross(N, T1);

	// rotate area light in (T1, T2, N) basis
	mat3 R = transpose(mat3(T1, T2, N));

	// polygon (allocate 5 vertices for clipping)
	vec3 L_[3];
	L_[0] =	R * (points[0] - P);
	L_[1] = R * (points[1] - P);
	L_[2] = R * (points[2] - P);

	vec3 Lo_i = vec3(0);

	// init ellipse
	vec3 C = 0.5 * (L_[0] + L_[2]);
	vec3 V1 = 0.5 * (L_[1] - L_[2]);
	vec3 V2 = 0.5 * (L_[1] - L_[0]);

	C = Minv * C;
	V1 = Minv * V1;
	V2 = Minv * V2;

	if (dot(cross(V1, V2), C) < 0.0)
		return vec3(0.0);

	// compute eigenvectors of ellipse
	float a, b;
	float d11 = dot(V1, V1);
	float d22 = dot(V2, V2);
	float d12 = dot(V1, V2);
	if (abs(d12) / sqrt(d11*d22) > 0.0001)
	{
		float tr = d11 + d22;
		float det = -d12 * d12 + d11 * d22;

		// use sqrt matrix to solve for eigenvalues
		det = sqrt(det);
		float u = 0.5*sqrt(tr - 2.0*det);
		float v = 0.5*sqrt(tr + 2.0*det);
		float e_max = (u + v) * (u + v);
		float e_min = (u - v) * (u - v);

		vec3 V1_, V2_;

		if (d11 > d22)
		{
			V1_ = d12 * V1 + (e_max - d11)*V2;
			V2_ = d12 * V1 + (e_min - d11)*V2;
		}
		else
		{
			V1_ = d12 * V2 + (e_max - d22)*V1;
			V2_ = d12 * V2 + (e_min - d22)*V1;
		}

		a = 1.0 / e_max;
		b = 1.0 / e_min;
		V1 = normalize(V1_);
		V2 = normalize(V2_);
	}
	else
	{
		a = 1.0 / dot(V1, V1);
		b = 1.0 / dot(V2, V2);
		V1 *= sqrt(a);
		V2 *= sqrt(b);
	}

	vec3 V3 = cross(V1, V2);
	if (dot(C, V3) < 0.0)
		V3 *= -1.0;

	float L = dot(V3, C);
	float x0 = dot(V1, C) / L;
	float y0 = dot(V2, C) / L;

	float E1 = inversesqrt(a);
	float E2 = inversesqrt(b);

	a *= L * L;
	b *= L * L;

	float c0 = a * b;
	float c1 = a * b*(1.0 + x0 * x0 + y0 * y0) - a - b;
	float c2 = 1.0 - a * (1.0 + x0 * x0) - b * (1.0 + y0 * y0);
	float c3 = 1.0;

	vec3 roots = SolveCubic(vec4(c0, c1, c2, c3));
	float e1 = roots.x;
	float e2 = roots.y;
	float e3 = roots.z;

	vec3 avgDir = vec3(a*x0 / (a - e2), b*y0 / (b - e2), 1.0);

	mat3 rotate = mat3(V1, V2, V3);

	avgDir = rotate * avgDir;
	avgDir = normalize(avgDir);

	float L1 = sqrt(-e2 / e3);
	float L2 = sqrt(-e2 / e1);

	float formFactor = L1 * L2*inversesqrt((1.0 + L1 * L1)*(1.0 + L2 * L2));

	// use tabulated horizon-clipped sphere
	vec2 uv = vec2(avgDir.z*0.5 + 0.5, formFactor);
	uv = uv * LUT_SCALE + LUT_BIAS;
	float scale = texture(helpTexture, vec3(uv, 0.0)).w;

	float spec = formFactor * scale;

	Lo_i = vec3(spec, spec, spec);

	return vec3(Lo_i);
}