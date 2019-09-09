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
	bool behind = (dot(dir, lightNormal) <= 0.001);
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
