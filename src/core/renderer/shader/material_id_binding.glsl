layout(binding = 1) readonly buffer u_matIndicexBuffer {
	// they are actually shorts
	uint u_matIndices[];
};

// reads a short from the material buffer
uint readMaterialShort(uint primitiveId) {
	uint index = primitiveId / 2;
	uint remainder = primitiveId % 2;
	if(remainder != 0) return u_matIndices[index] >> 16;
	return u_matIndices[index] & 0xFFFF;
}