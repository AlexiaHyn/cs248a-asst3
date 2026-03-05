implementing "../texture.slang";

public struct SharedTexture2DBuffer<T>
    where T : IFloat
{
    public StructuredBuffer<T> buffer;

    T getBufferValue(SharedTexture2D tex, uint2 texelCoord)
    {
        uint2 texSize = tex.size;
        uint offset = tex.offset;
        uint index = texelCoord.y * texSize.x + texelCoord.x;
        return buffer[offset + index];
    }

    public T pointSample(SharedTexture2D tex, float2 uv)
    {
        T outV = T(0.0);
        // TODO: Student implementation starts here.
        float2 coord = uv * (float2(tex.size) - 1);
        uint2 texelCoord = uint2(coord + 0.5);
        outV = getBufferValue(tex, texelCoord);

        // TODO: Student implementation ends here.
        return outV;
    }

    public T bilinearSample(SharedTexture2D tex, float2 uv)
    {
        T outV = T(0.0);
        // TODO: Student implementation starts here.
        float2 coord = uv * float2(tex.size) - 0.5;
        int2 i0 = int2(floor(coord));
        int2 i1 = i0 + 1;
        float2 f = coord - float2(i0);
        uint2 c0 = uint2(clamp(i0, int2(0), int2(tex.size - 1)));
        uint2 c1 = uint2(clamp(i1, int2(0), int2(tex.size - 1)));
        T v00 = getBufferValue(tex, c0);
        T v01 = getBufferValue(tex, uint2(c0.x, c1.y));
        T v10 = getBufferValue(tex, uint2(c1.x, c0.y));
        T v11 = getBufferValue(tex, c1);
        T v0 = v00 * T(1.0 - f.x) + v10 * T(f.x);
        T v1 = v01 * T(1.0 - f.x) + v11 * T(f.x);
        outV = v0 * T(1.0 - f.y) + v1 * T(f.y);

        // TODO: Student implementation ends here.
        return outV;
    }

    public T trilinearSample(SharedTexture2D[] mipmapLevels, float2 uv, float d, uint totalLevels)
    {
        T outV = T(0.0);
        // TODO: Student implementation starts here.
        float d_clamped = clamp(d, 0.0, float(totalLevels - 1));
        uint level0 = uint(floor(d_clamped));
        uint level1 = min(level0 + 1, totalLevels - 1);
        float weight = d_clamped - float(level0);
        T color0 = bilinearSample(mipmapLevels[level0], uv);
        T color1 = bilinearSample(mipmapLevels[level1], uv);
        outV = color0 * T(1.0 - weight) + color1 * T(weight);

        // TODO: Student implementation ends here.
        return outV;
    }
}

public struct SharedTexture2D
{
    public uint2 size;
    public uint offset;
}

public struct SharedTexture3DBuffer<T>
    where T : IFloat
{
    public StructuredBuffer<T> buffer;

    T getBufferValue(SharedTexture3D tex, uint3 voxelCoord)
    {
        uint3 texSize = tex.size;
        uint offset = tex.offset;
        uint index = voxelCoord.z * texSize.y * texSize.x +
                     voxelCoord.y * texSize.x +
                     voxelCoord.x;
        return buffer[offset + index];
    }

    /**
     * Sample the texture at the given uvw coordinates using nearest neighbor interpolation.
     * @param tex The texture to sample.
     * @param uvw The uvw coordinates to sample the texture at.
     * @return The sampled value.
     */
    public T pointSample(SharedTexture3D tex, float3 uvw)
    {
        // TODO: Student implementation starts here.
        float3 coord = uvw * (float3(tex.size) - 1);
        uint3 voxelCoord = uint3(coord + 0.5);
        T bufferVal = getBufferValue(tex, voxelCoord);
        return bufferVal;

        // TODO: Student implementation ends here.
    }

    /**
     * Sample the texture at the given uvw coordinates using trilinear interpolation.
     * @param tex The texture to sample.
     * @param uvw The uvw coordinates to sample the texture at.
     * @return The sampled value.
     **/
    public T trilinearSample(SharedTexture3D tex, float3 uvw)
    {
        // TODO: Student implementation starts here.
        float3 coord = uvw * (float3(tex.size) - 1);
        T[2] yValAvgs;
        uint3[2] xPoints;
        for (float i = 0; i < 2; i++) {
            T[2] zValAvgs;
            uint3[2] yzPoints;
            for (float j = 0; j < 2; j++) {
                T[2] zVals;
                uint3[2] zPoints;
                for (float k = 0; k < 2; k++) {
                    uint3 voxelCoord = uint3(uint(coord.x + i), uint(coord.y + j), uint(coord.z + k));
                    T bufferVal = getBufferValue(tex, voxelCoord);
                    zVals[int(k)] = bufferVal;
                    zPoints[int(k)] = voxelCoord;
                    if (k == 0) {
                        yzPoints[int(j)] = voxelCoord;
                    }
                }
                T zValAvg = T(float(coord.z - zPoints[0].z) / float(zPoints[1].z - zPoints[0].z)) * (zVals[1] - zVals[0]) + zVals[0];
                zValAvgs[int(j)] = zValAvg;
                if (j == 0) {
                    xPoints[int(i)] = yzPoints[0];
                }
            }
            T yValAvg = T(float(coord.y - yzPoints[0].y) / float(yzPoints[1].y - yzPoints[0].y)) * (zValAvgs[1] - zValAvgs[0]) + zValAvgs[0];
            yValAvgs[int(i)] = yValAvg;
        }

        T xValAvg = T(float(coord.x - xPoints[0].x) / float(xPoints[1].x - xPoints[0].x)) * (yValAvgs[1] - yValAvgs[0]) + yValAvgs[0];

        return xValAvg;

        // TODO: Student implementation ends here.
    }
}

public struct SharedTexture3D
{
    public uint3 size;
    public uint offset;
}

