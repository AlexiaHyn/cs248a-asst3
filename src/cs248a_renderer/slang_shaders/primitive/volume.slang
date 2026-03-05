implementing "../primitive.slang";
import math;
import texture;

// The size of each marching step.
static float STEP_SIZE = 0.005;
// The density threshold to consider a hit.
static float DENSITY_THRESHOLD = 0.25;
// The delta used for finite difference normal estimation.
static float FINITE_DIFF_DELTA = 0.001;

public struct Volume
{
    public BoundingBox bound;
    public SharedTexture3D tex;
    public float4x4 modelMatrix;
    public float4x4 invModelMatrix;

    /**
     * Test ray-volume bounding box intersection.
     * @param ray The ray to test against the volume.
     * @param volumeTexBuf The texture buffer contains all the volumetrically represented models in the scene .
     * @return An Optional containing RayHitResult if the ray hits the volume, otherwise an empty Optional.
     */
    public Optional<RayHitResult> hit(Ray ray, SharedTexture3DBuffer<float4> volumeTexBuf)
    {
        // TODO: Student implementation starts here.
        // check if ray hits the volume and retrieve hit range
        Ray localRay = ray.transform(invModelMatrix);
        float2 tRange = bound.hit(localRay);
        if (tRange.x <= tRange.y) {
            // if ray hits, walk along the ray to find the closest hit voxel
            float tMin = max(tRange.x, localRay.tRange.x);
            float tMax = min(tRange.y, localRay.tRange.y);

            float t = tMin;
            while (t <= tMax) {
                float3 pLocal = localRay.origin + t * localRay.direction;
                float3 uvw = getUVW(pLocal);
                if (any(uvw < 0.0) || any(uvw > 1.0)) {
                    t += STEP_SIZE;
                    continue;
                }
                float4 density = volumeTexBuf.trilinearSample(tex, uvw);
                if (density.w >= DENSITY_THRESHOLD) {
                    RayHitResult rayHit;
                    float3 pWorld = mul(modelMatrix, float4(pLocal, 1.0)).xyz;
                    rayHit.t = length(pWorld - ray.origin);

                    // calculate normal
                    float3 delta = float3(FINITE_DIFF_DELTA, 0, 0);
                    float gx = volumeTexBuf.trilinearSample(tex, getUVW(pLocal + delta.xyy)).w 
                            - volumeTexBuf.trilinearSample(tex, getUVW(pLocal - delta.xyy)).w;
                    float gy = volumeTexBuf.trilinearSample(tex, getUVW(pLocal + delta.yxy)).w 
                            - volumeTexBuf.trilinearSample(tex, getUVW(pLocal - delta.yxy)).w;
                    float gz = volumeTexBuf.trilinearSample(tex, getUVW(pLocal + delta.yyx)).w 
                            - volumeTexBuf.trilinearSample(tex, getUVW(pLocal - delta.yyx)).w;

                    float3 gradient = float3(gx, gy, gz);
                    float3 localNormal;
                    
                    if (length(gradient) > 0.0) {
                        localNormal = -normalize(gradient);
                    } else {
                        localNormal = -ray.direction;
                    }

                    float3x3 normalMatrix = transpose((float3x3)invModelMatrix);
                    rayHit.normal = normalize(mul(normalMatrix, localNormal));
                    return Optional<RayHitResult>(rayHit);
                }
                t += STEP_SIZE;
            }
        }

        // TODO: Student implementation ends here.

        return Optional<RayHitResult>();
    }

    /**
     * Get the uvw coordinates of a point in local space.
     * @param p The point in local space.
     * @return The uvw coordinates in [0, 1].
     */
    public float3 getUVW(float3 p)
    {
        // Compute uvw coordinates in [0, 1].
        float3 minBound = this.bound.pMin;
        float3 maxBound = this.bound.pMax;
        return (p - minBound) / (maxBound - minBound);
    }
};
