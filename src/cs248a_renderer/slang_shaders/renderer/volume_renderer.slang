implementing "../renderer.slang";

import math;
import texture;

public static uint MAX_STEPS = 1024;
public static float EPSILON = 1e-4;

public static float STEP_SIZE = 0.01;
public static float DENSITY_SCALE = 5.0;

/**
 * Perform ray marching through the volume and accumulate color.
 * @param uv The normalized uv coordinate from (0, 0) in the bottom-left to (1, 1) in the top-right.
 * @param uniforms The renderer uniform containing camera, sampling, and scene information.
 * @param tMax The maximum t value for ray marching (usually the nearest primitive intersection t).
 * @param primColor The color of the nearest primitive intersection.
 * @return The accumulated color after ray marching through the volume.
 */
public float4 volumeSample(float2 uv, RendererUniform uniforms, float tMax, float4 primColor)
{
    // TODO: Student implementation starts here.
    Ray ray = uniforms.camera.generateRay(uv);
    Ray localRay = ray.transform(uniforms.volume.invModelMatrix);

    localRay.tRange.y = tMax;

    float4 color = accumulateColor<1024>(
        localRay,
        uniforms.volume.bound,
        MAX_STEPS,
        STEP_SIZE,
        DENSITY_SCALE,
        primColor,
        uniforms.volume.tex
    );

    return color;
    
    // TODO: Student implementation ends here.
}

/**
 * Backward pass for volumetric ray marching at a single uv.
 * @param uv The normalized uv coordinate from (0, 0) in the bottom-left to (1, 1) in the top-right.
 * @param uniforms The renderer uniform containing camera, sampling, and scene information.
 * @param outGrad The output gradient w.r.t. the sampled color.
 */
public void volumeSampleBwd(float2 uv, RendererUniform uniforms, float4 outGrad)
{
    // TODO: Student implementation starts here.

    // get tMax and primColor
    uint2 tid = uint2(uv * float2(uniforms.camera.canvasSize));
    RNG rng = RNG(tid, (tid.y * uniforms.camera.canvasSize.x + tid.x) * 10000 + uniforms.seed);    
    Optional<RayMeshSampleResult> rayMeshSampleRes = meshSample(uv, uniforms, rng);

    float4 primColor = uniforms.ambientColor;
    float tMax = float.maxValue;
    if (rayMeshSampleRes.hasValue)
    {
        primColor = rayMeshSampleRes.value.color;
        tMax = rayMeshSampleRes.value.t;
    }

    // generate ray and set tRange
    Ray ray = uniforms.camera.generateRay(uv);
    Ray localRay = ray.transform(uniforms.volume.invModelMatrix);
    localRay.tRange.y = tMax;

    bwd_diff(accumulateColor<1024>)(
        localRay,
        uniforms.volume.bound,
        MAX_STEPS,
        STEP_SIZE,
        DENSITY_SCALE,
        primColor,
        DifferentialPtrPair<DiffTexture3D<float, 4>>(uniforms.volume.tex, uniforms.volume.dTex),
        outGrad
    );

    // TODO: Student implementation ends here.
}

/**
 * Ray march through the volume and accumulate color.
 * @param ray The ray in local space.
 * @param volumeBound The bounding box of the volume in local space.
 * @param maxSteps The maximum number of marching steps.
 * @param stepSize The size of each marching step.
 * @param densityScale The scale factor for the volume density (alpha channel).
 * @param backgroundColor The background color behind the volume
                          (this can be the ambient color if there's no primitive behind the volume, or the color of the primitive hit if there's one).
 * @param data The 3D texture storing the volume data (rgba).
 * @return The accumulated color after ray marching through the volume.
 */
[Differentiable]
public float4 accumulateColor<let N : uint>(
    no_diff Ray ray,
    no_diff BoundingBox volumeBound,
    no_diff uint maxSteps,
    no_diff float stepSize,
    no_diff float densityScale,
    no_diff float4 backgroundColor,
    DiffTexture3D<float, 4> data)
{
    // TODO: Student implementation starts here.
    float2 tHit = volumeBound.hit(ray);

    // ray intersect with bounding box
    if (tHit.x > tHit.y) {
        return backgroundColor;
    }

    float tMin = max(tHit.x, 0.0);
    float tMax = min(tHit.y, ray.tRange.y) - EPSILON;

    float t = tMin;
    float3 accColor = float3(0.0);
    float accAlpha = 0.0;

    [MaxIters(N)]
    for (uint i = 0; i < maxSteps; i++) {
        if (t >= tMax || accAlpha >= 0.99) {
            break;
        }
        // march along ray
        float3 position = ray.origin + t * ray.direction;
        // map local position to texture coordinates
        float3 uvw = (position - volumeBound.pMin) / (volumeBound.pMax - volumeBound.pMin);
        // trilinear sampling
        float4 sample = sampleTrilinear(data, uvw);
        // compute opacity
        float stepAlpha = 1.0 - exp(-sample.w * densityScale * stepSize);
        // accumulate absortion and emission
        accColor += (1.0 - accAlpha) * sample.xyz * stepAlpha;
        accAlpha += (1.0 - accAlpha) * stepAlpha;
        
        t += stepSize;
    }

    float3 finalColor = accColor + (1.0 - accAlpha) * backgroundColor.xyz * backgroundColor.w;
    float finalAlpha = accAlpha + (1.0 - accAlpha) * backgroundColor.w;
    return float4(finalColor, finalAlpha);

    // TODO: Student implementation ends here.
}
