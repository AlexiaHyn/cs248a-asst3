implementing "../renderer.slang";

import utils;

public float2 getShiftedTexUV(float2 uv, RendererUniform uniforms, Triangle triangle, float2 delta, float2 tex_uv)
{
    float2 shifted_uv = uv + delta;
    Ray ray = uniforms.camera.generateRay(shifted_uv);
    Optional<RayHitResult> hitResult = triangle.hit(ray);
    float2 result = tex_uv;
    if (hitResult.hasValue)
    {
        float3 hitPoint = ray.origin + hitResult.value.t * ray.direction;
        float3 barycentricCoord = triangle.calculateBarycentricCoord(hitPoint);
        return triangle.getUV(barycentricCoord);
    }

    Optional<RayMeshIntersectionResult> rayMeshIntersectionResult_dx = rayMeshIntersection(ray, uniforms);
    if (rayMeshIntersectionResult_dx.hasValue)
    {
        Triangle hitTriangle = rayMeshIntersectionResult_dx.value.hitTriangle;
        if (hitTriangle.getMaterialId() == triangle.getMaterialId())
        {
            float3 hitPoint = ray.origin + rayMeshIntersectionResult_dx.value.closestHit.t * ray.direction;
            float3 barycentricCoord = hitTriangle.calculateBarycentricCoord(hitPoint);
            return hitTriangle.getUV(barycentricCoord);
        }
    }

    return tex_uv;
}

public float getLevel(float2 uv, RendererUniform uniforms, Triangle triangle, float2 tex_uv)
{
    float level = 0.0;

    // TODO: Student implementation starts here.
    float2 pixelOffset = 1.0 / float2(uniforms.camera.canvasSize);

    Ray ray_dx = uniforms.camera.generateRay(uv + float2(pixelOffset.x, 0.0));
    Ray ray_dy = uniforms.camera.generateRay(uv + float2(0.0, pixelOffset.y));
    Optional<RayMeshIntersectionResult> hit_dx = rayMeshIntersection(ray_dx, uniforms);
    Optional<RayMeshIntersectionResult> hit_dy = rayMeshIntersection(ray_dy, uniforms);
    float2 tex_uv_dx = tex_uv;
    float2 tex_uv_dy = tex_uv;

    if (hit_dx.hasValue)
    {
        RayHitResult r = hit_dx.value.closestHit;
        Triangle ttri = hit_dx.value.hitTriangle;
        float3 p = ray_dx.origin + r.t * normalize(ray_dx.direction);
        float3 bc = ttri.calculateBarycentricCoord(p);
        tex_uv_dx = ttri.getUV(bc);
    }

    if (hit_dy.hasValue)
    {
        RayHitResult r = hit_dy.value.closestHit;
        Triangle ttri = hit_dy.value.hitTriangle;
        float3 p = ray_dy.origin + r.t * normalize(ray_dy.direction);
        float3 bc = ttri.calculateBarycentricCoord(p);
        tex_uv_dy = ttri.getUV(bc);
    }

    uint materialId = triangle.materialId + triangle.offset;
    float2 texSize = float2(uniforms.physicsBasedMaterialBuf[materialId].albedo.mipmap[0].size);

    float2 du = (tex_uv_dx - tex_uv) * texSize;
    float2 dv = (tex_uv_dy - tex_uv) * texSize;

    float rho = max(length(du), length(dv));
    return log2(max(rho, 1.0));
    
    // TODO: Student implementation ends here.

    return level;
}

public PhysicsBasedParameters getHitPointParameters(float2 uv, RendererUniform uniforms, Triangle triangle, float3 normal, float3 barycentricCoord, float3 hitPoint)
{
    float2 tex_uv = triangle.getUV(barycentricCoord);
    float level = getLevel(uv, uniforms, triangle, tex_uv);
    uint materialId = triangle.materialId + triangle.offset;
    PhysicsBasedMaterial physicsBasedMaterial = uniforms.physicsBasedMaterialBuf[materialId];
    PhysicsBasedParameters parameters = sampleMaterial(physicsBasedMaterial, uniforms.physicsBasedMaterialTextureBuf, tex_uv, level, uniforms.visualizeLevelOfDetail);
    if (uniforms.visualizeBarycentricCoords)
    {
        parameters.albedo = barycentricCoord;
    }
    if (uniforms.visualizeTexUV)
    {
        parameters.albedo = float3(tex_uv, 0.0);
    }
    return parameters;
}
