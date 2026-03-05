implementing "../primitive.slang";
import math;

public struct Triangle : IBoundingBox, IRayCastable
{
    public float3[3] vertices;
    public float3[3] normals;
    public float2[3] uvs;
    public float3[3] colors;
    public uint materialId;
    public uint offset;

    public BoundingBox boundingBox()
    {
        return BoundingBox(float3(0.0), float3(0.0));
    }

    /* Returns the average color of the triangle. */
    public float3 averageColor()
    {
        return (this.colors[0] + this.colors[1] + this.colors[2]) / 3.0;
    }

    /**
     * Calculates the barycentric coordinates of a point on the triangle.
     * @param p The point on the triangle to calculate the barycentric coordinates for.
     * @return The barycentric coordinates of the point.
     */
    public float3 calculateBarycentricCoord(float3 p)
    {
        const float EPSILON = 1e-6;
        float u = 0.0;
        float v = 0.0;
        float w = 0.0;

        // TODO: Student implementation starts here.
        float3 v0 = this.vertices[0];
        float3 v1 = this.vertices[1];
        float3 v2 = this.vertices[2];
        float3 v0v1 = v1 - v0;
        float3 v0v2 = v2 - v0;
        float3 vp = p - v0;
        float3 n = cross(v0v1, v0v2);
        float areaSq = dot(n, n);
        if (areaSq < EPSILON)
        {
            return float3(1.0, 0.0, 0.0);
        }

        float3 c1 = cross(vp, v0v2);
        float3 c2 = cross(v0v1, vp);
        v = dot(n, c1) / areaSq;
        w = dot(n, c2) / areaSq;
        u = 1.0 - v - w;

        // TODO: Student implementation ends here.

        return float3(u, v, w);
    }

    public float2 getUV(float3 barycentricCoord)
    {
        float2 uv = float2(0.0);

        float u = barycentricCoord.x;
        float v = barycentricCoord.y;
        float w = barycentricCoord.z;
        uv = u * this.uvs[0] + v * this.uvs[1] + w * this.uvs[2];
        
        return uv;
    }

    public float3 getColor(float3 barycentricCoord)
    {
        float u = barycentricCoord.x;
        float v = barycentricCoord.y;
        float w = barycentricCoord.z;
        float3 color = u * this.colors[0] + v * this.colors[1] + w * this.colors[2];
        return color;
    }

    public float3 getNormal(float3 barycentricCoord)
    {
        float u = barycentricCoord.x;
        float v = barycentricCoord.y;
        float w = barycentricCoord.z;
        float3 normal = u * this.normals[0] + v * this.normals[1] + w * this.normals[2];
        return normalize(normal);
    }

    /**
     * Test ray-triangle intersection.
     * @param ray The ray to test against the triangle.
     * @return An Optional containing RayHitResult if the ray hits the triangle, otherwise an empty Optional.
     */
    public Optional<RayHitResult> hit(Ray ray)
    {
        // TODO: Student implementation starts here.

        float3 edge1 = this.vertices[1] - this.vertices[0];
        float3 edge2 = this.vertices[2] - this.vertices[0];
        float3 h = cross(ray.direction, edge2);
        float det = dot(edge1, h);
        if (det >-1e-6 && det < 1e-6)
        {
            return Optional<RayHitResult>();
        }
        float invDet = 1.0 / det;
        float3 s = ray.origin - this.vertices[0];
        float u = invDet * dot(s, h);
        if (u < -1e-6 || u > 1.0 + 1e-6)
        {
            return Optional<RayHitResult>();
        }
        float3 q = cross(s, edge1);
        float v = invDet * dot(ray.direction, q);
        if (v < -1e-6 || u + v > 1.0 + 1e-6)
        {
            return Optional<RayHitResult>();
        }
        float t = invDet * dot(edge2, q);
        if (t < ray.tRange.x || t > ray.tRange.y)
        {
            return Optional<RayHitResult>();
        }
        RayHitResult hitResult;
        hitResult.t = t;

        // Compute geometric face normal (works even if mesh normals are missing)
        float3 faceN = cross(edge1, edge2);
        float3 normalizedFaceN = normalize(faceN);
        if (dot(normalizedFaceN, ray.direction) > -1e-6)
        {
            hitResult.normal = -normalizedFaceN;
        } else 
        {
            hitResult.normal = normalizedFaceN;
        }
        // hitResult.normal = faceN;
        return Optional<RayHitResult>(hitResult);

        // TODO: Student implementation ends here.
    }

    public uint getMaterialId()
    {
        return this.materialId + this.offset;
    }
}
