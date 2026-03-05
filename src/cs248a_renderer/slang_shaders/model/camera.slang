implementing "../model.slang";
import math;

public struct Camera
{
    // The inverse view matrix of the camera (matrix transform from view space to world space).
    public float4x4 invViewMatrix;
    // The canvas size of the target image in pixels.
    public uint2 canvasSize;
    // The focal length of the camera in pixels (distance between the pinhole and the image plane).
    public float focalLength;

    /**
     * Generate a camera ray given a normalized uv coordinate on canvas.
     * @param uv The normalized uv coordinate from (0, 0) in the bottom-left to (1, 1) in the top-right.
     * @return The generated Ray. Before you start working the transform section, generate ray as if the camera is at origin looking towards negative z direction.
     * After you finish the transform section, you should transform the ray from view space to world space using the inverse view matrix.
     */
    public Ray generateRay(float2 uv)
    {
        // TODO: Student implementation starts here.
        float2 W_H = float2(this.canvasSize);

        // pixel center coordinates (in pixels)
        float2 p = uv * W_H;
        float2 c = 0.5 * W_H;

        // View-space ray (focalLength is in pixels)
        float3 dirView = normalize(float3((p.x - c.x) / this.focalLength,
                                        (p.y - c.y) / this.focalLength,
                                        -1.0));

        // Transform to world
        float3 originWorld = mul(this.invViewMatrix, float4(0.0, 0.0, 0.0, 1.0)).xyz;
        float3 dirWorld    = normalize(mul(this.invViewMatrix, float4(dirView, 0.0)).xyz);

        return Ray(originWorld, dirWorld, float2(1e-3, float.maxValue));

        // TODO: Student implementation ends here.
    }
}
