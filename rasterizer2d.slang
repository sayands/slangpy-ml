// SPDX-License-Identifier: Apache-2.0

import "slangpy";
__exported import network.NeuralModules;
// __exported import network.Activations;
// __exported import network.MLP;
// __exported import network.Encodings;
// __exported import network.Losses;

// Has to be an include rather than an import right now
// Otherwise, __init is not exposed to reflection
// #include "network/Optimizer.slang"

struct Camera
{
    float3 o;          // Camera position in world space
    float3 look_at;    // Point the camera is looking at (added to match Python)
    float3 up;         // Up direction (added to match Python)
    float3 scale;      // World scale
    float2 frameDim;   // Frame dimensions (i.e. image resolution)
    float near;        // Near clipping plane
    float far;         // Far clipping plane
    float fov;         // Field of view (in radians)

    // Perspective projection matrix
    float4x4 getPerspectiveMatrix()
    {
        float aspect = frameDim.x / frameDim.y;
        float tanHalfFov = tan(fov / 2.0);

        float4x4 proj = float4x4(
            1.0 / (aspect * tanHalfFov), 0.0, 0.0, 0.0,
            0.0, 1.0 / tanHalfFov, 0.0, 0.0,
            0.0, 0.0, -(far + near) / (far - near), -1.0,
            0.0, 0.0, -(2.0 * far * near) / (far - near), 0.0
        );

        return transpose(proj);
    }

    // View matrix (world-to-camera transformation)
    float4x4 getViewMatrix()
    {
        // Use look_at and up to create the view matrix
        float3 forward = normalize(look_at - o);  // Direction from camera to look_at
        float3 right = cross(up, forward);  // Use provided up vector
        float3 corrected_up = cross(forward, right);   // Ensure orthogonality

        float4x4 view = float4x4(
            right.x, corrected_up.x, -forward.x, 0.0,
            right.y, corrected_up.y, -forward.y, 0.0,
            right.z, corrected_up.z, -forward.z, 0.0,
            -dot(right, o), -dot(corrected_up, o), dot(forward, o), 1.0
        );

        return transpose(view);
    }
};

float2 worldToScreen(float3 worldPos, Camera camera)
{
    // Convert to homogeneous coordinates
    float4 pos = float4(worldPos, 1.0);
    
    // Apply view and projection transformations
    float4x4 view = camera.getViewMatrix();
    float4x4 proj = camera.getPerspectiveMatrix();
    
    float4 viewPos = mul(view, pos);
    float4 clipPos = mul(proj, viewPos);
    
    // Perform perspective divide to get NDC coordinates
    float3 ndcPos = clipPos.xyz / clipPos.w;

    // Check NDC bounds for clipping
    if (ndcPos.x < -1.0 || ndcPos.x > 1.0 || 
        ndcPos.y < -1.0 || ndcPos.y > 1.0 || 
        ndcPos.z < -1.0 || ndcPos.z > 1.0)
    {
        // Point is outside the view frustum, return invalid position
        return float2(-1.0, -1.0);
    }
    
    // Convert NDC coordinates to screen coordinates
    float2 screenPos = float2(
        (ndcPos.x + 1.0) * 0.5 * camera.frameDim.x,
        (1.0 + ndcPos.y) * 0.5 * camera.frameDim.y
    );

    return screenPos;
}

// Simple point-in-triangle test using barycentric coordinates
bool isPointInTriangle(float2 p, float2 a, float2 b, float2 c, out float3 barycentric)
{
    // Compute vectors
    float2 v0 = c - a;
    float2 v1 = b - a;
    float2 v2 = p - a;
    
    // Compute dot products
    float dot00 = dot(v0, v0);
    float dot01 = dot(v0, v1);
    float dot02 = dot(v0, v2);
    float dot11 = dot(v1, v1);
    float dot12 = dot(v1, v2);
    
    // Compute barycentric coordinates
    float invDenom = 1.0 / (dot00 * dot11 - dot01 * dot01);
    float u = (dot11 * dot02 - dot01 * dot12) * invDenom;
    float v = (dot00 * dot12 - dot01 * dot02) * invDenom;
    float w = 1.0 - u - v;
    
    barycentric = float3(w, u, v);

     // Check if point is in triangle
    return (u >= 0) && (v >= 0) && (u + v <= 1);
}

// Rasterize function called from Python
float4 rasterize(Camera camera, float3 vertices[][3], float2 uvs[][3], int numTriangles, Texture2D<float4> texture, SamplerState sampler, int2 pixelCoord)
{
    // Convert pixel coordinate to screen space
    float2 point = float2(pixelCoord.x, pixelCoord.y);
    point.y = camera.frameDim.y - point.y;  // Flip y-axis
    
    // Default background color is black
    float4 outColor = float4(0.0, 0.0, 0.0, 1.0);
    
    // Loop through all triangles
    for (int i = 0; i < numTriangles; i++) {
        // Get world space vertices for current triangle
        float3 v0 = vertices[i][0];
        float3 v1 = vertices[i][1];
        float3 v2 = vertices[i][2];

        float2 uv0 = uvs[i][0];
        float2 uv1 = uvs[i][1];
        float2 uv2 = uvs[i][2];
        
        // Convert world space vertices to screen space
        float2 s0 = worldToScreen(v0, camera);
        float2 s1 = worldToScreen(v1, camera);
        float2 s2 = worldToScreen(v2, camera);

        
        // Check if current pixel is inside the triangle
        float3 barycentric;
        if (isPointInTriangle(point, s0, s1, s2, barycentric)) {
            // Color the first triangle red, the second triangle blue
            float2 uv = uv0 * barycentric.x + uv1 * barycentric.y + uv2 * barycentric.z;
            outColor = texture.Sample(sampler, uv);
            break;
        }
    }
    
    return outColor;
}

float4 evalNeuralTexture<Model : IModule<float, 2, 3>>(Model model, float2 inputUV)
{
    let prediction = model.forward(inputUV);
    return float4(prediction, 1.0f);
}
