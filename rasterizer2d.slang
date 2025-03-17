// SPDX-License-Identifier: Apache-2.0

import "slangpy";
__exported import network.NeuralModules;
__exported import network.Activations;
__exported import network.MLP;
__exported import network.Encodings;

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

float2 worldToScreen(float3 worldPos, Camera camera, out float w)
{
    // Convert to homogeneous coordinates
    float4 pos = float4(worldPos, 1.0);
    
    // Apply view and projection transformations
    float4x4 view = camera.getViewMatrix();
    float4x4 proj = camera.getPerspectiveMatrix();
    
    float4 viewPos = mul(view, pos);
    float4 clipPos = mul(proj, viewPos);

    w = clipPos.w;
    
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

// Compute perspective-correct UV gradients
void computeUVGradients(
    float2 s0, float2 s1, float2 s2,  // Screen-space positions
    float2 uv0, float2 uv1, float2 uv2,  // UV coordinates
    float w0, float w1, float w2,  // Clip-space w values
    int2 pixelCoord,  // Current pixel coordinate
    Camera camera,  // Camera parameters
    out float2 du, out float2 dv   // Gradients: du/dx, du/dy, dv/dx, dv/dy
)
{
    // Perspective-correct terms
    float u0_over_w = uv0.x / w0;
    float u1_over_w = uv1.x / w1;
    float u2_over_w = uv2.x / w2;
    float v0_over_w = uv0.y / w0;
    float v1_over_w = uv1.y / w1;
    float v2_over_w = uv2.y / w2;
    float inv_w0 = 1.0 / w0;
    float inv_w1 = 1.0 / w1;
    float inv_w2 = 1.0 / w2;

    // Screen-space differences
    float dx1 = s1.x - s0.x;
    float dy1 = s1.y - s0.y;
    float dx2 = s2.x - s0.x;
    float dy2 = s2.y - s0.y;

    float du1_over_w = u1_over_w - u0_over_w;
    float du2_over_w = u2_over_w - u0_over_w;
    float dv1_over_w = v1_over_w - v0_over_w;
    float dv2_over_w = v2_over_w - v0_over_w;
    float dinv_w1 = inv_w1 - inv_w0;
    float dinv_w2 = inv_w2 - inv_w0;

    float det = dx1 * dy2 - dx2 * dy1;
    if (abs(det) < 1e-6) {
        du = float2(0.0, 0.0);
        dv = float2(0.0, 0.0);
        return;
    }

    float invDet = 1.0 / det;

    // Gradients of u/w, v/w, 1/w in screen space
    float du_over_w_dx = (du1_over_w * dy2 - du2_over_w * dy1) * invDet;
    float du_over_w_dy = (dx1 * du2_over_w - dx2 * du1_over_w) * invDet;
    float dv_over_w_dx = (dv1_over_w * dy2 - dv2_over_w * dy1) * invDet;
    float dv_over_w_dy = (dx1 * dv2_over_w - dx2 * dv1_over_w) * invDet;
    float dinv_w_dx = (dinv_w1 * dy2 - dinv_w2 * dy1) * invDet;
    float dinv_w_dy = (dx1 * dinv_w2 - dx2 * dinv_w1) * invDet;

    // Perspective-correct gradients using the chain rule
    // u = (u/w) / (1/w), so du/dx = (d(u/w)/dx * (1/w) - (u/w) * d(1/w)/dx) / (1/w)^2
    // For simplicity, compute at the current pixel's interpolated 1/w
    float3 barycentric;
    float2 point = pixelCoord;  // Use current pixel for approximation
    point.y = camera.frameDim.y - point.y;
    if (!isPointInTriangle(point, s0, s1, s2, barycentric))
    {
        du = float2(0.0, 0.0);
        dv = float2(0.0, 0.0);
        return;
    }
    float inv_w = inv_w0 * barycentric.x + inv_w1 * barycentric.y + inv_w2 * barycentric.z;
    float u_over_w = u0_over_w * barycentric.x + u1_over_w * barycentric.y + u2_over_w * barycentric.z;
    float v_over_w = v0_over_w * barycentric.x + v1_over_w * barycentric.y + v2_over_w * barycentric.z;
    float w = 1.0 / inv_w;

    du.x = (du_over_w_dx * inv_w - u_over_w * dinv_w_dx) * w;
    du.y = (du_over_w_dy * inv_w - u_over_w * dinv_w_dy) * w;
    dv.x = (dv_over_w_dx * inv_w - v_over_w * dinv_w_dx) * w;
    dv.y = (dv_over_w_dy * inv_w - v_over_w * dinv_w_dy) * w;
}

// Rasterize function called from Python
float4 rasterize<Model : IModule<float, 2, 3>>(Camera camera, float3 vertices[][3], float2 uvs[][3], int numTriangles, Texture2D<float4> texture, SamplerState sampler, Model model, int2 pixelCoord)
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

        float w0, w1, w2;
        
        // Convert world space vertices to screen space
        float2 s0 = worldToScreen(v0, camera, w0);
        float2 s1 = worldToScreen(v1, camera, w1);
        float2 s2 = worldToScreen(v2, camera, w2);

        
        // Check if current pixel is inside the triangle
        float3 barycentric;
        if (isPointInTriangle(point, s0, s1, s2, barycentric)) {
            // Color the first triangle red, the second triangle blue
            // Perspective-correct UV interpolation
            float u0_over_w = uv0.x / w0;
            float u1_over_w = uv1.x / w1;
            float u2_over_w = uv2.x / w2;
            float v0_over_w = uv0.y / w0;
            float v1_over_w = uv1.y / w1;
            float v2_over_w = uv2.y / w2;
            float inv_w0 = 1.0 / w0;
            float inv_w1 = 1.0 / w1;
            float inv_w2 = 1.0 / w2;

            float u_over_w = u0_over_w * barycentric.x + u1_over_w * barycentric.y + u2_over_w * barycentric.z;
            float v_over_w = v0_over_w * barycentric.x + v1_over_w * barycentric.y + v2_over_w * barycentric.z;
            float inv_w = inv_w0 * barycentric.x + inv_w1 * barycentric.y + inv_w2 * barycentric.z;
            float2 uv = float2(u_over_w, v_over_w) / inv_w;

            // Compute perspective-correct gradients
            float2 du, dv;
            computeUVGradients(s0, s1, s2, uv0, uv1, uv2, w0, w1, w2, pixelCoord, camera, du, dv);
            
            // Compute LOD correctly
            int2 texSize = int2(256, 256);
            float2 texSizeF = float2(texSize);
            
            // Texture-space gradients per screen pixel
            float2 dUV_dx = float2(du.x * texSizeF.x, dv.x * texSizeF.y);  // Change in (u,v) per x-pixel
            float2 dUV_dy = float2(du.y * texSizeF.x, dv.y * texSizeF.y);  // Change in (u,v) per y-pixel
            
            // Length of texture-space gradient in each screen direction
            float len_dx = length(dUV_dx);
            float len_dy = length(dUV_dy);
            
            // LOD is log2 of the maximum sampling rate
            float lod = log2(max(len_dx, len_dy));
            lod = clamp(lod, 0.0, float(8 - 1));
            int floor_lod = int(floor(lod));
            int ceil_lod = int(ceil(lod));
            float alpha = lod - floor_lod;

            // outColor = evalMipLevel(models[floor_lod], models[ceil_lod], uv, alpha);
            // outColor = texture.SampleLevel(sampler, uv, lod);
            outColor = evalNeuralTexture(model, uv);
            // Debug: Visualize gradients (optional)
            // outColor = float4(lod / 7.0, 0.0, 0.0, 1.0);
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


float4 evalMipLevel<Model : IModule<float, 2, 3>>(Model model_floor, Model model_ceil, float2 inputUV, float alpha)
{
    let prediction_floor = model_floor.forward(inputUV);
    let prediction_ceil = model_ceil.forward(inputUV);

    let prediction = lerp(prediction_floor, prediction_ceil, alpha);
    return float4(prediction, 1.0f);
}