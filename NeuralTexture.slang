// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// clang-format off

__exported import network.NeuralModules;
__exported import network.Activations;
__exported import network.MLP;
__exported import network.Encodings;
__exported import network.Losses;

// Has to be an include rather than an import right now
// Otherwise, __init is not exposed to reflection
#include "network/Optimizer.slang"

struct RNG
{
    uint state;

    __init(uint state) { this.state = state; }

    [mutating]
    uint next()
    {
        state = state * 2739110765U + 2739110765U;
        return state;
    }

    [mutating]
    float next1D()
    {
        // Use upper 24 bits and divide by 2^24 to get a number u in [0,1).
        // In floating-point precision this also ensures that 1.0-u != 0.0.
        uint bits = next();
        return (bits >> 8) * 0x1p-24;
    }

    [mutating]
    float2 next2D()
    {
        return { next1D(), next1D() };
    }
}

float4 evalModel<Model : IModule<float, 2, 3>>(Model model, float2 inputUV)
{
    let prediction = model.forward(inputUV);
    return float4(prediction, 1.0f);
}

float4 residualEvalModel<PrevModel : IModule<float, 2, 3>, Model : IModule<float, 5, 3>>(Model model, PrevModel prev_model, float2 inputUV)
{
    float3 color = prev_model.forward(inputUV);
    vector<float, 5> input_res = { inputUV[0], inputUV[1], color[0], color[1], color[2] };
    float3 prediction = color + float3(model.forward(input_res));

    return float4(prediction, 1.0f);
}

[BackwardDifferentiable]
float evalModelAndLoss<Model : IModule<float, 2, 3>>(Model model, no_diff float2 input, no_diff float3 target)
{
    let lossFunc = L2Loss();
    float3 prediction = float3(model.forward(input));

    return lossFunc.eval(prediction, target);
}

void trainTexture<Model : IModule<float, 2, 3>>(Model model, inout RNG rng, Texture2D<float4> targetTex, SamplerState sampler, float mipmapLevel, float lossScale)
{
    float2 inputUV = rng.next2D();
    float4 target = targetTex.SampleLevel(sampler, inputUV, float(mipmapLevel));

    bwd_diff(evalModelAndLoss)(model, inputUV, target.rgb, lossScale);
}

[BackwardDifferentiable]
float residualEvalModelAndLoss<Model : IModule<float, 5, 3>, PrevModel : IModule<float, 2, 3>>(Model model, no_diff PrevModel prev_model, no_diff float2 inputUV, no_diff float3 target)
{
    let lossFunc = L2Loss();
    float3 model_input = prev_model.forward(inputUV);
    vector<float, 5> input_res = { inputUV[0], inputUV[1], model_input[0], model_input[1], model_input[2] };
    float3 prediction = model_input + float3(model.forward(input_res));

    return lossFunc.eval(prediction, target);
}

void residualTrainTexture<PrevModel : IModule<float, 2, 3>, Model : IModule<float, 5, 3>>(Model model, PrevModel prev_model, inout RNG rng, Texture2D<float4> targetTex, SamplerState sampler, float lossScale)
{
    float2 inputUV = rng.next2D();
    float4 target = targetTex.SampleLevel(sampler, inputUV, 0.0);

    bwd_diff(residualEvalModelAndLoss)(model, prev_model, inputUV, target.rgb, lossScale);
}

float4 evalMipLevel<Model : IModule<float, 2, 3>>(Model model_floor, Model model_ceil, float2 inputUV, float alpha)
{
    let prediction_floor = model_floor.forward(inputUV);
    let prediction_ceil = model_ceil.forward(inputUV);

    let prediction = lerp(prediction_floor, prediction_ceil, alpha);
    return float4(prediction, 1.0f);
}

float4 sampleMip(float2 inputUV, Texture2D<float4> targetTex, SamplerState sampler, float mipmapLevel)
{
    float4 target = targetTex.SampleLevel(sampler, inputUV, mipmapLevel);
    return float4(target.rgb, 1.0f);
}


float4 renderSphere<Model : IModule<float, 2, 3>>(Model model, float2 uv_grid, float zoom_factor, int2 coord) {
    // Check bounds
    // if (coord.x >= width) return float4(0, 0, 0, 1); // Out of bounds, return black

    // Get base UV from the grid (normalized [0, 1])
    float2 base_uv = uv_grid;

    // Apply zoom: scale UVs around center (0.5, 0.5)
    float2 center = float2(0.5, 0.5);
    float2 uv = center + (base_uv - center) / zoom_factor;

    // Clamp UVs to [0, 1] to avoid out-of-bounds sampling
    uv = clamp(uv, float2(0.0, 0.0), float2(1.0, 1.0));

    // Evaluate neural model and return color
    return evalModel(model, uv);
}


float4 renderTex(float2 uv_grid, float rotation_angle, float zoom_factor,  Texture2D<float4> targetTex, SamplerState sampler, int2 coord) {
    // Check bounds
    // if (coord.x >= width) return float4(0, 0, 0, 1); // Out of bounds, return black

    // Get base UV from the grid (normalized [0, 1])
    float2 base_uv = uv_grid;

    // Apply zoom: scale UVs around center (0.5, 0.5)
    float2 center = float2(0.5, 0.5);
    float2 uv = center + (base_uv - center) / zoom_factor;

    // Clamp UVs to [0, 1] to avoid out-of-bounds sampling
    uv = clamp(uv, float2(0.0, 0.0), float2(1.0, 1.0));

    // Evaluate neural model and return color
    return targetTex.SampleLevel(sampler, uv, 0.0);
}