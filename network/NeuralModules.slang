// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// clang-format off

typealias IScalar = __BuiltinFloatingPointType;

interface IModule<T : IScalar, int NumInputs, int NumOutputs>
{
    [BackwardDifferentiable]
    T[NumOutputs] forward(T x[NumInputs]);
}

interface IMultiModule<T : IScalar, int NumInputs, int NumOutputs, int NumChains>
{
    [BackwardDifferentiable]
    T[NumOutputs] forward(T x[NumInputs]);
}

// Chain two modules together, i.e. pass the output of the first to the second
// Can be nested arbitrarily
struct ModuleChain<
    T : IScalar,
    int NumInputs, int NumMiddle, int NumOutputs,
    First : IModule<T, NumInputs, NumMiddle>,
    Second : IModule<T, NumMiddle, NumOutputs>
> : IModule<T, NumInputs, NumOutputs>
{
    First first;
    Second second;

    [BackwardDifferentiable]
    T[NumOutputs] forward(T x[NumInputs])
    {
        let middle = first.forward(x);
        return second.forward(middle);
    }
}

// // MultiModuleChain structure which chains multiple modules together
// // First module takes original inputs. Subsequent modules take original inputs plus previous output.
// // Each module chain's last 3 outputs are expected to be r, g, b.
// // Final output is the sum of the r, g, b outputs of all module chains.
// struct MultiModuleChain<
//     T : IScalar,
//     int NumInputs, int NumOutputs, int NumChains,
//     First : IModule<T, NumInputs, NumOutputs>,
//     Second : IModule<T, NumSecondInputs, NumOutputs>,
// > : IMultiModule<T, NumInputs, NumOutputs, NumChains>
// {
//     First first;
//     Second second[NumChains-1];

//     [BackwardDifferentiable]
//     T[NumOutputs] forward(T x[NumInputs])
//     {
//         T result[NumOutputs];
        
//         // Initialize the result array to zeros
//         [ForceUnroll]
//         for (int i = 0; i < NumOutputs; ++i)
//             result[i] = T(0);
            
//         // Process each module chain
//         [ForceUnroll]
//         for (int chainIdx = 0; chainIdx < NumChains-1; ++chainIdx)
//         {
//             T chainOutput[NumOutputs];
            
//             if (chainIdx == 0)
//             {
//                 // First chain only takes original input
//                 chainOutput = first.forward(x);
//             }
//             else
//             {
//                 // Create combined input (original input + previous RGB output)
//                 T combinedInput[NumSecondInputs];
                
//                 // Copy the RGB output from previous result
//                 [ForceUnroll]
//                 for (int i = 0; i < 3; ++i)
//                     combinedInput[i] = result[i];
                
//                 // Copy the original input values
//                 [ForceUnroll]
//                 for (int i = 0; i < NumInputs; ++i)
//                     combinedInput[3 + i] = x[i];
                
//                 chainOutput = second[chainIdx].forward(combinedInput);
//             }
            
//             // Add the RGB output to the result
//             [ForceUnroll]
//             for (int i = 0; i < NumOutputs; ++i)
//                 result[i] += chainOutput[i];
//         }

       
//         return result;
//     }
// }

// Extensions: An array IModule can also be used as a vector IModule
extension<T : IScalar, int NumInputs, int NumOutputs, Module : IModule<T, NumInputs, NumOutputs>> Module
{
    [BackwardDifferentiable]
    vector<T, NumOutputs> forward(vector<T, NumInputs> x)
    {
        T xArray[NumInputs];
        [ForceUnroll]
        for (int i = 0; i < NumInputs; ++i)
            xArray[i] = x[i];

        let yArray = forward(xArray);

        vector<T, NumOutputs> y;
        [ForceUnroll]
        for (int i = 0; i < NumOutputs; ++i)
            y[i] = yArray[i];

        return y;
    }
}
