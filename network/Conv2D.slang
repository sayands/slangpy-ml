// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import NeuralModules;

struct Conv2DLayer<T : IScalar, 
                int InputChannels, 
                int InputHeight, 
                int InputWidth,
                int OutputChannels,
                int KernelHeight,
                int KernelWidth,
                int Stride = 1,
                int Padding = 0> 
    : IModule<T, InputChannels * InputHeight * InputWidth, 
              OutputChannels * ((InputHeight + 2*Padding - KernelHeight)/Stride + 1) * 
                                    ((InputWidth + 2*Padding - KernelWidth)/Stride + 1)>
    where T.Differential : IArithmeticAtomicable
{
    // Weight layout: [OutputChannels][InputChannels][KernelHeight][KernelWidth]
    StructuredBuffer<T> weights;
    
    // Bias layout: [OutputChannels]
    StructuredBuffer<T> biases;
    
    // Gradient buffers
    RWStructuredBuffer<Atomic<T.Differential>> weightGrads;
    RWStructuredBuffer<Atomic<T.Differential>> biasGrads;

    static int OutputHeight = (InputHeight + 2*Padding - KernelHeight) / Stride + 1;
    static int OutputWidth = (InputWidth + 2*Padding - KernelWidth) / Stride + 1;

    [BackwardDerivative(backward)]
    T[OutputChannels * OutputHeight * OutputWidth] forward(T x[InputChannels * InputHeight * InputWidth])
    {
        T y[OutputChannels * OutputHeight * OutputWidth];
        
        for (int oc = 0; oc < OutputChannels; oc++)
        {
            for (int oh = 0; oh < OutputHeight; oh++)
            {
                for (int ow = 0; ow < OutputWidth; ow++)
                {
                    T sum = biases[oc];
                    
                    for (int ic = 0; ic < InputChannels; ic++)
                    {
                        [ForceUnroll]
                        for (int kh = 0; kh < KernelHeight; kh++)
                        {
                            [ForceUnroll]
                            for (int kw = 0; kw < KernelWidth; kw++)
                            {
                                int ih = oh * Stride - Padding + kh;
                                int iw = ow * Stride - Padding + kw;
                                
                                if (ih >= 0 && ih < InputHeight && iw >= 0 && iw < InputWidth)
                                {
                                    int inputIdx = ic * InputHeight * InputWidth + ih * InputWidth + iw;
                                    int weightIdx = oc * (InputChannels * KernelHeight * KernelWidth) 
                                                  + ic * (KernelHeight * KernelWidth) 
                                                  + kh * KernelWidth 
                                                  + kw;
                                    
                                    sum += weights[weightIdx] * x[inputIdx];
                                }
                            }
                        }
                    }
                    
                    int outputIdx = oc * OutputHeight * OutputWidth + oh * OutputWidth + ow;
                    y[outputIdx] = sum;
                }
            }
        }
        return y;
    }

    void backward(inout DifferentialPair<T[InputChannels * InputHeight * InputWidth]> x, 
                T.Differential grad[OutputChannels * OutputHeight * OutputWidth])
    {
        T.Differential dx[InputChannels * InputHeight * InputWidth] = T.Differential.dzero();

        // Accumulate bias gradients
        for (int oc = 0; oc < OutputChannels; oc++)
        {
            T.Differential bGrad = T.Differential.dzero();
            for (int oh = 0; oh < OutputHeight; oh++)
            {
                for (int ow = 0; ow < OutputWidth; ow++)
                {
                    int outputIdx = oc * OutputHeight * OutputWidth + oh * OutputWidth + ow;
                    bGrad = T.dadd(bGrad, grad[outputIdx]);
                }
            }
            biasGrads[oc].add(bGrad);
        }

        // Accumulate weight gradients and compute input gradients
        for (int oc = 0; oc < OutputChannels; oc++)
        {
            for (int ic = 0; ic < InputChannels; ic++)
            {
                [ForceUnroll]
                for (int kh = 0; kh < KernelHeight; kh++)
                {
                    [ForceUnroll]
                    for (int kw = 0; kw < KernelWidth; kw++)
                    {
                        T.Differential wGrad = T.Differential.dzero();
                        
                        for (int oh = 0; oh < OutputHeight; oh++)
                        {
                            for (int ow = 0; ow < OutputWidth; ow++)
                            {
                                int ih = oh * Stride - Padding + kh;
                                int iw = ow * Stride - Padding + kw;
                                
                                if (ih >= 0 && ih < InputHeight && iw >= 0 && iw < InputWidth)
                                {
                                    int inputIdx = ic * InputHeight * InputWidth + ih * InputWidth + iw;
                                    int outputIdx = oc * OutputHeight * OutputWidth + oh * OutputWidth + ow;
                                    int weightIdx = oc * (InputChannels * KernelHeight * KernelWidth) 
                                                  + ic * (KernelHeight * KernelWidth) 
                                                  + kh * KernelWidth 
                                                  + kw;

                                    // Accumulate weight gradient
                                    wGrad = T.dadd(wGrad, T.dmul(x.p[inputIdx], grad[outputIdx]));
                                    
                                    // Accumulate input gradient
                                    dx[inputIdx] = T.dadd(dx[inputIdx], 
                                                         T.dmul(weights[weightIdx], grad[outputIdx]));
                                }
                            }
                        }
                        
                        weightGrads[oc * (InputChannels * KernelHeight * KernelWidth) 
                                  + ic * (KernelHeight * KernelWidth) 
                                  + kh * KernelWidth 
                                  + kw].add(wGrad);
                    }
                }
            }
        }

        x = diffPair(x.p, dx);
    }
}