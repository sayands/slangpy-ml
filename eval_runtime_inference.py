# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Standard library imports
import math
import time
import os
import os.path as osp
from pathlib import Path
import matplotlib.pyplot as plt

# Third-party imports
import numpy as np
import sgl
import slangpy
from slangpy import Module
from slangpy.backend import DataType, Device, DeviceType, TextureLoader
from slangpy.types import NDBuffer

# Local application imports
from app import App
from network import (
    Activation, ELUAct, ExpAct, FrequencyEncoding, LeakyReLUAct, 
    LinearLayer, ModuleChain, NoneAct, ReLUAct, SigmoidAct, 
    SwishAct, TanhAct
)

from utils import model_util, timer, uv_util

def inference_main(model_path, resolution):
    _timer = timer.Timer()
    
    app = App("Neural Texture Inference", device_type=DeviceType.vulkan, width=resolution, height=resolution)
    device = app.device
    
    # Load the module for evaluation
    module = Module.load_from_file(device, "NeuralTexture.slang")
    # Create a UV grid for evaluation, just like in training
    uv_grid = uv_util.create_uv_grid(app.device, resolution)
    app.device.wait()
    
    _timer.start()
    
    model = ModuleChain(
        FrequencyEncoding(2, 5),
        LinearLayer(20, 64),
        LeakyReLUAct(64),
        LinearLayer(64, 64),
        LeakyReLUAct(64),
        LinearLayer(64, 64),
        LeakyReLUAct(64),
        LinearLayer(64, 3),
        SigmoidAct(3)
    )
    
    # Initialize the model (allocate storage for parameters)
    model.initialize(device)
    model_util.load_model_weights(model, model_path)
    
    # Evaluate the model once to generate the texture
    module.evalModel(model, uv_grid, _result=app.output)
    _timer.stop()
    elapsed_time = _timer.elapsed() * 1e3
    
    return elapsed_time
    
def inference_groundtruth(texture_path, resolution):
    _timer = timer.Timer()
    app = App("Neural Texture", device_type=DeviceType.vulkan, width=resolution, height=resolution)
    device = app.device

    module = Module.load_from_file(device, "NeuralTexture.slang")
    uv_grid = uv_util.create_uv_grid(app.device, resolution)
    loader = TextureLoader(device)
    
    _timer.start()
    target_tex = loader.load_texture(texture_path, {"load_as_normalized": True, "generate_mips": True})
    sampler = device.create_sampler(min_lod=0, max_lod=0)    
    module.sampleMip(uv_grid, target_tex, sampler, 0.0, _result=app.output)
    _timer.stop()
    elapsed_time = _timer.elapsed() * 1e3
    return elapsed_time

def make_plot(data_nn, data_gt, resolutions, plot_name, texture_name):
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(resolutions, data_nn, marker='o', label='Neural Network', color='blue', linestyle='-', linewidth=2)
    plt.plot(resolutions, data_gt, marker='s', label='Ground Truth', color='red', linestyle='--', linewidth=2)

    # Adding labels and title
    plt.xlabel('Resolution', fontsize=14)
    plt.ylabel('Time (ms)', fontsize=14)
    
    if 'timing' in plot_name:
        plt.title(f'Inference Time: Neural Network vs Ground Truth (on {texture_name})', fontsize=16)
    elif 'memory' in plot_name:
        plt.title(f'Memory: Neural Network vs Ground Truth (on {texture_name})', fontsize=16)
    else:
        raise NotImplementedError

    # Adding grid
    plt.grid(True, which="both", ls="--", linewidth=0.5)

    # Adding legend
    plt.legend(fontsize=12)

    # Display the plot
    plt.tight_layout()
    plt.savefig(f'{plot_name}.png')
    

if __name__ == "__main__":
    texture_name = "perlin"
    lod = 0.0
    
    resolutions = [256, 512, 1024, 2048, 4096, 8192, 16384]
    
    time_nn, time_gt = [], []
    size_nn, size_gt = [], []
    
    for resolution in resolutions:
        texture_path = f"/mnt/sdb/tejan/code/sayan_code/slangpy-ml/inputs/perlin_{resolution}.png"
        model_path = f"/mnt/sdb/tejan/code/sayan_code/slangpy-ml/checkpoints/perlin/model_{resolution}.npz"
        
        size_nn.append(os.path.getsize(model_path) / (1024 ** 2))
        size_gt.append(os.path.getsize(texture_path) / (1024 ** 2))
        
        time_nn.append(inference_main(model_path, resolution=512))
        time_gt.append(inference_groundtruth(texture_path, resolution=512))
   
    
    make_plot(size_nn, size_gt, resolutions, 'memory_plot', texture_name)
    make_plot(time_nn, time_gt, resolutions, 'timing_plot', texture_name)