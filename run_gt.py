# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Standard library imports
import math
import time
import os
import os.path as osp
from pathlib import Path

# Third-party imports
import numpy as np
import sgl
from slangpy import Module
from slangpy.backend import Device, DeviceType, TextureLoader
from slangpy.types import NDBuffer
from main import Timer

# Local application imports
from app import App

def run_groundtruth(input_path, output_path, mipmap_level: float = 0):
    resolution = 512 # depends on the resolution of the input image
    timer = Timer()
    
    app = App("Neural Texture", device_type=DeviceType.vulkan, width=resolution, height=resolution)
    device = app.device
    # For headless training, bypass the App class and create the device directly
    # device = slangpy.create_device(DeviceType.vulkan, include_paths=[Path(__file__).parent.parent])
    module = Module.load_from_file(device, "NeuralTexture.slang")
    
    timer.start()
    loader = TextureLoader(device)
    target_tex = loader.load_texture(f"{input_path}", {"load_as_normalized": True, "generate_mips": True})
    sampler = device.create_sampler(min_lod=0, max_lod=7)
    
    uv_grid = create_uv_grid(device, resolution)
    module.sampleMip(uv_grid, target_tex, sampler, mipmap_level, _result=app.output)
    timer.stop()
    
    print(f"Time: {timer.elapsed() * 1e3:.1f}ms")
    
    bitmap = app.output.to_bitmap()
    bitmap.convert(
        sgl.Bitmap.PixelFormat.rgb,
        sgl.Bitmap.ComponentType.uint8,
        srgb_gamma=True
    ).write(f'{output_path}')
    print(f"Output image saved to {output_path}")

def create_uv_grid(device: Device, resolution: int):
    span = np.linspace(0, 1, resolution, dtype=np.float32)
    uvs_np = np.stack(np.broadcast_arrays(span[None, :], span[:, None]), axis=2)
    uvs = NDBuffer(device, 'float2', shape=(resolution, resolution))
    uvs.copy_from_numpy(uvs_np)
    return uvs

"""
Scripts for sampling texture in the traditional way
xvfb-run python run_baseline.py --input inputs/bernie.jpg --output outputs/groundtruth/gt_bernie$mipmap_level_float.png/ --mipmap_level $$mipmap_level_float
"""
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Neural Texture Generator")
    parser.add_argument("--input", default="input.png",
                        help="Path for the input image in inference mode")
    parser.add_argument("--output", default="output.png",
                        help="Path for the output image in inference mode")
    parser.add_argument("--resolution", type=int, default=512,
                        help="Resolution of the output image")
    parser.add_argument("--mipmap_level", type=float, default=0,
                        help="Mipmap level")
    
    args = parser.parse_args()
    run_groundtruth(args.input, args.output, args.mipmap_level)