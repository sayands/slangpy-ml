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

def training_main(max_epochs=100, mipmap_level=0):
    resolution = 512 # depends on the resolution of the input image
    
    app = App("Neural Texture", device_type=DeviceType.vulkan, width=resolution, height=resolution)
    device = app.device
    # For headless training, bypass the App class and create the device directly
    # device = slangpy.create_device(DeviceType.vulkan, include_paths=[Path(__file__).parent.parent])

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
    model.initialize(device)

    module = Module.load_from_file(device, "NeuralTexture.slang")

    optimizers = [module.AdamOptimizer(p) for p in model.parameters()]

    batch_shape = (256, 256)
    learning_rate = 0.001
    grad_scale = 128.0
    loss_scale = grad_scale / math.prod(batch_shape)
    num_batches_per_epoch = 10

    seeds = np.random.get_bit_generator().random_raw(batch_shape).astype(np.uint32)
    rng = module.RNG(seeds)

    loader = TextureLoader(device)
    target_tex = loader.load_texture("inputs/checkerboard.png", {"load_as_normalized": True, "generate_mips": True})
    sampler = device.create_sampler(min_lod=0, max_lod=7)
    uv_grid = create_uv_grid(device, resolution)
    
    # module.sampleMip(uv_grid, target_tex, sampler, _result=app.output)
    # bitmap = app.output.to_bitmap()
    # bitmap.convert(
    #     sgl.Bitmap.PixelFormat.rgb,
    #     sgl.Bitmap.ComponentType.uint8,
    #     srgb_gamma=True
    # ).write('../sayan_code/slangpy-ml/outputs/mipsample.png')
    # print(f"Output image saved to ../sayan_code/slangpy-ml/outputs/mipsample.png")
    
    epoch_count = 0
    timer = Timer()
    cmd = device.create_command_buffer()

    while app.process_events(): # Change to while True for headless training
        if epoch_count >= max_epochs:
            print(f"Reached maximum epochs ({max_epochs}). Stopping training.")
            break
        timer.start()

        # Prefetch functions so we don't do module lookups in a tight loop
        train = module.trainTexture
        step = module.AdamOptimizer.step

        cmd.open()
        for i in range(num_batches_per_epoch):
            train.append_to(cmd, model, rng, target_tex, sampler, mipmap_level, loss_scale)
            for params, optim in zip(model.parameters(), optimizers):
                step.append_to(cmd, optim, params, params.grad_out, learning_rate, grad_scale)
        cmd.close()

        id = device.submit_command_buffer(cmd)
        # Stall and wait, then garbage collect for a good interactive experience.
        # Will slow things down a lot though - headless training will run faster.
        device.wait_command_buffer(id)
        device.run_garbage_collection()

        msamples = (num_batches_per_epoch * math.prod(batch_shape)) * 1e-6
        epoch_count += 1
        
        print(f"Epoch {epoch_count}/{max_epochs} - Throughput: {timer.frequency() * msamples:.2f} MSamples/s "
              f"Epoch time: {timer.elapsed() * 1e3:.1f}ms")

        # Evaluate the model once per epoch, comment below 2 lines for headless training
        # module.evalModel(model, uv_grid, _result=app.output)
        # app.present()

        timer.stop()

    device.wait()
    device.run_garbage_collection()
    return model

def create_uv_grid(device: Device, resolution: int):
    span = np.linspace(0, 1, resolution, dtype=np.float32)
    uvs_np = np.stack(np.broadcast_arrays(span[None, :], span[:, None]), axis=2)
    uvs = NDBuffer(device, 'float2', shape=(resolution, resolution))
    uvs.copy_from_numpy(uvs_np)
    return uvs

def save_model_weights(model, filename):
    """Save model weights to a file."""
    weights_dict = {}
    
    # Filter for LinearLayers first
    linear_layers = [m for m in model.modules() if isinstance(m, LinearLayer)]
    
    # Save each LinearLayer's weights
    for i, layer in enumerate(linear_layers):
        if layer.weights is not None and layer.biases is not None:
            weights = layer.weights.storage.to_numpy()
            biases = layer.biases.storage.to_numpy()
            weights_dict[f"layer_{i}_weights"] = weights
            weights_dict[f"layer_{i}_biases"] = biases
    
    # Save to file
    np.savez(filename, **weights_dict)
    print(f"Model weights saved to {filename} with {len(linear_layers)} linear layers")

def load_model_weights(model, filename):
    """Load model weights from a file."""
    # Load the weights
    weights_dict = np.load(filename)
    
    # Find all LinearLayers in the model
    linear_layers = [m for m in model.modules() if isinstance(m, LinearLayer)]
    
    # Assign weights to each layer
    for i, layer in enumerate(linear_layers):
        if f"layer_{i}_weights" in weights_dict and f"layer_{i}_biases" in weights_dict:
            # Copy weights and biases to the device
            layer.weights.storage.copy_from_numpy(weights_dict[f"layer_{i}_weights"])
            layer.biases.storage.copy_from_numpy(weights_dict[f"layer_{i}_biases"])
    
    print(f"Model weights loaded from {filename}")

def inference_main(model_path_base, lod, output_image_path, resolution=512):
    """Run inference with a saved model and save the output as an image."""
    # Create app with window, just like in training
    app = App("Neural Texture Inference", device_type=DeviceType.vulkan, width=resolution, height=resolution)
    device = app.device
    
    # Create the model with the same architecture as during training
    floor_model = ModuleChain(
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

    ceil_model = ModuleChain(
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
    floor_model.initialize(device)
    ceil_model.initialize(device)
    
    # Load the saved weights
    floor_mip_level = math.floor(lod)
    floor_model_path = os.path.join(model_path_base, f"model_{floor_mip_level}.npz")
    load_model_weights(floor_model, floor_model_path)
    
    
    ceil_mip_level = math.ceil(lod)
    ceil_model_path = os.path.join(model_path_base, f"model_{ceil_mip_level}.npz")
    load_model_weights(ceil_model, ceil_model_path)
    
    alpha = lod - floor_mip_level
    
    # Create a UV grid for evaluation, just like in training
    uv_grid = create_uv_grid(device, resolution)
    
    # Load the module for evaluation
    module = Module.load_from_file(device, "NeuralTexture.slang")

    device.wait()
    
    # Evaluate the model once to generate the texture
    module.evalMipLevel(floor_model, ceil_model, uv_grid, alpha, _result=app.output)

    # Convert the output texture to a bitmap and save it
    bitmap = app.output.to_bitmap()
    bitmap.convert(
        sgl.Bitmap.PixelFormat.rgb,
        sgl.Bitmap.ComponentType.uint8,
        srgb_gamma=True
    ).write(output_image_path)
    print(f"Output image saved to {output_image_path}")

    # # Present the result and keep the window open
    # while app.process_events():
    #     # Keep presenting the result
    #     app.present()
    #     time.sleep(0.01) 
    

class Timer:
    def __init__(self, history: int = 16):
        super().__init__()
        self.index = 0
        self.begin = None
        self.times = [0.0] * history
        self.history = history

    def start(self):
        self.begin = time.time()

    def stop(self):
        if self.begin is None:
            return

        t = time.time()
        elapsed = t - self.begin
        self.begin = t

        self.times[self.index % self.history] = elapsed
        self.index += 1

        return self.elapsed()

    def elapsed(self):
        l = min(self.index, self.history)
        return 0 if l == 0 else sum(self.times[:l]) / l

    def frequency(self):
        e = self.elapsed()
        return 0 if e == 0 else 1.0 / e

"""
Scripts for training and inference with the neural texture generator.
python main.py --mode train --max_epochs 100 --save_path my_model.npz
python main.py --mode inference --save_path my_model.npz --output my_result.png
"""
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Neural Texture Generator")
    parser.add_argument("--mode", choices=["train", "inference"], default="train",
                        help="Whether to train a model or run inference")
    parser.add_argument("--save_dir", default="/mnt/sdb/tejan/code/sayan_code/slangpy-ml/checkpoints",
                        help="Path to save or load model weights")
    parser.add_argument("--output", default="output.png",
                        help="Path for the output image in inference mode")
    parser.add_argument("--resolution", type=int, default=512,
                        help="Resolution of the output image")
    parser.add_argument("--max_epochs", type=int, default=100,
                        help="Maximum number of epochs for training")
    parser.add_argument("--model_path_base", default="/mnt/sdb/tejan/code/sayan_code/slangpy-ml/checkpoints/checkerboard",
                        help="Base path for model weights")
    
    parser.add_argument("--mipmap_level", type=float, default=0,
                        help="Mipmap level for training")
    
    args = parser.parse_args()
    
    # Save the trained model
    if not osp.exists(args.save_dir):
        os.makedirs(args.save_dir)
    save_path = Path(args.save_dir) / f"model_{int(args.mipmap_level)}.npz"
    
    if args.mode == "train":
        # Run training
        model = training_main(max_epochs=args.max_epochs, mipmap_level=int(args.mipmap_level))
        save_model_weights(model, save_path)
    else:
        # Run inference
        inference_main(args.model_path_base, args.mipmap_level, output_image_path=args.output, resolution=args.resolution)

