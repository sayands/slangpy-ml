import numpy as np
import slangpy
from slangpy import Module
from slangpy.backend import Device, DeviceType, TextureLoader
from slangpy.types import NDBuffer, call_id
import sgl
from pathlib import Path
import math
from app import App

import cv2
import os
import time

from matplotlib import pyplot as plt
import io


from PIL import Image

# Assume these are from your provided training script
from network import (
    FrequencyEncoding, LinearLayer, LeakyReLUAct, ModuleChain, SigmoidAct
)

# Load model weights function from your utils (assuming model_util is available)

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
            print(f"Loading weights for layer {i}")
            layer.weights.storage.copy_from_numpy(weights_dict[f"layer_{i}_weights"])
            layer.biases.storage.copy_from_numpy(weights_dict[f"layer_{i}_biases"])
    
    print(f"Model weights loaded from {filename}")

def create_uv_grid(device: Device, width: int, height: int):
    # Create UV grid
    w_span = np.linspace(0, 1, width, dtype=np.float32)
    h_span = np.linspace(0, 1, height, dtype=np.float32)
    uvs_np = np.stack(np.broadcast_arrays(w_span[None, :], h_span[:, None]), axis=2)
    uvs = NDBuffer(device, 'float2', shape=(height, width))
    uvs.copy_from_numpy(uvs_np)
    return uvs

# Create sphere geometry
def create_sphere_vertices(radius, rings, sectors):
    vertices = []
    for r in range(rings + 1):
        y = radius * math.cos(math.pi * r / rings - math.pi / 2)
        r_xy = radius * math.sin(math.pi * r / rings - math.pi / 2)
        for s in range(sectors):
            x = r_xy * math.cos(2 * math.pi * s / sectors)
            z = r_xy * math.sin(2 * math.pi * s / sectors)
            u = s / (sectors - 1)
            v = r / rings
            vertices.append([x, y, z, u, v])
    return np.array(vertices, dtype=np.float32)

def create_sphere_indices(rings, sectors):
    indices = []
    for r in range(rings):
        for s in range(sectors):
            r0 = r * sectors + s
            r1 = r * sectors + (s + 1) % sectors
            r2 = (r + 1) * sectors + (s + 1) % sectors
            r3 = (r + 1) * sectors + s
            indices.extend([r0, r1, r2, r0, r2, r3])
    return np.array(indices, dtype=np.uint32)

class EarthDemoHeadless:
    def __init__(self, model_path="/mnt/sdb/tejan/code/sayan_code/slangpy-ml/checkpoints/earth/8192/model_0.npz", width=8192, height=4096):
        # Store dimensions
        self.width = width
        self.height = height

        # Initialize device
        self.device = slangpy.create_device(DeviceType.vulkan, include_paths=[Path(__file__).parent])
        self.app = App(width=width, height=height, device_type=sgl.DeviceType.vulkan)
        self.device = self.app.device

        # Load trained neural model (single LOD)
        self.prev_model = ModuleChain(
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
        self.prev_model.initialize(self.device)
        # breakpoint()
        load_model_weights(self.prev_model, model_path)

        self.model = ModuleChain(
            FrequencyEncoding(5, 5),
            LinearLayer(50, 64),
            LeakyReLUAct(64),
            LinearLayer(64, 64),
            LeakyReLUAct(64),
            LinearLayer(64, 64),
            LeakyReLUAct(64),
            LinearLayer(64, 3),
            SigmoidAct(3)
        )
        self.model.initialize(self.device)
        model_path_base = "/mnt/sdb/tejan/code/sayan_code/slangpy-ml/checkpoints/earth/8192"
        model_path = os.path.join(model_path_base, f"model_1.npz")
        load_model_weights(self.model, model_path)

        # Create UV grid
        self.uv_grid = create_uv_grid(self.device, self.width, self.height)

        # Load rendering module
        self.render_module = Module.load_from_file(self.device, "NeuralTexture.slang")

        # Create output texture
        self.output_texture = self.device.create_texture(
            width=self.width,
            height=self.height,
            format=sgl.Format.rgba32_float,
            usage=sgl.ResourceUsage.shader_resource | sgl.ResourceUsage.unordered_access,
            debug_name="output_texture"
        )
        loader = TextureLoader(self.device)
        self.input_texture = loader.load_texture('/mnt/sdb/tejan/code/sayan_code/slangpy-ml/inputs/earth.jpg', {"load_as_normalized": True, "generate_mips": True})
        self.sampler = self.device.create_sampler(min_lod=0, max_lod=7)
    
    def render_frame(self, zoom_factor, center_u, center_v, use_res=False):
        # Create a buffer to hold the frame data
        frame_data = np.zeros((self.height, self.width, 4), dtype=np.float32)

        # Render each pixel by calling renderSphere
        # breakpoint()
        if use_res:
            self.render_module.residualRenderSphere(
                self.model,
                self.prev_model,
                self.uv_grid,
                zoom_factor,
                center_u, center_v,
                call_id(),
                _result=self.app.output
            )
        else:
            self.render_module.renderSphere(
                self.prev_model,
                self.uv_grid,
                zoom_factor,
                center_u, center_v,
                call_id(),
                _result=self.app.output
            )

        # self.render_module.renderTex(
        #     self.uv_grid,
        #     0.0,
        #     zoom_factor,
        #     self.input_texture,
        #     self.sampler,
        #     call_id(),
        #     _result=self.app.output
        # )

        # self.render_module.evalMipLevel(self.model, self.model, self.uv_grid, 1.0, _result=self.app.output)
        # Copy frame data to texture (since we can't write directly in Slang)
        bitmap = self.app.output.to_bitmap()
        bitmap_rgb = bitmap.convert(
            sgl.Bitmap.PixelFormat.rgb,
            sgl.Bitmap.ComponentType.uint8,
            srgb_gamma=True
        )
        bitmap_rgb.write("output.png")
        img_array = cv2.imread("output.png")
        return img_array

    def generate_video(self, output_path="earth_zoom.mp4", duration=5, fps=30):
        # Calculate total frames
        total_frames = int(duration * fps)  # e.g., 5s * 30fps = 150 frames
        zoom_in_frames = total_frames // 2  # Half for zoom in
        zoom_out_frames = total_frames - zoom_in_frames  # Half for zoom out

        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4 codec
        video_writer = cv2.VideoWriter(output_path, fourcc, fps, (640, 480))

        # Zoom in: 1.0 to 20.0
        times = []
        centers = [(0.167, 0.3), (0.72, 0.37), (0.9, 0.69)]
        for center_u, center_v in centers:
            for i in range(zoom_in_frames):
                t = i / (zoom_in_frames - 1) if zoom_in_frames > 1 else 0  # Normalized time [0, 1]
                zoom_factor = 1.0 + (20.0 - 1.0) * t  # Linear interpolation
                start_time = time.time()
                frame = self.render_frame(zoom_factor, center_u, center_v)
                frame2 = self.render_frame(zoom_factor, center_u, center_v, use_res=True)
                times.append(time.time() - start_time)
                # frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                # stack both images side by side in matplotlib. give them title
                fig, ax = plt.subplots(1, 2)
                ax[0].imshow(frame)
                ax[0].set_title('Base Model')
                ax[1].imshow(frame2)
                ax[1].set_title('With Residual')           
                bytesio = io.BytesIO()
                plt.tight_layout()
                ax[0].set_xticks([]), ax[0].set_yticks([])
                ax[1].set_xticks([]), ax[1].set_yticks([])

                plt.savefig('output2.png', format='png')
                bytesio.seek(0)
                img_array = cv2.imread('output2.png')
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                bytesio.close()
                plt.close()

                video_writer.write(img_array)
                print(f"Rendered frame {i+1}/{total_frames} (Zoom In: {zoom_factor:.2f}x)")

            # Zoom out: 20.0 to 1.0
            for i in range(zoom_out_frames):
                t = i / (zoom_out_frames - 1) if zoom_out_frames > 1 else 0  # Normalized time [0, 1]
                zoom_factor = 20.0 - (20.0 - 1.0) * t  # Linear interpolation
                start_time = time.time()
                frame = self.render_frame(zoom_factor, center_u, center_v)
                frame2 = self.render_frame(zoom_factor, center_u, center_v, use_res=True)
                times.append(time.time() - start_time)
                # frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                # stack both images side by side in matplotlib. give them title
                fig, ax = plt.subplots(1, 2)
                ax[0].imshow(frame)
                ax[0].set_title('Base Model')
                ax[1].imshow(frame2)
                ax[1].set_title('With Residual')     
                plt.tight_layout()
                ax[0].set_xticks([]), ax[0].set_yticks([])
                ax[1].set_xticks([]), ax[1].set_yticks([])      
                buffer = io.BytesIO()
                plt.savefig('output2.png', format='png')
                plt.close()
                buffer.seek(0)
                img_array = cv2.imread('output2.png')
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                buffer.close()
                video_writer.write(img_array)
                print(f"Rendered frame {zoom_in_frames+i+1}/{total_frames} (Zoom Out: {zoom_factor:.2f}x)")

        # Release video writer
        video_writer.release()
        print(f"Video saved to {output_path}")
        print(f"Average render time: {np.mean(times):.4f}s")

if __name__ == "__main__":
    demo = EarthDemoHeadless(model_path="/mnt/sdb/tejan/code/sayan_code/slangpy-ml/checkpoints/earth/8192/model_0.npz", width=512, height=512)
    demo.generate_video(output_path="earth_zoom.mp4", duration=5, fps=30)