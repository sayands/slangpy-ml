# SPDX-License-Identifier: Apache-2.0

from app import App
import slangpy as spy
import sgl
import numpy as np
import cv2
from slangpy.types import call_id
from slangpy.backend import DeviceType, TextureLoader

from network import (
    Activation, ELUAct, ExpAct, FrequencyEncoding, LeakyReLUAct, 
    LinearLayer, ModuleChain, NoneAct, ReLUAct, SigmoidAct, 
    SwishAct, TanhAct
)

import os
from glob import glob
from time import time


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

# Function to normalize a vector
def normalize(v):
    """Normalize a vector to have a magnitude of 1."""
    magnitude = np.sqrt(v.x**2 + v.y**2 + v.z**2)
    if magnitude == 0:
        return v
    return sgl.float3(v.x / magnitude, v.y / magnitude, v.z / magnitude)

def cross(v1, v2):
    """Compute the cross product of two vectors."""
    return sgl.float3(
        v1.y * v2.z - v1.z * v2.y,
        v1.z * v2.x - v1.x * v2.z,
        v1.x * v2.y - v1.y * v2.x
    )

def dot(v1, v2):
    """Compute the dot product of two vectors."""
    return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z

# Rotation matrix around X-axis
def rotation_matrix_x(angle_rad):
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([
        [1, 0, 0, 0],
        [0, c, -s, 0],
        [0, s, c, 0],
        [0, 0, 0, 1]
    ])

# Rotate a vertex around X-axis
def rotate_vertex(vertex, angle_rad):
    pos = np.array([vertex.x, vertex.y, vertex.z, 1.0])
    rot_matrix = rotation_matrix_x(angle_rad)
    rotated = np.dot(rot_matrix, pos)
    return sgl.float3(rotated[0], rotated[1], rotated[2])

# Camera class
class Camera:
    def __init__(self, app):
        self.o = sgl.float3(0.0, 0.0, 2.0)
        self.look_at = sgl.float3(0.0, 0.0, 0.0)
        self.up = sgl.float3(0.0, 1.0, 0.0)
        self.scale = sgl.float3(1.0, 1.0, 1.0)
        self.app = app
        self.near = 0.1
        self.far = 100.0
        self.fov = float(np.radians(50.0))

    def get_this(self):
        return {
            "o": self.o,
            "look_at": self.look_at,
            "up": self.up,
            "scale": self.scale,
            "frameDim": sgl.float2(float(self.app._window.width), float(self.app._window.height)),
            "near": float(self.near),
            "far": float(self.far),
            "fov": float(self.fov),
            "_type": "Camera"
        }

app = App("Neural Texture", device_type=DeviceType.vulkan, width=1024, height=1024)
device = app.device
rasterizer2d = spy.Module.load_from_file(app.device, "rasterizer2d.slang")

# Set up camera
camera = Camera(app)

# Load texture
loader = TextureLoader(device)
texture = loader.load_texture("/mnt/sdb/tejan/code/sayan_code/slangpy-ml/inputs/bernie.jpg", {"load_as_normalized": True, "generate_mips": True})
sampler = device.create_sampler(min_lod=0, max_lod=7)

# Load Neural Textures
models = []
base_path = os.path.join('/mnt/sdb/tejan/code/sayan_code/slangpy-ml/checkpoints/bernie')
#number of models in base path
model_paths = sorted(glob(os.path.join(base_path, 'model_*.npz')))
for model_path in model_paths:
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
    load_model_weights(model, model_path)
    models.append(model)

num_models = len(models)

# Define vertices with UVs
depth = 0.0
base_v0 = sgl.float3(-0.5, -0.5, depth) # Bottom-left
base_v1 = sgl.float3(-0.5, 0.5, depth)  # Top-left
base_v2 = sgl.float3(0.5, 0.5, depth)   # Top-right
base_v3 = sgl.float3(0.5, -0.5, depth)  # Bottom-right

uv0 = sgl.float2(0.0, 1.0)  # Bottom-left
uv1 = sgl.float2(1.0, 0.0)  # Top-left
uv2 = sgl.float2(0.0, 0.0)  # Top-right
uv3 = sgl.float2(1.0, 1.0)  # Bottom-right
uv_triangles = [
    [uv0, uv1, uv2],
    [uv0, uv3, uv1]
]

# Animation parameters
num_frames = 60  # Number of frames for 0° to 90° rotation
angle_step = np.radians(90.0) / (num_frames - 1)  # Incremental angle per frame

# Video setup
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec
fps = 30
width, height = app._window.width, app._window.height
video_writer = cv2.VideoWriter("rotated_triangles.mp4", fourcc, fps, (width, height))

# Render and save frames
print("Rendering and saving video...")
render_times = []
for frame in range(num_frames):
    angle = -frame * angle_step
    
    # Rotate vertices
    v0 = rotate_vertex(base_v0, angle)
    v1 = rotate_vertex(base_v1, angle)
    v2 = rotate_vertex(base_v2, angle)
    v3 = rotate_vertex(base_v3, angle)
    
    triangle1 = [v0, v1, v2]
    triangle2 = [v0, v2, v3]
    triangles = [triangle1, triangle2]
    num_triangles = len(triangles)
    
    # Render the frame
    time_start = time()
    all_lods = []
    rasterizer2d.rasterize(camera.get_this(), triangles, uv_triangles, num_triangles, texture, sampler, models[0], call_id(), _result=app.output)
    bitmap = app.output.to_bitmap()
    bitmap_rgb = bitmap.convert(
        sgl.Bitmap.PixelFormat.rgb,
        sgl.Bitmap.ComponentType.uint8,
        srgb_gamma=True
    )
    pixel_data = np.array(bitmap_rgb, dtype=np.uint8).reshape(height, width, 3)
    all_lods.append(pixel_data)
    total_time = time() - time_start
    render_times.append(total_time)

    # # Extract pixel data
    # bitmap = app.output.to_bitmap()
    # bitmap_rgb = bitmap.convert(
    #     sgl.Bitmap.PixelFormat.rgb,
    #     sgl.Bitmap.ComponentType.uint8,
    #     srgb_gamma=True
    # )
    # pixel_data = np.array(bitmap_rgb, dtype=np.uint8).reshape(height, width, 3)
    frame_data = cv2.cvtColor(pixel_data, cv2.COLOR_RGB2BGR)
    
    # Write frame to video
    if frame != num_frames - 1:
        video_writer.write(frame_data)
    
    # Optional: Print progress
    if frame % 10 == 0:
        print(f"Frame {frame}/{num_frames} at angle {np.degrees(angle):.1f}°")

for frame in range(num_frames-2, -1, -1):
    angle = -frame * angle_step
    
    # Rotate vertices
    v0 = rotate_vertex(base_v0, angle)
    v1 = rotate_vertex(base_v1, angle)
    v2 = rotate_vertex(base_v2, angle)
    v3 = rotate_vertex(base_v3, angle)
    
    triangle1 = [v0, v1, v2]
    triangle2 = [v0, v2, v3]
    triangles = [triangle1, triangle2]
    num_triangles = len(triangles)
    
    # Render the frame
    time_start = time()
    all_lods = []
    rasterizer2d.rasterize(camera.get_this(), triangles, uv_triangles, num_triangles, texture, sampler, models[0], call_id(), _result=app.output)
    bitmap = app.output.to_bitmap()
    bitmap_rgb = bitmap.convert(
        sgl.Bitmap.PixelFormat.rgb,
        sgl.Bitmap.ComponentType.uint8,
        srgb_gamma=True
    )
    pixel_data = np.array(bitmap_rgb, dtype=np.uint8).reshape(height, width, 3)
    all_lods.append(pixel_data)
    total_time = time() - time_start
    render_times.append(total_time)

    # # Extract pixel data
    # bitmap = app.output.to_bitmap()
    # bitmap_rgb = bitmap.convert(
    #     sgl.Bitmap.PixelFormat.rgb,
    #     sgl.Bitmap.ComponentType.uint8,
    #     srgb_gamma=True
    # )
    # pixel_data = np.array(bitmap_rgb, dtype=np.uint8).reshape(height, width, 3)
    frame_data = cv2.cvtColor(pixel_data, cv2.COLOR_RGB2BGR)
    
    # Write frame to video
    video_writer.write(frame_data)
    
    # Optional: Print progress
    if frame % 10 == 0:
        print(f"Frame {frame}/{num_frames} at angle {np.degrees(angle):.1f}°")

# Release video writer
video_writer.release()
print("Video saved to: rotated_triangles.mp4")

# Print render times
print("\nRender times:")
print(f"  Min: {np.min(render_times):.3f} s")
print(f"  Max: {np.max(render_times):.3f} s")
print(f"  Avg: {np.mean(render_times):.3f} s")

# # Print debug information for final frame
# print("\nCamera settings:")
# print(f"  Position: ({camera.o.x}, {camera.o.y}, {camera.o.z})")
# print(f"  Field of view: {np.degrees(camera.fov)} degrees")
# print(f"  Image dimensions: {width}x{height}")

# print("\nFinal triangle vertices (world space):")
# for i, triangle in enumerate(triangles):
#     print(f"  Triangle {i+1}:")
#     for j, vertex in enumerate(triangle):
#         print(f"    Vertex {j+1}: ({vertex.x:.3f}, {vertex.y:.3f}, {vertex.z:.3f})")