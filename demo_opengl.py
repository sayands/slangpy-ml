import numpy as np
import cv2
import glfw
from OpenGL.GL import *
from OpenGL.GL.shaders import compileShader, compileProgram
import time
from PIL import Image

# Vertex Shader: Pass UVs to fragment shader
vertex_shader_source = """
#version 330 core
layout(location = 0) in vec3 position;
layout(location = 1) in vec2 texCoord;
out vec2 vTexCoord;
void main() {
    gl_Position = vec4(position, 1.0);
    vTexCoord = texCoord;
}
"""

# Fragment Shader: Zoom texture around center
fragment_shader_source = """
#version 330 core
in vec2 vTexCoord;
out vec4 fragColor;
uniform sampler2D textureSampler;
uniform float zoomFactor;
void main() {
    vec2 center = vec2(0.5, 0.5);
    vec2 uv = center + (vTexCoord - center) / zoomFactor;
    uv = clamp(uv, 0.0, 1.0); // Clamp UVs to [0, 1]
    fragColor = texture(textureSampler, uv);
}
"""

class EarthDemoOpenGL:
    def __init__(self, texture_path="/mnt/sdb/tejan/code/sayan_code/slangpy-ml/inputs/earth.jpg", width=2048, height=2048):
        self.width = width
        self.height = height
        self.texture_path = texture_path

        # Initialize GLFW
        if not glfw.init():
            raise Exception("GLFW initialization failed")

        # Create window
        self.window = glfw.create_window(self.width, self.height, "Earth Texture Zoom", None, None)
        if not self.window:
            glfw.terminate()
            raise Exception("GLFW window creation failed")
        glfw.make_context_current(self.window)

        # Set up OpenGL
        glViewport(0, 0, self.width, self.height)
        glEnable(GL_TEXTURE_2D)

        # Compile shaders
        vertex_shader = compileShader(vertex_shader_source, GL_VERTEX_SHADER)
        fragment_shader = compileShader(fragment_shader_source, GL_FRAGMENT_SHADER)
        self.shader_program = compileProgram(vertex_shader, fragment_shader)
        glUseProgram(self.shader_program)

        # Load texture
        self.texture_id = self.load_texture()

        # Set up quad geometry (full-screen)
        vertices = np.array([
            # x, y, z, u, v
            -1.0, -1.0, 0.0, 0.0, 0.0,  # Bottom-left
             1.0, -1.0, 0.0, 1.0, 0.0,  # Bottom-right
             1.0,  1.0, 0.0, 1.0, 1.0,  # Top-right
            -1.0,  1.0, 0.0, 0.0, 1.0   # Top-left
        ], dtype=np.float32)
        indices = np.array([0, 1, 2, 0, 2, 3], dtype=np.uint32)

        # Vertex Buffer Object (VBO) and Vertex Array Object (VAO)
        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)

        vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, vbo)
        glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)

        ebo = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)

        # Vertex attributes
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * 4, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * 4, ctypes.c_void_p(3 * 4))
        glEnableVertexAttribArray(1)

        # Get uniform location
        self.zoom_factor_loc = glGetUniformLocation(self.shader_program, "zoomFactor")

    def load_texture(self):
        # Load image with PIL
        img = Image.open(self.texture_path).convert("RGB")
        img_data = np.array(img, dtype=np.uint8)

        # Create OpenGL texture
        texture_id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, texture_id)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, img.width, img.height, 0, GL_RGB, GL_UNSIGNED_BYTE, img_data)
        glGenerateMipmap(GL_TEXTURE_2D)

        # Texture parameters
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

        return texture_id

    def render_frame(self, zoom_factor):
        glClear(GL_COLOR_BUFFER_BIT)
        glUseProgram(self.shader_program)
        glUniform1f(self.zoom_factor_loc, zoom_factor)

        glBindTexture(GL_TEXTURE_2D, self.texture_id)
        glBindVertexArray(self.vao)
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, None)

        # Read pixels from framebuffer
        glFinish()  # Ensure rendering is complete
        frame_data = glReadPixels(0, 0, self.width, self.height, GL_RGB, GL_UNSIGNED_BYTE)
        frame_array = np.frombuffer(frame_data, dtype=np.uint8).reshape(self.height, self.width, 3)
        # Flip vertically (OpenGL origin is bottom-left)
        frame_array = np.flipud(frame_array)
        return frame_array

    def generate_video(self, output_path="earth_zoom_opengl.mp4", duration=5, fps=30):
        total_frames = int(duration * fps)
        zoom_in_frames = total_frames // 2
        zoom_out_frames = total_frames - zoom_in_frames

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_path, fourcc, fps, (self.width, self.height))

        times = []
        for i in range(zoom_in_frames):
            t = i / (zoom_in_frames - 1) if zoom_in_frames > 1 else 0
            zoom_factor = 1.0 + (20.0 - 1.0) * t  # Zoom in from 1x to 20x
            start_time = time.time()
            frame = self.render_frame(zoom_factor)
            times.append(time.time() - start_time)
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            video_writer.write(frame_bgr)
            print(f"Rendered frame {i+1}/{total_frames} (Zoom In: {zoom_factor:.2f}x)")

        for i in range(zoom_out_frames):
            t = i / (zoom_out_frames - 1) if zoom_out_frames > 1 else 0
            zoom_factor = 20.0 - (20.0 - 1.0) * t  # Zoom out from 20x to 1x
            start_time = time.time()
            frame = self.render_frame(zoom_factor)
            times.append(time.time() - start_time)
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            video_writer.write(frame_bgr)
            print(f"Rendered frame {zoom_in_frames+i+1}/{total_frames} (Zoom Out: {zoom_factor:.2f}x)")

        video_writer.release()
        glfw.terminate()
        print(f"Video saved to {output_path}")
        print(f"Average render time: {np.mean(times):.4f}s")

    def run(self):
        self.generate_video()

if __name__ == "__main__":
    demo = EarthDemoOpenGL(texture_path="/mnt/sdb/tejan/code/sayan_code/slangpy-ml/inputs/earth.jpg", width=2048, height=2048)
    demo.run()