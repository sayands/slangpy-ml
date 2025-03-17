import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np
import imageio

# Initialize Pygame
pygame.init()
display = (800, 600)
pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
gluPerspective(45, (display[0]/display[1]), 0.1, 50.0)
glTranslatef(0.0, 0.0, -5)

# Load BC1-compressed Earth texture (assume it's pre-compressed to .dds)
texture_id = glGenTextures(1)
glBindTexture(GL_TEXTURE_2D, texture_id)
# Load compressed texture data (placeholder, use actual loading for .dds)
glTexImage2D(GL_TEXTURE_2D, 0, GL_COMPRESSED_RGB_S3TC_DXT1_EXT, 8192, 4096, 0, GL_RGB, GL_UNSIGNED_BYTE, None)
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

# Create sphere (using gluSphere for simplicity)
quad = gluNewQuadric()
gluQuadricTexture(quad, GL_TRUE)

# Rotation variables
angle = 0
frames = []
frame_count = 360  # For 360-degree rotation, 1 frame per degree

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            quit()

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glLoadIdentity()
    glTranslatef(0.0, 0.0, -5)
    glRotatef(angle, 0, 1, 0)  # Rotate around Y-axis

    glEnable(GL_TEXTURE_2D)
    gluSphere(quad, 1.0, 32, 32)  # Render sphere with texture
    glDisable(GL_TEXTURE_2D)

    # Capture frame
    data = glReadPixels(0, 0, 800, 600, GL_RGB, GL_UNSIGNED_BYTE)
    image = np.frombuffer(data, dtype=np.uint8).reshape(600, 800, 3)
    frames.append(image)

    pygame.display.flip()
    angle += 1
    if angle >= 360:
        break

# Save frames as video
imageio.mimsave('traditional_rotation.mp4', frames, fps=30)