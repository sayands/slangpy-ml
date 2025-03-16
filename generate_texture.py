from PIL import Image, ImageDraw
import noise
import numpy as np

def generate_checkerboard():
    # Define the size of the image
    image_size = (512, 512)

    # Create a new image with white background
    image = Image.new("RGB", image_size, "white")
    draw = ImageDraw.Draw(image)

    # Define the size of each square in the checkerboard
    square_size = 32  # 512 / 8 = 64, so 8x8 squares

    # Define the colors for the checkerboard
    color1 = (0, 0, 0)  # Black
    color2 = (255, 255, 255)  # White

    # Draw the checkerboard pattern
    for i in range(0, image_size[0], square_size):
        for j in range(0, image_size[1], square_size):
            if (i // square_size + j // square_size) % 2 == 0:
                draw.rectangle([i, j, i + square_size, j + square_size], fill=color1)
            else:
                draw.rectangle([i, j, i + square_size, j + square_size], fill=color2)

    # Save the image
    image.save("inputs/checkerboard.png")

def generate_perlin_texture():
    # Parameters
    resolution = 256 * 32 
    scale = 100.0  # Controls the frequency of the noise

    # Generate Perlin noise
    def create_perlin_noise(resolution, scale):
        texture = np.zeros((resolution, resolution))
        for y in range(resolution):
            for x in range(resolution):
                texture[y][x] = noise.pnoise2(x / scale, y / scale, octaves=6, persistence=0.5, lacunarity=2.0)
        # Normalize to 0-255
        texture = (texture - texture.min()) / (texture.max() - texture.min()) * 255
        return np.uint8(texture)

    # Generate the texture
    texture = create_perlin_noise(resolution, scale)

    # Save the texture as an image
    image = Image.fromarray(texture, mode='L')  # 'L' mode for grayscale
    image.save(f"inputs/perlin_{resolution}.png")

if __name__ == "__main__":
    generate_perlin_texture()