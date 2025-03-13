from PIL import Image, ImageDraw

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