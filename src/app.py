import gradio as gr
import numpy as np
from PIL import Image
from utils import load_model, segment_person, resize_image

# Load model and processor once
processor, model = load_model()

# Default background (solid color)
default_bg = Image.new("RGB", (512, 512), color=(95, 147, 89))





def generate_3d_outputs(person_img, background_img=None, shift_pixels=10,  person_size=100):
    # Resize images to match
    image = resize_image(person_img, person_size)

    if background_img is None:
        background = default_bg.resize(image.size)
    else:
        background = Image.fromarray(background_img).convert("RGB").resize(image.size)

    # Step 1: Segment person
    mask = segment_person(image, processor, model)

    image_np = np.array(image)
    background_np = np.array(background)

    person_only = image_np * mask
    background_only = background_np * (1 - mask)

    # Step 2: Create stereo pair
    person_left = np.roll(person_only, shift=-shift_pixels, axis=1)
    person_right = np.roll(person_only, shift=shift_pixels, axis=1)

    left_eye = np.clip(person_left + background_only, 0, 255).astype(np.uint8)
    right_eye = np.clip(person_right + background_only, 0, 255).astype(np.uint8)


    # --- Combine left and right images side by side ---
    stereo_pair = np.concatenate([left_eye, right_eye], axis=1)
    stereo_image = Image.fromarray(stereo_pair)

    # Step 3: Create anaglyph
    anaglyph = np.stack([
        left_eye[:, :, 0],  # Red from left
        right_eye[:, :, 1],  # Green from right
        right_eye[:, :, 2]   # Blue from right
    ], axis=2)

    anaglyph_img = Image.fromarray(anaglyph.astype(np.uint8))
    left_img = Image.fromarray(left_eye)
    right_img = Image.fromarray(right_eye)

    return anaglyph_img, stereo_image

# Gradio Interface
demo = gr.Interface(
    fn=generate_3d_outputs,
    inputs=[
        gr.Image(label="Person Image"),
        gr.Image(label="Optional Background Image"),
        gr.Slider(minimum=0, maximum=10, step=1, value=10, label="interaxial distance"),
        gr.Slider(minimum=10, maximum=200, step=10, value=100, label="Person Size %"),
  
    ],
    outputs=[
        gr.Image(label="3D Anaglyph Image"),
        gr.Image(label="Stereo_pair"),
    ],
    title="3D Person Segmentation Viewer",
    description="Upload a person photo and optionally a background image. Outputs anaglyph and stereo views."
)

if __name__ == "__main__":
    demo.launch()
