import gradio as gr
import numpy as np
from PIL import Image
from utils import load_model, segment_person, resize_image, split_stereo_image

# Load model and processor once
processor, model = load_model()

# Default background (solid color)
default_bg = Image.new("RGB", (512, 512), color=(95, 147, 89))





def generate_3d_outputs(person_img, background_img=None, shift_pixels=10,  person_size=100):
    # Resize images to match
    image = resize_image(person_img, person_size)
    background_img = background_img if background_img is not None else default_bg


    # Split background image into left and right halves
    leftBackground, rightBackground = split_stereo_image(Image.fromarray(background_img))

    # Resize image to match background dimensions

    
    image = Image.fromarray(np.array(image)).resize((leftBackground.shape[1], leftBackground.shape[0]))
    # Step 1: Segment person
    mask = segment_person(image, processor, model)

    image_np = np.array(image)

    leftBackground_np = np.array(leftBackground)
    rightBackground_np = np.array(rightBackground)


    person_only = image_np * mask
    leftBackground_only = leftBackground_np * (1 - mask)
    rightBackground_only = rightBackground_np * (1 - mask)

    # Step 2: Create stereo pair
    person_left = np.roll(person_only, shift=-shift_pixels, axis=1)
    person_right = np.roll(person_only, shift=shift_pixels, axis=1)
    

    left_eye = np.clip(person_right + leftBackground_only, 0, 255).astype(np.uint8)
    right_eye = np.clip(person_left + rightBackground_only, 0, 255).astype(np.uint8)
    person_segmentation = np.clip(person_only, 0, 255).astype(np.uint8)

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

    return person_segmentation, stereo_image, anaglyph_img

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
        gr.Image(label="segmentation mask"),
        gr.Image(label="Stereo_pair"),
        gr.Image(label="3D Anaglyph Image")
    ],
    title="3D Person Segmentation Viewer",
    description="Upload a person photo and optionally a background image. Outputs anaglyph and stereo views."
)

if __name__ == "__main__":
    demo.launch()
