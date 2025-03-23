import os
import numpy as np
from PIL import Image
from utils import load_model, segment_person

def create_anaglyph(person_img_path, background_img_path, output_path="output_anaglyph.png"):
    image = Image.open(person_img_path).convert("RGB")
    background = Image.open(background_img_path).convert("RGB").resize(image.size)

    processor, model = load_model()
    mask = segment_person(image, processor, model)

    image_np = np.array(image)
    background_np = np.array(background)

    person_only = image_np * mask
    background_only = background_np * (1 - mask)

    # Stereoscopic shift
    shift_pixels = 10
    person_left = np.roll(person_only, shift=-shift_pixels, axis=1)
    person_right = np.roll(person_only, shift=shift_pixels, axis=1)

    left_eye = np.clip(person_left + background_only, 0, 255).astype(np.uint8)
    right_eye = np.clip(person_right + background_only, 0, 255).astype(np.uint8)

    # Merge into red-cyan anaglyph
    anaglyph = np.stack([
        left_eye[:, :, 0],
        right_eye[:, :, 1],
        right_eye[:, :, 2]
    ], axis=2)

    anaglyph_img = Image.fromarray(anaglyph.astype(np.uint8))
    anaglyph_img.save(output_path)
    print(f"✅ Anaglyph image saved to: {output_path}")

if __name__ == "__main__":
    create_anaglyph("person.png", "bg.png")
