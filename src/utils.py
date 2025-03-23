import torch
import numpy as np
from PIL import Image
import cv2
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation

def load_model():
    processor = AutoImageProcessor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
    model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
    return processor, model

def segment_person(image: Image.Image, processor, model):
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    upsampled_logits = torch.nn.functional.interpolate(
        logits,
        size=image.size[::-1],
        mode="bilinear",
        align_corners=False,
    )
    pred_classes = upsampled_logits.argmax(dim=1)[0].cpu().numpy()
    mask = (pred_classes == 12).astype(np.uint8) * 255  # Class 12 = person

    # Clean mask
    kernel = np.ones((7, 7), np.uint8)
    eroded_mask = cv2.erode(mask, kernel, iterations=1)
    blurred_mask = cv2.GaussianBlur(eroded_mask, (3, 3), sigmaX=0, sigmaY=0)

    final_mask = blurred_mask.astype(np.float32) / 255.0
    final_mask_3ch = np.stack([final_mask]*3, axis=-1)

    return final_mask_3ch


def resize_image(image, size_percent):
  # Convert image to RGB if it's RGBA
  image = Image.fromarray(image).convert("RGB")
  width, height = image.size
  new_width = int(width * size_percent / 100)
  new_height = int(height * size_percent / 100)
  
  # Create new transparent image with original dimensions
  resized_image = Image.new('RGB', (width, height), (0, 0, 0))
  
  # Resize original image
  scaled_content = image.resize((new_width, new_height))
  
  # Calculate position to paste resized content in center
  x = (width - new_width) // 2
  y = (height - new_height) // 2
  
  # Paste resized content onto transparent background
  resized_image.paste(scaled_content, (x, y))
  
  return resized_image
