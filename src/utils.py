import torch
import numpy as np
from PIL import Image
import cv2
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation
from imagehash import average_hash

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

# Check if two images are similar
def check_image_similarity(image1, image2):
 
    hash1 = average_hash(Image.fromarray(image1))
    hash2 = average_hash(Image.fromarray(image2)) 
    return hash1 - hash2  < 10


def split_stereo_image(image):
    """
    Splits an image into left and right halves for stereographic viewing.
    
    Args:
        image: PIL Image or numpy array
        
    Returns:
        tuple: (left_half, right_half) as numpy arrays
    """
    # Convert to numpy array if PIL Image
    if isinstance(image, Image.Image):
        image = np.array(image)
        
    # Get width and calculate split point
    width = image.shape[1]
    split_point = width // 2
    
    # Split into left and right halves
    left_half = image[:, :split_point]
    right_half = image[:, split_point:]

    #If stereo image is provided, return left and right halves
    if check_image_similarity(left_half, right_half):
        return left_half, right_half
    else:
        return image, resize_image(image, 99)
    
