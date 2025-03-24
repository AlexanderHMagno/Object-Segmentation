---
title: Object Segmentation
emoji: 👁
colorFrom: gray
colorTo: pink
sdk: gradio
sdk_version: 5.22.0
app_file: src/app.py
pinned: false
---

# 3D Person Segmentation and Anaglyph Generation

[![Open in Spaces](https://img.shields.io/badge/🤗-Open%20in%20Spaces-blue.svg)](https://huggingface.co/spaces/axelhortua/Object-segmentation)

## Documentation Examples

### Input and Segmentation
![Input Image and Segmentation](public/images/documentation/input_segmentation.png)
*Example of original input image and its segmentation mask*

### Stereo Processing
![Stereo Processing](public/images/documentation/stereo_process.png)
*Demonstration of stereo pair generation with different interaxial distances*

### Anaglyph Output
![Anaglyph Result](public/images/documentation/anaglyph_output.png)
*Final anaglyph output with red-cyan 3D effect*

### Interface Overview
![Gradio Interface](public/images/documentation/interface.png)
*The Gradio web interface with all adjustment controls*

## Lab Report

### Introduction
This project implements a sophisticated 3D image processing system that combines person segmentation with stereoscopic and anaglyph image generation. The main objectives were to:
1. Accurately segment people from images using advanced AI models
2. Generate stereoscopic 3D effects from 2D images
3. Create red-cyan anaglyph images for 3D viewing
4. Provide an interactive web interface for real-time processing
5. Handle varying image sizes with intelligent mask alignment

### Methodology

#### Tools and Technologies Used
- **SegFormer (nvidia/segformer-b0)**: State-of-the-art transformer-based model for semantic segmentation
- **PyTorch**: Deep learning framework for running the SegFormer model
- **OpenCV**: Image processing operations and mask refinement
- **Gradio**: Web interface development
- **NumPy**: Efficient array operations for image manipulation
- **PIL (Python Imaging Library)**: Image loading and basic transformations

#### Mask Processing Deep Dive

The mask processing is a crucial component of our system, designed to handle various challenges in creating high-quality 3D effects:

1. **Why Mask Resizing is Necessary**
   - **Input Variability**: User-uploaded images come in different sizes and aspect ratios
   - **Model Constraints**: SegFormer outputs masks at a fixed resolution (512x512)
   - **Background Compatibility**: Backgrounds may have different dimensions than person images
   - **3D Effect Quality**: Proper alignment is crucial for convincing stereoscopic effects

2. **Mask Processing Pipeline**
   ```
   Original Image → SegFormer Segmentation → Initial Mask (512x512)
         ↓
   Resize to Match Background
         ↓
   Add Transparent Padding
         ↓
   Center Alignment
         ↓
   Final Processed Mask
   ```

3. **Technical Implementation**
   ```python
   # Pseudocode for mask processing
   def process_mask(mask, background_size):
       # Calculate padding dimensions
       pad_top = (background_height - mask_height) // 2
       pad_bottom = background_height - mask_height - pad_top
       pad_left = (background_width - mask_width) // 2
       pad_right = background_width - mask_width - pad_left
       
       # Add padding with transparency
       padded_mask = np.pad(mask, 
                           ((pad_top, pad_bottom), 
                            (pad_left, pad_right), 
                            (0,0)), 
                           mode='constant')
       
       return padded_mask
   ```

#### Visual Process Explanation

```
+----------------+     +----------------+     +----------------+
|   Original     |     |   Segmented   |     |    Padded     |
|    Image       | --> |     Mask      | --> |     Mask      |
|  (Variable)    |     |   (512x512)   |     | (Background)  |
+----------------+     +----------------+     +----------------+
         |                                           |
         v                                           v
+----------------+     +----------------+     +----------------+
|   Left View    |     |  Stereo Pair  |     |   Anaglyph    |
|    Shifted     | --> |  Combined     | --> |    Output     |
|                |     |               |     |               |
+----------------+     +----------------+     +----------------+
```

**Key Processing Steps Visualization:**

1. **Mask Generation and Sizing:**
   ```
   +------------+    +-----------+    +-------------+
   | Raw Image  |    | Raw Mask  |    | Sized Mask  |
   |  ******   | -> | ########  | -> | ########    |
   | *Image *  |    | #Mask  #  |    | #Mask    #  |
   |  ******   |    | ########  |    | ########    |
   +------------+    +-----------+    +-------------+
   ```

2. **Transparency Handling:**
   ```
   Original       Padded        Final
   +----+        +------+      +------+
   |####|        |      |      |  ##  |
   |####|   ->   |####  |  ->  |######|
   |####|        |####  |      |  ##  |
   +----+        +------+      +------+
   ```

#### Implementation Steps

1. **Person Segmentation**
   - Utilized SegFormer model fine-tuned on ADE20K dataset
   - Applied post-processing with erosion and Gaussian blur for mask refinement
   - Implemented mask scaling and centering for various input sizes
   - Added transparent padding for proper background integration

2. **Mask Processing and Alignment**
   - Implemented dynamic mask resizing to match background dimensions
   - Added centered padding for smaller masks
   - Preserved transparency in padded regions
   - Ensured proper aspect ratio maintenance

3. **Stereoscopic Processing**
   - Created depth simulation through horizontal pixel shifting
   - Implemented parallel view stereo pair generation
   - Added configurable interaxial distance for 3D effect adjustment
   - Enhanced alignment between stereo pairs with mask centering

4. **Anaglyph Generation**
   - Combined left and right eye views into red-cyan anaglyph
   - Implemented color channel separation and recombination
   - Added background image support with proper masking
   - Improved blending between foreground and background

5. **User Interface**
   - Developed interactive web interface using Gradio
   - Added real-time parameter adjustment capabilities
   - Implemented support for custom background images
   - Added size adjustment controls

### Results

The system produces three main outputs:
1. Segmentation mask showing the isolated person with proper transparency
2. Side-by-side stereo pair for parallel viewing with centered alignment
3. Red-cyan anaglyph image for 3D glasses viewing

Key Features:
- Adjustable person size (10-200%)
- Configurable interaxial distance (0-10 pixels)
- Optional custom background support
- Real-time processing and preview
- Intelligent mask alignment and padding
- Transparent background handling

### Discussion

#### Technical Challenges
1. **Mask Alignment**: Ensuring proper alignment between segmentation masks and background images required careful consideration of image dimensions and aspect ratios.
2. **Stereo Effect Quality**: Balancing the interaxial distance for comfortable viewing while maintaining the 3D effect.
3. **Performance Optimization**: Efficient processing of large images while maintaining real-time interaction.
4. **Transparency Handling**: Implementing proper transparency in padded regions while maintaining mask quality.
5. **Size Adaptation**: Managing different input image sizes while preserving aspect ratios and alignment.

#### Learning Outcomes
- Deep understanding of stereoscopic image generation
- Experience with state-of-the-art segmentation models
- Practical knowledge of image processing techniques
- Web interface development for ML applications
- Advanced mask manipulation and alignment strategies

### Conclusion

This project successfully demonstrates the integration of modern AI-powered segmentation with classical stereoscopic image processing techniques. The system provides an accessible way to create 3D effects from regular 2D images, with robust handling of different image sizes and proper transparency management.

#### Future Work
- Implementation of depth-aware 3D effect generation
- Support for video processing
- Additional 3D viewing formats (side-by-side, over-under)
- Enhanced background replacement options
- Mobile device optimization
- Advanced depth map generation
- Multi-person segmentation support

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
cd src
python app.py
```

## Parameters

- **Person Image**: Upload an image containing a person
- **Background Image**: (Optional) Custom background image
- **Interaxial Distance**: Adjust the 3D effect strength (0-10)
- **Person Size**: Adjust the size of the person in the output (10-200%)

## Output Types

1. **Segmentation Mask**: Shows the isolated person with proper transparency
2. **Stereo Pair**: Side-by-side stereo image for parallel viewing
3. **Anaglyph**: Red-cyan 3D image viewable with anaglyph glasses

## Technical Notes

- **Mask Processing Details**:
  - Initial mask is generated at 512x512 resolution
  - Dynamic padding calculation: `pad = (background_size - mask_size) // 2`
  - Transparency preservation using NumPy's constant padding mode
  - Aspect ratio maintained through centered scaling
  - Real-time size adjustments (10-200%) applied before padding

- **Size Handling Algorithm**:
  1. Calculate target dimensions based on background
  2. Resize mask while maintaining aspect ratio
  3. Add transparent padding to match background
  4. Center the mask content
  5. Apply any user-specified size adjustments

- The system automatically handles different input image sizes
- Masks are dynamically padded and centered for optimal alignment
- Transparent regions are properly preserved in the final output
- Background images are automatically scaled to match the person image
- Real-time preview updates as parameters are adjusted

