# 3D Person Segmentation and Anaglyph Generation

## Lab Report

### Introduction
This project implements a sophisticated 3D image processing system that combines person segmentation with stereoscopic and anaglyph image generation. The main objectives were to:
1. Accurately segment people from images using advanced AI models
2. Generate stereoscopic 3D effects from 2D images
3. Create red-cyan anaglyph images for 3D viewing
4. Provide an interactive web interface for real-time processing

### Methodology

#### Tools and Technologies Used
- **SegFormer (nvidia/segformer-b0)**: State-of-the-art transformer-based model for semantic segmentation
- **PyTorch**: Deep learning framework for running the SegFormer model
- **OpenCV**: Image processing operations and mask refinement
- **Gradio**: Web interface development
- **NumPy**: Efficient array operations for image manipulation
- **PIL (Python Imaging Library)**: Image loading and basic transformations

#### Implementation Steps

1. **Person Segmentation**
   - Utilized SegFormer model fine-tuned on ADE20K dataset
   - Applied post-processing with erosion and Gaussian blur for mask refinement
   - Implemented mask scaling and centering for various input sizes

2. **Stereoscopic Processing**
   - Created depth simulation through horizontal pixel shifting
   - Implemented parallel view stereo pair generation
   - Added configurable interaxial distance for 3D effect adjustment

3. **Anaglyph Generation**
   - Combined left and right eye views into red-cyan anaglyph
   - Implemented color channel separation and recombination
   - Added background image support with proper masking

4. **User Interface**
   - Developed interactive web interface using Gradio
   - Added real-time parameter adjustment capabilities
   - Implemented support for custom background images

### Results

The system produces three main outputs:
1. Segmentation mask showing the isolated person
2. Side-by-side stereo pair for parallel viewing
3. Red-cyan anaglyph image for 3D glasses viewing

Key Features:
- Adjustable person size (10-200%)
- Configurable interaxial distance (0-10 pixels)
- Optional custom background support
- Real-time processing and preview

### Discussion

#### Technical Challenges
1. **Mask Alignment**: Ensuring proper alignment between segmentation masks and background images required careful consideration of image dimensions and aspect ratios.
2. **Stereo Effect Quality**: Balancing the interaxial distance for comfortable viewing while maintaining the 3D effect.
3. **Performance Optimization**: Efficient processing of large images while maintaining real-time interaction.

#### Learning Outcomes
- Deep understanding of stereoscopic image generation
- Experience with state-of-the-art segmentation models
- Practical knowledge of image processing techniques
- Web interface development for ML applications

### Conclusion

This project successfully demonstrates the integration of modern AI-powered segmentation with classical stereoscopic image processing techniques. The system provides an accessible way to create 3D effects from regular 2D images.

#### Future Work
- Implementation of depth-aware 3D effect generation
- Support for video processing
- Additional 3D viewing formats (side-by-side, over-under)
- Enhanced background replacement options
- Mobile device optimization

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

1. **Segmentation Mask**: Shows the isolated person
2. **Stereo Pair**: Side-by-side stereo image for parallel viewing
3. **Anaglyph**: Red-cyan 3D image viewable with anaglyph glasses

