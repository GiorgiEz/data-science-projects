# Image Processing Project

## Overview
This project provides a set of functions to manipulate images using Python. The operations include rotating, adjusting brightness, cropping, and performing mathematical operations on images. The manipulated images are saved in a specified directory.

## Features
1. **Rotate Image**: Rotate an image by 90, 180, and 270 degrees counterclockwise.
2. **Adjust Brightness**: Increase or decrease the brightness of an image by a specified factor.
3. **Crop Center**: Crop the central portion of an image (50% of its original area).
4. **Mathematical Operations**: Compute the mean, max, min, and sum of pixel values in an image.

## Prerequisites
1. Python 3
2. Required libraries:
   - `numpy`
   - `Pillow`
   - `os`

Install the dependencies using:
```bash
pip install numpy pillow
```

## Directory Structure
```
project_root/
├── ImageManipulation.py
├── sample_image.png
├── ManipulatedImages/  # Directory where manipulated images will be saved
└── README.md
```

## How to Use

### 1. Prepare the Image
Place the image you want to manipulate in the project directory. Update the `image_path` variable with the name of your image file.

### 2. Run the Script
Execute the script by running:
```bash
python ImageManipulation.py
```

### 3. Outputs
Manipulated images will be saved in the `ManipulatedImages` directory:
- `rotated_90.png`
- `rotated_180.png`
- `rotated_270.png`
- `brightened_image.png`
- `cropped_img.png`

### 4. View Mathematical Operations
The results of mathematical operations (mean, max, min, sum) will be printed to the console.

## Code Breakdown

### Functions
- **`rotate_image()`**
  Rotates the input image by 90, 180, and 270 degrees counterclockwise and saves the results.

- **`adjust_image_brightness()`**
  Adjusts the brightness of the image by multiplying pixel values by a `brightness_factor`. The factor can be customized in the script.

- **`crop_center()`**
  Crops the central 50% of the input image and saves the result.

- **`mathematical_operations()`**
  Performs mathematical calculations (mean, max, min, sum) on the image array and displays the results in the console.

### Entry Point
The `if __name__ == '__main__':` block ensures that the script runs the defined operations sequentially:
1. Loads the image.
2. Rotates the image.
3. Adjusts its brightness.
4. Crops the central portion.
5. Prints mathematical analysis of the image.
