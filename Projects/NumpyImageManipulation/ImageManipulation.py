import numpy as np
import os
from PIL import Image


image_path = "sample_image.png"
directory_path = 'ManipulatedImages'

def rotate_image():
    """
    This function rotates an image by 90, 180, and 270 degrees counterclockwise.
    It converts the rotated arrays back to images and saves them in the directory.
    """
    rotated_90 = np.rot90(img_array, k=1)  # 90 degrees counterclockwise
    rotated_180 = np.rot90(img_array, k=2)  # 180 degrees counterclockwise
    rotated_270 = np.rot90(img_array, k=3)  # 270 degrees counterclockwise

    # Convert the rotated arrays back to images
    rotated_90_img = Image.fromarray(rotated_90)
    rotated_180_img = Image.fromarray(rotated_180)
    rotated_270_img = Image.fromarray(rotated_270)

    # Save the images with appropriate filenames
    rotated_90_img.save(os.path.join(directory_path, "rotated_90.png"))
    rotated_180_img.save(os.path.join(directory_path, "rotated_180.png"))
    rotated_270_img.save(os.path.join(directory_path, "rotated_270.png"))

def adjust_image_brightness():
    """
    Adjusts the brightness of the image by brightness_factor and saves it
    """
    brightened_array = img_array * brightness_factor

    # Clip the values to ensure they stay within the range [0, 255]
    brightened_array = np.clip(brightened_array, 0, 255)

    brightened_image = Image.fromarray(brightened_array.astype(np.uint8))
    brightened_image.save(os.path.join(directory_path, "brightened_image.png"))


def crop_center():
    """
    Crops 50% area of the image and saves it
    """
    # Get original image dimensions
    original_height, original_width = img_array.shape[:2]

    # Calculate dimensions for the cropped area (50% of original size)
    new_height = original_height // 2
    new_width = original_width // 2

    # Calculate starting and ending points for cropping
    start_y = (original_height - new_height) // 2
    start_x = (original_width - new_width) // 2
    end_y = start_y + new_height
    end_x = start_x + new_width

    # Crop the central part using NumPy slicing
    cropped_array = img_array[start_y:end_y, start_x:end_x]

    # Convert the cropped array back to a PIL image
    cropped_img = Image.fromarray(cropped_array)
    cropped_img.save(os.path.join(directory_path, "cropped_img.png"))

def mathematical_operations():
    """
    Uses Mathematical Numpy functions to calculate the mean, max, min and sum.
    """
    print(f'Mean: {np.mean(img_array)}')
    print(f'Max: {np.max(img_array)}')
    print(f'Min: {np.min(img_array)}')
    print(f'Sum: {np.sum(img_array)}')



if __name__ == '__main__':
    """ Step 1: Load an Image as a NumPy Array """
    image = Image.open(image_path)  # Load image
    img_array = np.array(image)  # Convert image to array

    """ Step 2: Rotate the Image """
    rotate_image()

    """ Step 3: Adjust Image Brightness """
    brightness_factor = 1.5
    adjust_image_brightness()

    """ Step 4: Crop the central portion """
    crop_center()

    """ Step 5: Apply Mathematical operations """
    mathematical_operations()
