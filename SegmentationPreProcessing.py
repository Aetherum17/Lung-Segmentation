import os
import shutil
import random
import numpy as np
from PIL import Image

# Get the current directory
directory = os.getcwd()

# Paths data folders
images_folder = directory+'\\COVID-19_Radiography_Dataset\\Normal\\images\\'
masks_folder = directory+'\\COVID-19_Radiography_Dataset\\Normal\\masks\\'

# Paths to organized data folders
train_images_folder = directory+'\\segmentation_data\\normal\\train\\images\\Subfolder'
train_masks_folder = directory+'\\segmentation_data\\normal\\train\\masks\\Subfolder'
val_images_folder = directory+'\\segmentation_data\\normal\\validate\\images\\Subfolder'
val_masks_folder = directory+'\\segmentation_data\\normal\\validate\\masks\\Subfolder'
test_images_folder = directory+'\\segmentation_data\\normal\\test\\images\\Subfolder'
test_masks_folder = directory+'\\segmentation_data\\normal\\test\\masks\\Subfolder'

# List all image files in the images folder and randomize them
image_files = os.listdir(images_folder)
random.shuffle(image_files)

# Define split ratios, calculate split indices and perform the splitting itself
train_ratio = 0.50
val_ratio = 0.20
test_ratio = 0.30

num_samples = len(image_files)
num_train = int(num_samples * train_ratio)
num_val = int(num_samples * val_ratio)

train_files = image_files[:num_train]
val_files = image_files[num_train:num_train + num_val]
test_files = image_files[num_train + num_val:]

# This function will copy the files to their new folders
def copy_files(file_list, source_folder, dest_folder):
    for file_name in file_list:
        src_path = os.path.join(source_folder, file_name)
        dest_path = os.path.join(dest_folder, file_name)
        shutil.copy(src_path, dest_path)

# All input images should also be normalized
def normalize_image(image):
    image = np.array(image)
    normalized_image = (image - image.min()) / (image.max() - image.min())
    return normalized_image

# This function will process the images by performing normalization 
def process_images(file_list, source_folder, dest_folder):
    for file_name in file_list:
        src_path = os.path.join(source_folder, file_name)
        dest_path = os.path.join(dest_folder, file_name)
        
        image = Image.open(src_path)
        resized_image = image.resize((256, 256))  # Resize to 256x256
        normalized_image = normalize_image(resized_image)
        
        normalized_image = (normalized_image * 255).astype(np.uint8)
        normalized_image = Image.fromarray(normalized_image)
        
        normalized_image.save(dest_path)

# Create new directories
os.makedirs(train_images_folder, exist_ok=True)
os.makedirs(train_masks_folder, exist_ok=True)
os.makedirs(val_images_folder, exist_ok=True)
os.makedirs(val_masks_folder, exist_ok=True)
os.makedirs(test_images_folder, exist_ok=True)
os.makedirs(test_masks_folder, exist_ok=True)

# Execute the image processing
process_images(train_files, images_folder, train_images_folder)
process_images(val_files, images_folder, val_images_folder)
process_images(test_files, images_folder, test_images_folder)

# Copy the files to their new folders
copy_files(train_files, masks_folder, train_masks_folder)
copy_files(val_files, masks_folder, val_masks_folder)
copy_files(test_files, masks_folder, test_masks_folder)