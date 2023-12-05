from PIL import Image, ImageOps
import os

# Get the current directory
directory = os.getcwd()

# Paths to the original images and masks folders
images_folder = directory+'\\segmentation_data\\normal\\train\\images\\Subfolder'
masks_folder = directory+'\\segmentation_data\\normal\\train\\masks\\Subfolder'

images_output_folder = directory+'\\segmentation_data\\normal\\train_extended\\images\\Subfolder'
masks_output_folder = directory+'\\segmentation_data\\normal\\train_extended\\masks\\Subfolder'
os.makedirs(images_output_folder, exist_ok=True)
os.makedirs(masks_output_folder, exist_ok=True)

# Get a list of filenames in the images folder
image_filenames = os.listdir(images_folder)
total_images = len(image_filenames)

# Process each image
for idx, image_filename in enumerate(image_filenames, 1):
    # Form the paths for the original image and mask
    original_image_path = os.path.join(images_folder, image_filename)
    original_mask_path = os.path.join(masks_folder, image_filename)
    
    # Open the original image and mask
    original_image = Image.open(original_image_path)
    original_mask = Image.open(original_mask_path)
    
    # Transform Images and Masks
    rotated_image_90 = original_image.rotate(90, expand=True)
    rotated_mask_90 = original_mask.rotate(90, expand=True)
    
    rotated_image_180 = original_image.rotate(180, expand=True)
    rotated_mask_180 = original_mask.rotate(180, expand=True)
    
    rotated_image_270 = original_image.rotate(270, expand=True)
    rotated_mask_270 = original_mask.rotate(270, expand=True)
    
    mirrored_image = ImageOps.mirror(original_image)
    mirrored_mask = ImageOps.mirror(original_mask)
    
    # Save the image and mask
    original_image.save(os.path.join(images_output_folder, image_filename))
    original_mask.save(os.path.join(masks_output_folder, image_filename))
    
    rotated_image_90.save(os.path.join(images_output_folder, os.path.splitext(image_filename)[0]+"_rotated_90.png"))
    rotated_mask_90.save(os.path.join(masks_output_folder, os.path.splitext(image_filename)[0]+"_rotated_90.png"))
    
    rotated_image_180.save(os.path.join(images_output_folder, os.path.splitext(image_filename)[0]+"_rotated_180.png"))
    rotated_mask_180.save(os.path.join(masks_output_folder, os.path.splitext(image_filename)[0]+"_rotated_180.png"))
    
    rotated_image_270.save(os.path.join(images_output_folder, os.path.splitext(image_filename)[0]+"_rotated_270.png"))
    rotated_mask_270.save(os.path.join(masks_output_folder, os.path.splitext(image_filename)[0]+"_rotated_270.png"))
    
    mirrored_image.save(os.path.join(images_output_folder, os.path.splitext(image_filename)[0]+"_mirrored.png"))
    mirrored_mask.save(os.path.join(masks_output_folder, os.path.splitext(image_filename)[0]+"_mirrored.png"))
    
    percentage_done = (idx / total_images) * 100
    print(f"Processed: {image_filename} - {percentage_done:.2f}% done")

print("Done")