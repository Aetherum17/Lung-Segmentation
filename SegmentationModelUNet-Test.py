import os
import shutil
import random
import numpy as np
from PIL import Image

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix

# Get the current directory
directory = os.getcwd()

input_shape = (256, 256, 3)  # Adjust this based on your image size and channels

# Load the saved model
model = tf.keras.models.load_model(directory+'\\segmentation_model_extended.h5')

# Paths to organized data folders
test_images_folder = directory+'\\segmentation_data\\normal\\test\\images\\'
test_masks_folder = directory+'\\segmentation_data\\normal\\test\\masks\\'

# Create a directory to save the predicted masks
predicted_masks_folder = directory+'\\segmentation_data\\normal\\test\\prediction\\'
os.makedirs(predicted_masks_folder, exist_ok=True)

# Create data generators
data_gen_args = dict(rescale=1.0 / 255)
image_datagen = ImageDataGenerator(**data_gen_args)
mask_datagen = ImageDataGenerator(**data_gen_args)

batch_size = 4

test_image_generator = image_datagen.flow_from_directory(
    test_images_folder, class_mode=None, seed=42, target_size=input_shape[:2], batch_size=batch_size
)
test_mask_generator = mask_datagen.flow_from_directory(
    test_masks_folder, class_mode=None, seed=42, target_size=input_shape[:2], batch_size=batch_size
)

# Combine generators into one which yields both images and masks
test_generator = zip(test_image_generator, test_mask_generator)

# Calculate the number of batches based on the smaller generator
num_batches = min(len(test_image_generator), len(test_mask_generator))

iou_scores_test = []
dice_scores_test = []

for i, (images, masks) in enumerate(test_generator):
    if i >= num_batches:
        break
    
    predicted_masks = model.predict(images)
    
    print(f'Processing batch {i + 1}/{len(test_image_generator)}')
    
    for j in range(predicted_masks.shape[0]):
        true_mask = masks[j]
        predicted_mask = predicted_masks[j]
        
        true_mask = true_mask[:, :, 0]
        
        true_mask = true_mask.flatten()
        predicted_mask = (predicted_mask >= 0.5).astype(int).flatten()
        
        # Calculate Intersection over Union (IoU)
        intersection = np.sum(true_mask * predicted_mask)
        union = np.sum(true_mask) + np.sum(predicted_mask) - intersection
        iou = intersection / union
        iou_scores_test.append(iou)
        
        # Dice score formula
        dice = (2 * np.sum(true_mask * predicted_mask)) / (np.sum(true_mask) + np.sum(predicted_mask))
        dice_scores_test.append(dice)
        
        # Get the original mask file name
        mask_file_name = os.path.basename(test_mask_generator.filenames[i * batch_size + j])
        
        # Create the predicted mask file name
        predicted_mask_file_name = mask_file_name.replace('.png', '_predicted.png')
        
        # Save predicted mask as an image with the new file name
        predicted_mask_image = (predicted_mask * 255).astype(np.uint8).reshape(input_shape[:2])
        predicted_mask_image_pil = Image.fromarray(predicted_mask_image)
        predicted_mask_image_pil.save(os.path.join(predicted_masks_folder, predicted_mask_file_name))

# Calculate the average Dice Coefficient score for the entire test dataset
average_dice_score_test = np.mean(dice_scores_test)
average_iou_score_test = np.mean(iou_scores_test)
print(f'Average IoU on Test Data: {average_iou_score_test:.4f}')
print(f'Average Dice Coefficient on Test Data: {average_dice_score_test:.4f}')
