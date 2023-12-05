import os
import shutil
import random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix

# Get the current directory
directory = os.getcwd()

# Paths to organized data folders
train_images_folder = directory+'\\segmentation_data\\normal\\train_extended\\images\\'
train_masks_folder = directory+'\\segmentation_data\\normal\\train_extended\\masks\\'
val_images_folder = directory+'\\segmentation_data\\normal\\validate\\images\\'
val_masks_folder = directory+'\\segmentation_data\\normal\\validate\\masks\\'

# Define the U-Net architecture for segmentation
def unet_model(input_shape):
    inputs = tf.keras.Input(shape=input_shape)
    
    # Contracting Path
    conv1 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(inputs)
    conv1 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(conv1)
    pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(pool1)
    conv2 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(conv2)
    pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)
    
    conv3 = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same')(pool2)
    conv3 = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same')(conv3)
    pool3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv3)
    
    conv4 = tf.keras.layers.Conv2D(512, 3, activation='relu', padding='same')(pool3)
    conv4 = tf.keras.layers.Conv2D(512, 3, activation='relu', padding='same')(conv4)
    pool4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv4)
    
    conv5 = tf.keras.layers.Conv2D(1024, 3, activation='relu', padding='same')(pool4)
    conv5 = tf.keras.layers.Conv2D(1024, 3, activation='relu', padding='same')(conv5)
    
    up6 = tf.keras.layers.UpSampling2D(size=(2, 2))(conv5)
    up6 = tf.keras.layers.Conv2D(512, 2, activation='relu', padding='same')(up6)
    merge6 = tf.keras.layers.concatenate([conv4, up6], axis=3)
    conv6 = tf.keras.layers.Conv2D(512, 3, activation='relu', padding='same')(merge6)
    conv6 = tf.keras.layers.Conv2D(512, 3, activation='relu', padding='same')(conv6)
    
    up7 = tf.keras.layers.UpSampling2D(size=(2, 2))(conv6)
    up7 = tf.keras.layers.Conv2D(256, 2, activation='relu', padding='same')(up7)
    merge7 = tf.keras.layers.concatenate([conv3, up7], axis=3)
    conv7 = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same')(merge7)
    conv7 = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same')(conv7)
    
    up8 = tf.keras.layers.UpSampling2D(size=(2, 2))(conv7)
    up8 = tf.keras.layers.Conv2D(128, 2, activation='relu', padding='same')(up8)
    merge8 = tf.keras.layers.concatenate([conv2, up8], axis=3)
    conv8 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(merge8)
    conv8 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(conv8)
    
    up9 = tf.keras.layers.UpSampling2D(size=(2, 2))(conv8)
    up9 = tf.keras.layers.Conv2D(64, 2, activation='relu', padding='same')(up9)
    merge9 = tf.keras.layers.concatenate([conv1, up9], axis=3)
    conv9 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(merge9)
    conv9 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(conv9)
    
    outputs = tf.keras.layers.Conv2D(1, 1, activation='sigmoid')(conv9)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

input_shape = (256, 256, 3)  # Adjust this based on your image size and channels
model = unet_model(input_shape)
opt = tf.keras.optimizers.Adam(learning_rate=0.0001)
model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

# Create data generators
data_gen_args = dict(rescale=1.0 / 255)
image_datagen = ImageDataGenerator(**data_gen_args)
mask_datagen = ImageDataGenerator(**data_gen_args)

batch_size = 4

train_image_generator = image_datagen.flow_from_directory(
    train_images_folder, class_mode=None, seed=42, target_size=input_shape[:2], batch_size=batch_size
)
train_mask_generator = mask_datagen.flow_from_directory(
    train_masks_folder, class_mode=None, seed=42, target_size=input_shape[:2], batch_size=batch_size
)

val_image_generator = image_datagen.flow_from_directory(
    val_images_folder, class_mode=None, seed=42, target_size=input_shape[:2], batch_size=batch_size
)
val_mask_generator = mask_datagen.flow_from_directory(
    val_masks_folder, class_mode=None, seed=42, target_size=input_shape[:2], batch_size=batch_size
)

# Combine generators into one which yields both images and masks
train_generator = zip(train_image_generator, train_mask_generator)
val_generator = zip(val_image_generator, val_mask_generator)
    
# Calculate steps per epoch for training
num_samples_train = len(train_image_generator.filenames)
steps_per_epoch_train = num_samples_train // batch_size

# Calculate steps per epoch for validation
num_samples_val = len(val_image_generator.filenames)
steps_per_epoch_val = num_samples_val // batch_size

# Train the model on the training dataset
history_train = model.fit(train_generator, validation_data = val_generator, epochs=1, steps_per_epoch=steps_per_epoch_train, validation_steps = steps_per_epoch_val, workers=1, use_multiprocessing=False)

# Save the model
model.save(directory+'\\segmentation_model_extended.h5')

print(history_train.history['accuracy'])
print(history_train.history['loss'])

plt.plot(history_train.history['accuracy'])
plt.plot(history_train.history['loss'])
plt.title('Training Accuracy and Loss')
plt.ylabel('Accuracy & Loss')
plt.xlabel('Epoch')
plt.legend(['Accuracy', 'Loss'], loc='upper left')
plt.savefig(directory+'\\training_graph.png')
plt.show()