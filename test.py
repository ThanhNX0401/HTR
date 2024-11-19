import cv2
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
import random
import pandas as pd
from PIL import Image, ImageDraw

# Paths for your images
sentence_images_folder = '/kaggle/input/datasetanh/Dataset/InkData_word'  # Folder containing sentence images
background_images_path = '/kaggle/working/resized_images/*.jpg'  # Assuming resized images are in this folder
output_image_folder = '/kaggle/working/processed_images'  # Folder for combined images
output_txt_folder = '/kaggle/working/coordinates'  # Folder for coordinate text files
csv_file_path = '/kaggle/input/your_labels.csv'  # Path to your CSV file with labels

# Ensure output directories exist
os.makedirs(output_image_folder, exist_ok=True)
os.makedirs(output_txt_folder, exist_ok=True)

# Load all background images
background_images = glob.glob(background_images_path)

# Load the CSV file into a DataFrame
labels_df = pd.read_csv(csv_file_path)

def boldWords(sentence_image):
    if sentence_image.shape[2] == 4:  # If RGBA
        gray = cv2.cvtColor(sentence_image[:, :, :3], cv2.COLOR_RGBA2GRAY)
    else:
        gray = cv2.cvtColor(sentence_image, cv2.COLOR_BGR2GRAY)

    _, binary_mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((10, 10), np.uint8)
    bold_mask = cv2.dilate(binary_mask, kernel, iterations=1)

    bolded_sentence_image = np.zeros_like(sentence_image)
    bolded_sentence_image[..., 3] = 0

    bolded_sentence_image[bold_mask == 255] = [0, 0, 0, 255] 
    return bolded_sentence_image

def get_random_position(box_width, box_height, placed_boxes, background_size=1024, center_area=768):
    margin = (background_size - center_area) // 2
    max_attempts = 100
    
    for _ in range(max_attempts):
        x = random.randint(margin, margin + center_area - box_width)
        y = random.randint(margin, margin + center_area - box_height)
        
        new_box = (x, y, x + box_width, y + box_height)
        if not any(boxes_overlap(new_box, existing_box) for existing_box in placed_boxes):
            return x, y
    
    return None, None

def boxes_overlap(box1, box2):
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    return not (x2_1 < x1_2 or x1_1 > x2_2 or y2_1 < y1_2 or y1_1 > y2_2)

# Repeat the process for 100 iterations
for iteration in range(1, 101):
    for background_image_path in background_images:
        background_image = cv2.imread(background_image_path)
        
        all_sentence_paths = glob.glob(os.path.join(sentence_images_folder, '*.png'))
        selected_sentences = random.sample(all_sentence_paths, min(10, len(all_sentence_paths)))
        
        placed_boxes = []
        sentence_coordinates = []
        
        # Process each selected sentence
        for img_path in selected_sentences:
            sentence_image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            bold_sentence_image = boldWords(sentence_image)
            
            scale_factor = min(200 / sentence_image.shape[1], 100 / sentence_image.shape[0])
            new_width = int(sentence_image.shape[1] * scale_factor)
            new_height = int(sentence_image.shape[0] * scale_factor)
            resized_sentence = cv2.resize(bold_sentence_image, (new_width, new_height))
            
            x_offset, y_offset = get_random_position(new_width, new_height, placed_boxes)
            
            if x_offset is not None:
                placed_boxes.append((x_offset, y_offset, x_offset + new_width, y_offset + new_height))
                sentence_label = os.path.basename(img_path).split('.')[0]  # Extract label from filename
                
                # Find the corresponding label in the CSV
                label_row = labels_df[labels_df['id'] == sentence_label]
                if not label_row.empty:
                    label = label_row['label'].values[0]  # Get the label from the DataFrame
                else:
                    label = "unknown"  # Default label if not found
                
                # Add the sentence to the background using alpha blending
                for c in range(0, 3):
                    background_image[y_offset:y_offset + new_height, x_offset:x_offset + new_width, c] = (
                        background_image[y_offset:y_offset + new_height, x_offset:x_offset + new_width, c] *
                        (1 - resized_sentence[..., 3] / 255.0) +
                        resized_sentence[..., c] * (resized_sentence[..., 3] / 255.0)
                    )
                
                # Save coordinates in the format: x1 y1 x2 y2 label
                sentence_coordinates.append((x_offset, y_offset, x_offset + new_width, y_offset + new_height, label))
        
        # Save the resulting image with iteration number
        output_image_path = os.path.join(output_image_folder, f"combined_{iteration:03d}_{os.path.basename(background_image_path)}")
        cv2.imwrite(output_image_path, background_image)
        
        # Save coordinates to a text file
        coord_file = os.path.join(output_txt_folder, f"combined_{iteration:03d}_{os.path.basename(background_image_path).rsplit('.', 1)[0]}_coordinates.txt")
        with open(coord_file, 'w') as f:
            for coord in sentence_coordinates:
                f.write(f"{coord[0]} {coord[1]} {coord[2]} {coord[3]} {coord[4]}\n")  # x1 y1 x2 y2 label
        
        # Display result
        plt.figure(figsize=(10, 10))
        plt.imshow(cv2.cvtColor(background_image, cv2.COLOR_BGR2RGB))
        plt.title("Combined Result")
        plt.axis("off")
        plt.show()
        
        print(f"Processed and saved: {output_image_path}")
        print(f"Coordinates saved: {coord_file}")

import concurrent.futures
import itertools
import warnings
import typing
import zipfile
import random
import glob
import json
import os

import tqdm
import imgaug
import PIL.Image
import numpy as np

# import keras_ocr.tools  
import models.keras_detection.tools
import models.keras_detection.detection

def get_detector_dataset(cache_dir=None, skip_illegible=False):
    """Get the ICDAR 2013 text segmentation dataset for detector
    training. Only the training set has the necessary annotations.
    For the test set, only segmentation maps are provided, which
    do not provide the necessary information for affinity scores.

    Args:
        cache_dir: The directory in which to store the data.
        skip_illegible: Whether to skip illegible characters.

    Returns:
        Lists of (image_path, lines, confidence) tuples. Confidence
        is always 1 for this dataset. We record confidence to allow
        for future support for weakly supervised cases.
    """
    if cache_dir is None:
        cache_dir = tools.get_default_cache_dir()

    training_images_dir = os.path.join(cache_dir, "processed_images")  # Adjust to your image folder
    training_gt_dir = os.path.join(cache_dir, "coordinates") 

    dataset = []
    for gt_filepath in glob.glob(os.path.join(training_gt_dir, "*.txt")):
        # Drop the '_coordinates' suffix to match the image file
        image_id = os.path.splitext(os.path.basename(gt_filepath))[0].replace('_coordinates', '')
        image_path = os.path.join(training_images_dir, image_id + ".jpg")
        lines = []
        
        with open(gt_filepath, "r", encoding="utf8") as f:
            current_line = []
            for raw_row in f.read().split("\n"):
                if raw_row.strip() == "":
                    if current_line:  # Only append if current_line is not empty
                        lines.append(current_line)
                        current_line = []
                else:
                    parts = raw_row.split()
                    if len(parts) < 5:
                        continue  # Skip lines that don't have enough parts
                    x1, y1, x2, y2 = map(int, parts[:4])
                    label = ' '.join(parts[4:])  # Join the rest as the label
                    current_line.append(
                        (np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]]), label)
                    )
            if current_line:  # Append the last line if it exists
                lines.append(current_line)

        # Some lines only have illegible characters and if skip_illegible is True,
        # then these lines will be blank.
        lines = [line for line in lines if line]
        dataset.append((image_path, lines, 1))
    
    return dataset

def get_detector_image_generator(
    labels,
    width,
    height,
    augmenter=None,
    area_threshold=0.5,
    focused=False,
    min_area=None,
    shuffle=True,
):
    """Generated augmented (image, lines) tuples from a list
    of (filepath, lines, confidence) tuples. Confidence is
    not used right now but is included for a future release
    that uses semi-supervised data.

    Args:
        labels: A list of (image, lines, confience) tuples.
        augmenter: An augmenter to apply to the images.
        width: The width to use for output images
        height: The height to use for output images
        area_threshold: The area threshold to use to keep
            characters in augmented images.
        min_area: The minimum area for a character to be
            included.
        focused: Whether to pre-crop images to width/height containing
            a region containing text.
        shuffle: Whether to shuffle the data on each iteration.
    """
    labels = labels.copy()
    for index in itertools.cycle(range(len(labels))):
        if index == 0 and shuffle:
            random.shuffle(labels)
        image_filepath, lines, confidence = labels[index]
        image = tools.read(image_filepath)
        if augmenter is not None:
            image, lines = tools.augment(
                boxes=lines,
                boxes_format="lines",
                image=image,
                area_threshold=area_threshold,
                min_area=min_area,
                augmenter=augmenter,
            )
        if focused:
            boxes = [tools.combine_line(line)[0] for line in lines]
            if boxes:
                selected = np.array(boxes[np.random.choice(len(boxes))])
                left, top = selected.min(axis=0).clip(0, np.inf).astype("int")
                if left > 0:
                    left -= np.random.randint(0, min(left, width / 2))
                if top > 0:
                    top -= np.random.randint(0, min(top, height / 2))
                image, lines = tools.augment(
                    boxes=lines,
                    augmenter=imgaug.augmenters.Sequential(
                        [
                            imgaug.augmenters.Crop(px=(int(top), 0, 0, int(left))),
                            imgaug.augmenters.CropToFixedSize(
                                width=width, height=height, position="right-bottom"
                            ),
                        ]
                    ),
                    boxes_format="lines",
                    image=image,
                    min_area=min_area,
                    area_threshold=area_threshold,
                )
        image, scale = tools.fit(
            image, width=width, height=height, mode="letterbox", return_scale=True
        )
        lines = tools.adjust_boxes(boxes=lines, boxes_format="lines", scale=scale)
        yield image, lines, confidence