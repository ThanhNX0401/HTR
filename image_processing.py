import os
import shutil
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import csv
from typing import Tuple, List, Union
from gradio_image_prompter import ImagePrompter
import gdown
from docx import Document
from model import ImageToWordModel, DetectionModel
from utils import (
    calculate_avg_height_and_midpoints,
    group_words_by_flexible_lines,
    sort_lines_and_words,
    reconstruct_paragraph
)

def export_to_word(text, file_path="output_text.docx"):
    # Create a new Document
    doc = Document()
    # Add the extracted text to the Word document
    doc.add_paragraph(text)
    # Save the document to the specified path
    doc.save(file_path)
    return file_path

def clear_old_files(output_cropped_dir='cropped_images', output_bboxes_dir='bbox_image', csv_file_path='cropped_images_coordinates.csv'):
    # Remove old cropped images directory
    if os.path.exists(output_cropped_dir):
        shutil.rmtree(output_cropped_dir)  # Remove the entire directory

    # Remove old bounding box images directory
    if os.path.exists(output_bboxes_dir):
        shutil.rmtree(output_bboxes_dir)  # Remove the entire directory

    # Remove old CSV file
    if os.path.exists(csv_file_path):
        os.remove(csv_file_path)


# Function to process the uploaded image
def process_image(image):
    # Convert the uploaded image to grayscale
    clear_old_files()
    print("start")
    
    img = np.array(image)
    detection_model = DetectionModel("models_train/linknet_resnet50_bestb.onnx")
    det_result = detection_model.predict(img)

    output_folder='cropped_images'
    os.makedirs(output_folder, exist_ok=True)
    csv_filename='bounding_boxes.csv'

    cropped_image_data = []
    img_with_boxes = img.copy()
    # print("det:",det_result)
    for idx, box in enumerate(det_result[0]):
        # Extract the bounding box coordinates (rescale to image size if necessary)
        x1, y1, x2, y2 = (box[0] * img.shape[1], box[1] * img.shape[0],
                          box[2] * img.shape[1], box[3] * img.shape[0])
        score = box[4]  # Confidence score, if you want to use it

        # Check if coordinates are valid (non-None and form a valid bounding box)
        if x1 is not None and x2 is not None and y1 is not None and y2 is not None:
            # Convert coordinates to integers
            x1_int, y1_int, x2_int, y2_int = int(x1), int(y1), int(x2), int(y2)
        
            # Check if the integer coordinates form a valid box (x2 > x1 and y2 > y1)
            if x1_int >= 0 and y1_int >= 0 and x2_int > x1_int and y2_int > y1_int:
                # Crop the image
                cropped_img = img[y1_int:y2_int, x1_int:x2_int]
    
                # Draw the bounding box on the image
                cv2.rectangle(img_with_boxes, (x1_int, y1_int), (x2_int, y2_int), (255, 0, 0), 2)
            else:
                # If coordinates are invalid, use the full image
                continue
        else:
            # If any coordinate is None, use the full image
            cropped_img = img

        
        # Save the cropped image
        cropped_filename = f'crop_{idx}.jpg'
        cropped_image_path = os.path.join(output_folder, cropped_filename)
        # print(f'{int(x1)},{int(y1)},{int(x2)},{int(y2)}')
        
        cv2.imwrite(cropped_image_path, cropped_img)
        coordinates_str = f'{int(x1)},{int(y1)},{int(x2)},{int(y2)}'

        # Append full path and bounding box details to CSV data
        cropped_image_data.append([cropped_image_path, coordinates_str])

    # Save the bounding box details to a CSV file
    csv_filepath = os.path.join(output_folder, csv_filename)
    with open(csv_filepath, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['crop_image_path', 'coordinates'])  # CSV header
        writer.writerows(cropped_image_data)

    # Convert img_with_boxes back to PIL format to return
    img_with_boxes = Image.fromarray(img_with_boxes)
    
    # Load the original CSV file
    df = pd.read_csv(csv_filepath)

    # Initialize the model
    model = ImageToWordModel(model_path="models_train/model_vgg.onnx")

    # Add a new column 'prediction' if it doesn't exist
    if 'prediction' not in df.columns:
        df['prediction'] = ""

    # Loop through the DataFrame, predict for each image, and update the CSV
    for idx, row in df.iterrows():
        image_path = row['crop_image_path']
        image = cv2.imread(image_path.replace("\\", "/"))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        # image = np.stack((gray_image,)*3, axis=-1)
        
        prediction_text = model.predict(image)
        df.at[idx, 'prediction'] = prediction_text

    # Calculate the average height and midpoints of word bounding boxes
    avg_height, midpoints = calculate_avg_height_and_midpoints(df)

    # Group words into flexible lines based on midpoints and average height
    lines = group_words_by_flexible_lines(df, avg_height, midpoints)

    # Sort the lines and words within them
    sorted_lines = sort_lines_and_words(lines)

    # Reconstruct the paragraph from the sorted lines and words
    reconstructed_paragraph = reconstruct_paragraph(sorted_lines)
    
    word_file_path = export_to_word(reconstructed_paragraph)

    return reconstructed_paragraph, img_with_boxes, word_file_path


# Function to process the inputs and return cropped images or original image
def process_and_crop(prompts):
    # global points_variable
    image = prompts["image"]  # Get the uploaded image
    points_variable = prompts["points"]  # Get the points
    
    if points_variable is None or len(points_variable) == 0:
        # If points_variable is empty, return the text from the original image
        return process_image(image) 

    # If points_variable has coordinates, crop the image based on each bounding box
    cropped_images = []
    img_array = np.array(image)  # Convert image to a NumPy array for cropping

    for box in points_variable:
        # Extract bounding box coordinates: [x_min, y_min, 2.0, x_max, y_max, 3.0]
        x_min, y_min, _, x_max, y_max, _ = box

        # Convert to integers for cropping
        x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)

        # Crop the image based on the coordinates
        cropped_image = img_array[y_min:y_max, x_min:x_max]

        # Convert back to PIL Image
        cropped_pil = Image.fromarray(cropped_image)
        cropped_images.append(cropped_pil)

    # Process each cropped image to extract text
    cropped_with_text = [process_image(img) for img in cropped_images]
    print("out")
    return cropped_with_text[0]