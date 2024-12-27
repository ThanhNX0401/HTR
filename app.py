import sys
import os
import shutil
# Import necessary libraries
import gradio as gr
import cv2
import numpy as np
import pandas as pd

# import EasyOCR.easyocr
# from EasyOCR.easyocr.detection import get_detector, get_textbox
from mltu.inferenceModel import OnnxInferenceModel
from mltu.utils.text_utils import ctc_decoder
import matplotlib.pyplot as plt

import csv
import torch
from tqdm import tqdm
from gradio_image_prompter import ImagePrompter
from PIL import Image
from mltu.transformers import ImageResizer

from onnxtr.io import DocumentFile
from onnxtr.models import ocr_predictor, fast_base, linknet_resnet50
from onnxtr.models.detection import detection_predictor

# # Initialize EasyOCR reader
# reader = EasyOCR.easyocr.Reader(
#     lang_list=["vi"],
#     # detector=False,
# )
# reader.get_detector, reader.get_textbox = get_detector, get_textbox
# # reader.detector = reader.initDetector("/kaggle/input/ededeede/detector10.pth")


# Define the model class for recognition
class ImageToWordModel(OnnxInferenceModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def predict(self, image: np.ndarray): #(h,w,3)
        # image = cv2.resize(image, self.input_shapes[0][1:3][::-1])
        # image_pred = np.expand_dims(image, axis=0).astype(np.float32)
        # # image = ImageResizer.resize_maintaining_aspect_ratio(image, *self.input_shapes[0][1:3][::-1])
        # # image_pred = np.expand_dims(image, axis=0).astype(np.float32)
        # # image_pred = np.expand_dims(image_pred, axis=0)
        # # print(image_pred.shape)
        # preds = self.model.run(self.output_names, {self.input_names[0]: image_pred})[0]
        # text = ctc_decoder(preds, self.metadata["vocab"])[0]
        # return text
        image = ImageResizer.resize_maintaining_aspect_ratio(image, *self.input_shapes[0][1:3][::-1])
        image_pred = np.expand_dims(image, axis=0).astype(np.float32)
        
        preds = self.model.run(self.output_names, {self.input_names[0]: image_pred})[0]
        
        text = ctc_decoder(preds, self.metadata["vocab"])[0]
    
        return text

det_model = linknet_resnet50("/content/drive/MyDrive/final_model/detection/linknet_resnet50_100.onnx")
detection = detection_predictor(arch=det_model)
# detection.model.postprocessor.bin_thresh = 0.3
# detection.model.postprocessor.box_thresh = 0.1

def calculate_avg_height_and_midpoints(df):
    heights = []
    midpoints = []
    
    for _, row in df.iterrows():
        coordinates = list(map(int, row['coordinates'].split(',')))
        x_min, y_min, x_max, y_max = coordinates[0], coordinates[1], coordinates[2], coordinates[3]
        
        height = y_max - y_min
        midpoint = (y_min + y_max) / 2
        heights.append(height)
        midpoints.append(midpoint)
    
    avg_height = np.mean(heights)
    return avg_height, midpoints

def group_words_by_flexible_lines(df, avg_height, midpoints):
    lines = []
    
    for i, row in df.iterrows():
        # Extract word coordinates and midpoint
        coordinates = list(map(int, row['coordinates'].split(',')))
        x_min, y_min, x_max, y_max = coordinates[0], coordinates[1], coordinates[2], coordinates[3]
        
        midpoint = midpoints[i]
        
        # Find or create a line based on vertical proximity (within a threshold of avg height)
        added_to_line = False
        for line in lines:
            line_midpoint = np.mean([word['midpoint'] for word in line])  # Average of midpoints in the line
            if abs(midpoint - line_midpoint) < avg_height * 0.6:  # Threshold set as 60% of average height
                line.append({'prediction': row['prediction'], 'coordinates': (x_min, y_min, x_max, y_max), 'midpoint': midpoint})
                added_to_line = True
                break
        
        # If no line found, create a new line
        if not added_to_line:
            lines.append([{'prediction': row['prediction'], 'coordinates': (x_min, y_min, x_max, y_max), 'midpoint': midpoint}])

    return lines

def sort_lines_and_words(lines):
    # Sort words within each line by their x_min (horizontal position)
    for line in lines:
        line.sort(key=lambda word: word['coordinates'][0])  # Sort by x_min

    # Sort lines themselves by their average midpoint (top to bottom order)
    lines.sort(key=lambda line: np.mean([word['midpoint'] for word in line]))  # Sort by average midpoint

    return lines

def reconstruct_paragraph(lines):
    paragraph = []
    for line in lines:
        line_text = ' '.join([word['prediction'] for word in line])  # Join the words in each line
        paragraph.append(line_text)
    return '\n'.join(paragraph)

def clear_old_files(output_cropped_dir='cropped_images1', output_bboxes_dir='bbox_image1', csv_file_path='cropped_images_coordinates.csv'):
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
    
    # print(image)
    # if isinstance(image, np.ndarray):
    #     img = Image.fromarray(image)
    #     print("1")
    # # If the image is a Pillow Image, use it directly
    # elif isinstance(image, Image.Image):
    #     img = image
    #     print("2")
    # # If it's a file path, open the image
    # else:
    #     img = Image.open(image)
    print("start")
    
    img = np.array(image)
    det_result=detection([img])

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
                cv2.rectangle(img_with_boxes, (x1_int, y1_int), (x2_int, y2_int), (0, 255, 0), 2)
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
    
    print("5")
    # Load the original CSV file
    df = pd.read_csv(csv_filepath)

    # Initialize the model
    model = ImageToWordModel(model_path="/content/drive/MyDrive/final_model/resnet/model.onnx")

    # Add a new column 'prediction' if it doesn't exist
    if 'prediction' not in df.columns:
        df['prediction'] = ""

    # Loop through the DataFrame, predict for each image, and update the CSV
    for idx, row in df.iterrows():
        image_path = row['crop_image_path']
        image = cv2.imread(image_path.replace("\\", "/"))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Step 3: Convert grayscale to a 3-channel image by stacking the grayscale image along the third axis
        image = np.stack((gray_image,)*3, axis=-1)

        # image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # print(image_gray.shape, image_gray.size)
        # # Expand dimensions to make it (height, width, 1)
        # # image1 = image_gray[:, :, None]

        # image1 = np.expand_dims(image_gray, axis=-1)
        # print(image1.shape)
        # Make a prediction
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

    return reconstructed_paragraph, img_with_boxes


# Define a variable to store the points
points_variable = None
# def process_image(image):
#     return text=get_text(image)

# Function to process the inputs and return cropped images or original image
def process_and_crop(prompts):
    global points_variable
    image = prompts["image"]  # Get the uploaded image
    points_variable = prompts["points"]  # Get the points
    
    if points_variable is None or len(points_variable) == 0:
        # If points_variable is empty, return the text from the original image
        return process_image(image)  # No gallery output, only text

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
    return cropped_with_text[0]  # Gallery output with labels, no text output

# def prepare_image_choices(uploaded_files):
#     images = []
#     for file in uploaded_files:
#         img = Image.open(file.name).convert("RGB")
#         images.append(img)
#     return images

# Theme and JavaScript for Gradio app
theme = gr.themes.Base(
    primary_hue="rose",
    secondary_hue="rose",
    text_size="lg",
).set(
    body_text_color='*neutral_950',
    body_text_color_dark='*neutral_50',
    block_info_text_color='black',
    block_info_text_color_dark='white',
    block_title_text_color='black',
    block_title_text_color_dark='white',
    # panel_background_fill_dark='#cccccc'
)


js_func = """
function refresh() {
    const url = new URL(window.location);

    if (url.searchParams.get('__theme') !== 'dark') {
        url.searchParams.set('__theme', 'dark');
        window.location.href = url.href;
    }
}
"""

# Gradio app interface
with gr.Blocks(js=js_func, theme=theme) as demo: #, 
    with gr.Column(variant="panel"):
        gr.Markdown("<center><h1 style='font-size: 40px;'>Vietnamese Handwritten Text Recognition Model</h1><center>")
        gr.Markdown("<center><span style='font-size:24px;'>This is a project to run Vietnamese handwritten recognition model</span><center>")

    
    with gr.Row():
        with gr.Column(variant="panel", scale=1):
            gr.Markdown("<h2 style='font-size: 24px;'>Image input</h2>")
            image_input = ImagePrompter(show_label=False)

            with gr.Row():
                clear_button = gr.Button("Clear", variant="secondary")
                submit_button = gr.Button("Submit", variant="primary")

        with gr.Column(variant="panel", scale=1):
            gr.Markdown("<h2 style='font-size: 24px;'>Box Image</h2>")
            outputs = gr.Image()
            
        with gr.Column(variant="panel", scale=1):
            gr.Markdown("<h2 style='font-size: 24px;'>Box Image</h2>")
            result = gr.Textbox(label="Output Text", placeholder="The regconition text will appear here...", lines=5)
        
        # Set up the button click event for the Submit button
        submit_button.click(
            fn=process_and_crop,
            inputs=image_input,
            outputs=[result, outputs]
        )

        # Set up the button click event for the Clear button
        clear_button.click(
            fn=lambda: (None, ""),  # Clear outputs
            inputs=None,
            outputs=[outputs, result]
        )

print("Generating Gradio app LINK:")
demo.launch(share=True)
