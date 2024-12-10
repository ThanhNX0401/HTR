# Import necessary libraries
import gradio as gr
import cv2
import numpy as np
import pandas as pd
import os
import EasyOCR.easyocr
from EasyOCR.easyocr.detection import get_detector, get_textbox
from mltu.inferenceModel import OnnxInferenceModel
from mltu.utils.text_utils import ctc_decoder
import matplotlib.pyplot as plt

# Initialize EasyOCR reader
reader = EasyOCR.easyocr.Reader(
    lang_list=["vi"],
    detector=False,
)
reader.get_detector, reader.get_textbox = get_detector, get_textbox
reader.detector = reader.initDetector("/kaggle/input/ededeede/detector10.pth")

# Define the model class for recognition
class ImageToWordModel(OnnxInferenceModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def predict(self, image: np.ndarray):
        image = cv2.resize(image, self.input_shapes[0][1:3][::-1])
        image_pred = np.expand_dims(image, axis=0).astype(np.float32)
        preds = self.model.run(self.output_names, {self.input_names[0]: image_pred})[0]
        text = ctc_decoder(preds, self.metadata["vocab"])[0]
        return text

# Function to process the uploaded image
def process_image(image):
    # Convert the uploaded image to grayscale
    img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Run detection
    bbox = reader.detect(img_gray, width_ths=0.05, height_ths=0.05, ycenter_ths=0.1)

    # Create output directories for cropped images and image with bounding boxes
    output_cropped_dir = 'cropped_images1'
    output_bboxes_dir = 'bbox_image1'
    os.makedirs(output_cropped_dir, exist_ok=True)
    os.makedirs(output_bboxes_dir, exist_ok=True)

    # Path to save the CSV file
    csv_file_path = 'cropped_images_coordinates.csv'

    # Open CSV file in write mode, and initialize CSV writer
    with open(csv_file_path, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['crop_image_path', 'coordinates'])

        # List of bounding boxes
        boxes1 = bbox[1][0]
        boxes0 = bbox[0][0]

        # Load the image for processing
        image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image_with_boxes = img.copy()

        # Crop and save bounding box regions for 'boxes1'
        for i, box in enumerate(boxes1):
            points = np.array(box, dtype=np.int32)
            x_min = np.min(points[:, 0])
            y_min = np.min(points[:, 1])
            x_max = np.max(points[:, 0])
            y_max = np.max(points[:, 1])

            cropped_image = img[y_min:y_max, x_min:x_max]
            gray_cropped = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
            _, binary_cropped = cv2.threshold(gray_cropped, 128, 255, cv2.THRESH_BINARY)

            cropped_image_path = os.path.join(output_cropped_dir, f'cropped_box_{i+1}.png')
            cv2.imwrite(cropped_image_path, binary_cropped)

            coordinates = ','.join(map(str, points.flatten()))
            csv_writer.writerow([cropped_image_path, coordinates])

            cv2.polylines(image_with_boxes, [points], isClosed=True, color=(0, 255, 0), thickness=2)

        # Crop and save bounding box regions for 'boxes0'
        for i, box in enumerate(boxes0):
            x_min, x_max, y_min, y_max = box
            cropped_image = img[y_min:y_max, x_min:x_max]
            gray_cropped = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
            _, binary_cropped = cv2.threshold(gray_cropped, 128, 255, cv2.THRESH_BINARY)

            cropped_image_path = os.path.join(output_cropped_dir, f'cropped_box_{i + len(boxes1) + 1}.png')
            cv2.imwrite(cropped_image_path, binary_cropped)

            coordinates = f"{x_min},{y_min},{x_max},{y_min},{x_max},{y_max},{x_min},{y_max}"
            csv_writer.writerow([cropped_image_path, coordinates])

            cv2.rectangle(image_with_boxes, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)

        # Save the image with all bounding boxes drawn
        bbox_image_path = os.path.join(output_bboxes_dir, 'image_with_bboxes.jpg')
        cv2.imwrite(bbox_image_path, image_with_boxes)

    # Load the original CSV file
    df = pd.read_csv(csv_file_path)

    # Initialize the model
    model = ImageToWordModel(model_path="/kaggle/input/xin-chao1/202411260607/202411260607/model.onnx")

    # Add a new column 'prediction' if it doesn't exist
    if 'prediction' not in df.columns:
        df['prediction'] = ""

    # Loop through the DataFrame, predict for each image, and update the CSV
    for idx, row in df.iterrows():
        image_path = row['crop_image_path']
        image = cv2.imread(image_path.replace("\\", "/"))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

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

    return reconstructed_paragraph

# Create the Gradio interface with enhanced customization
# Create the Gradio interface
iface = gr.Interface(
    fn=process_image,
    inputs=gr.Image(type="numpy", label="Upload Image"),
    outputs="text",
    title="Image Detection and Recognition",
    description="Upload an image to detect and recognize text."
)

# Launch the interface
iface.launch()

import gradio as gr


DESCRIPTION = "Text to Video generation using Stable diffusion and multimodel"
            
def process_text2video():
    return print("1")

def create_demo_text_to_video():

    return demo

from __future__ import annotations

import gradio as gr
import sys
import os
# from app_text_to_video import create_demo_text_to_video
from css.theme import Seafoam
from css import theme

image_path="/kaggle/working/myT2V/css/bk.png"

seafoam=Seafoam()


with gr.Blocks(css="css/style.css", theme=gr.themes.Monochrome()) as demo:
        with gr.Blocks() as demo:
        title = gr.HTML(
            f"""<h1><span>{DESCRIPTION}</span></h1>""",
            elem_id="title",
        )
        with gr.Row():
            with gr.Column(variant="panel",scale=2):
                with gr.Tab("Txt2Vid"):
                    with gr.Group():
                        prompt = gr.Text(
                            label="Prompt",
                            max_lines=5,
                            placeholder="Enter your prompt",
                        )
                        negative_prompt = gr.Text(
                            label="Negative Prompt",
                            max_lines=5,
                            placeholder="Enter a negative prompt",
                        )
                with gr.Tab("Advanced Settings"):
                    with gr.Group():
                        video_length = gr.Slider(
                            label="Video length",  minimum=1, maximum=16, step=1, value=4)
                    with gr.Group():
                        seed = gr.Slider(
                            label="Seed", minimum=0, maximum=100000, step=1, value=40
                        )
                    with gr.Group():
                        fps = gr.Slider(
                            label="FPS",  minimum=0, maximum=8, step=1, value=1)
                    with gr.Group():
                        with gr.Row():
                            guidance_scale = gr.Slider(
                                label="Guidance scale",
                                minimum=1,
                                maximum=12,
                                step=0.1,
                                value=7.5,
                            )
                            num_inference_steps = gr.Slider(
                                label="Number of inference steps",
                                minimum=1,
                                maximum=100,
                                step=1,
                                value=50,
                            )
                    with gr.Group():
                        gr.Markdown("Perform DDPM steps from t0 to t1. The larger the gap between t0 and t1, the more variance between the frames. Ensure t0 < t1 ")
                        t0 = gr.Slider(label="Timestep t0", minimum=0, maximum=100, value=21, step=1,)
                        t1 = gr.Slider(label="Timestep t1", minimum=1, maximum=100, value=27, step=1,)

            with gr.Column(variant="panel",scale=3):
                with gr.Blocks():
                    run_button = gr.Button("Generate", variant="primary")
                result = gr.Video(label="Generated Video")

        inputs = [
            prompt,
            negative_prompt,
            seed,
            guidance_scale,
            num_inference_steps,
            video_length,
            fps,
            t0,
            t1
        ]

        run_button.click(fn=process_text2video,
                        inputs=inputs,
                        outputs=result,)


print("Generating Gradio app LINK:")
demo.launch()

