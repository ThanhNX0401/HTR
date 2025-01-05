import gradio as gr
from image_processing import process_and_crop
from gradio_interface import create_gradio_interface

# Gradio app interface
demo = create_gradio_interface()

print("Generating Gradio app LINK:")
demo.launch(share=True)