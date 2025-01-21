import gradio as gr
from gradio_image_prompter import ImagePrompter
from image_processing import process_and_crop

def create_gradio_interface():
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

    with gr.Blocks(js=js_func, theme=theme) as demo:
        with gr.Column(variant="panel"):
            gr.Markdown("<center><h1 style='font-size: 40px;'>Vietnamese Handwritten Text Recognition Model</h1><center>")
            gr.Markdown("<center><span style='font-size:24px;'>Nguyen Xuan Thanh - 20203766</span><center>")

        
        with gr.Row():
            with gr.Column(variant="panel", scale=1):
                gr.Markdown("<h2 style='font-size: 24px;'>Image input</h2>")
                image_input = ImagePrompter(show_label=False)

                with gr.Row():
                    clear_button = gr.Button("Clear", variant="secondary")
                    submit_button = gr.Button("Submit", variant="primary")

            with gr.Column(variant="panel", scale=1):
                gr.Markdown("<h2 style='font-size: 24px;'>BBox Image</h2>")
                outputs = gr.Image()
                
            with gr.Column(variant="panel", scale=1):
                gr.Markdown("<h2 style='font-size: 24px;'>Text</h2>")
                result = gr.Textbox(label="Output Text", placeholder="The regconition text will appear here...", lines=5)
            
            # Set up the button click event for the Submit button
            submit_button.click(
                fn=process_and_crop,
                inputs=image_input,
                outputs=[result, outputs]
            )

            # Set up the button click event for the Clear button
            clear_button.click(
                fn=lambda: (None, None, ""),  # Clear outputs
                inputs=None,
                outputs=[image_input, outputs, result]
            )
    return demo