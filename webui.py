import os
import argparse
import numpy as np
import gradio as gr
from PIL import Image
from interpolate_engine import InterpolateEngine
from interpolate import Interpolate
from collections import namedtuple
from simple_log import SimpleLog
from simple_config import SimpleConfig


def interpolate(img_before_file : str, img_after_file : str):
    global log, config, engine
    if img_before_file and img_after_file:
        interpolater = Interpolate(engine.model, log.log)
        output_path = config.get("output_dirs")["output_interpolate"]
        img_between_file = os.path.join(output_path, "between_frame.png")
        interpolater.create_mid_frame(img_before_file, img_after_file, img_between_file)
        return Image.open(img_after_file)
    else:
        return None



def main():
    global log, config, engine

    parser = argparse.ArgumentParser(description='VFIformer Web UI')
    parser.add_argument("--config_path", type=str, default="config.yaml", help="path to config YAML file")
    parser.add_argument("--verbose", dest="verbose", default=False, action="store_true", help="Show extra details")
    args = parser.parse_args()

    log = SimpleLog(args.verbose)
    config = SimpleConfig(args.config_path)
    engine = InterpolateEngine(config.get("model"), config.get("gpu_ids"))

    with gr.Blocks() as app:
        gr.Markdown("VFIformer Web UI (draft)")
        with gr.Tab("Interpolate"):
            with gr.Row():
                with gr.Column():
                    img1_input = gr.Image(type="filepath", label="Pre Image")
                    img2_input = gr.Image(type="filepath", label="Post Image")
                with gr.Column():
                    img_output = gr.Image(type="pil", label="Interpolated")
                    save_as_button = gr.Button("Save As...")
            interpolate_button = gr.Button("Interpolate", variant="primary")
        with gr.Tab("Slow Motion"):
            with gr.Row():
                image_input = gr.Image()
                image_output = gr.Image()
            image_button = gr.Button("Flip")

        with gr.Accordion("Open for More!"):
            gr.Markdown("Look at me...")

        interpolate_button.click(interpolate, inputs=[img1_input, img2_input], outputs=img_output)
        # image_button.click(flip_image, inputs=image_input, outputs=image_output)

    app.launch()




if __name__ == '__main__':
    main()
