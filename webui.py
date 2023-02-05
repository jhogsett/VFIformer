import os
import argparse
import numpy as np
import gradio as gr
from interpolate_engine import InterpolateEngine
from interpolate import Interpolate
from collections import namedtuple
from simple_log import SimpleLog
from simple_config import SimpleConfig
from auto_increment_filename import AutoIncrementFilename
from image_utils import create_gif
from file_utils import create_directories

def interpolate(img_before_file : str, img_after_file : str):
    global log, config, engine
    if img_before_file and img_after_file:
        interpolater = Interpolate(engine.model, log.log)

        output_path = config.output_dirs["output_interpolate"]
        output_basename = "interpolate"
        img_between_file = AutoIncrementFilename(output_path).next_filename(output_basename, "png")
        log.log("creating frame file " + img_between_file)
        interpolater.create_mid_frame(img_before_file, img_after_file, img_between_file)

        img_output_gif = AutoIncrementFilename(output_path).next_filename(output_basename, "gif")
        log.log("creating animated gif file " + img_between_file)
        create_gif([img_before_file, img_between_file, img_after_file], img_output_gif)

        return img_output_gif, img_between_file
    else:
        return None

def main():
    global log, config, engine

    parser = argparse.ArgumentParser(description='VFIformer Web UI')
    parser.add_argument("--config_path", type=str, default="config.yaml", help="path to config YAML file")
    parser.add_argument("--verbose", dest="verbose", default=False, action="store_true", help="Show extra details")
    args = parser.parse_args()

    log = SimpleLog(args.verbose)
    config = SimpleConfig(args.config_path).config_obj()
    create_directories(config.output_dirs)
    engine = InterpolateEngine(config.model, config.gpu_ids)

    app = create_ui()
    app.launch(inbrowser = config.auto_launch_browser, 
                server_name = config.server_name,
                server_port = config.server_port)

def create_ui():
    global config
    with gr.Blocks(analytics_enabled=False, 
                    title="VFIformer Web UI", 
                    theme=config.user_interface["theme"],
                    css=config.user_interface["css_file"]) as app:
        gr.Markdown("VFIformer Web UI")
        with gr.Tab("Frame Interpolation"):
            with gr.Row(variant="compact"):
                with gr.Column(variant="panel"):
                    img1_input = gr.Image(type="filepath", label="Before Image") #.style(height=300, width=400)
                    img2_input = gr.Image(type="filepath", label="After Image") #.style(height=300, width=400)
                    interpolate_button = gr.Button("Interpolate", variant="primary")
                with gr.Column(variant="panel"):
                    img_output = gr.Image(type="filepath", label="Animated Preview") #.style(height=300, width=400)
                    file_output = gr.File(type="file", label="Download Interpolated Image")
        with gr.Tab("Slow Motion"):
            with gr.Row(variant="compact"):
                image_input = gr.Image()
                image_output = gr.Image()
            image_button = gr.Button("Flip")

        interpolate_button.click(interpolate, inputs=[img1_input, img2_input], outputs=[img_output, file_output])
        # image_button.click(flip_image, inputs=image_input, outputs=image_output)
    return app

if __name__ == '__main__':
    main()
