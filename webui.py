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
from auto_increment_filename import AutoIncrementFilename

# idea make GIF to show the output

def create_gif(images : dict, filepath : str, duration : int = 1000):
    if len(images) < 1:
        return None 
    images = [Image.open(image) for image in images]
    if len(images) == 1:
        images[0].save(filepath)
    else:
        images[0].save(filepath, save_all=True, append_images=images[1:], optimize=False, duration=duration, loop=0)

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
    init_directories(config.output_dirs)
    engine = InterpolateEngine(config.model, config.gpu_ids)

    app = create_ui()
    app.launch(inbrowser = config.auto_launch_browser, 
                server_name = config.server_name,
                server_port = config.server_port)

def create_ui():
    with gr.Blocks() as app:
        gr.Markdown("VFIformer Web UI")
        with gr.Tab("Frame Interpolation"):
            with gr.Row():
                with gr.Column():
                    img1_input = gr.Image(type="filepath", label="Before Image") #.style(height=300, width=400)
                    img2_input = gr.Image(type="filepath", label="After Image") #.style(height=300, width=400)
                with gr.Column():
                    img_output = gr.Image(type="filepath", label="Animated Preview") #.style(height=300, width=400)
                    file_output = gr.File(type="file", label="Download Interpolated Image")
            interpolate_button = gr.Button("Interpolate", variant="primary")
        with gr.Tab("Slow Motion"):
            with gr.Row():
                image_input = gr.Image()
                image_output = gr.Image()
            image_button = gr.Button("Flip")

        interpolate_button.click(interpolate, inputs=[img1_input, img2_input], outputs=[img_output, file_output])
        # image_button.click(flip_image, inputs=image_input, outputs=image_output)
    return app

def init_directories(dirs : dict):
    for key in dirs.keys():
        dir = dirs[key]
        if not os.path.exists(dir):
            log.log(f"creating output directory {dir}")
            os.makedirs(dir)

if __name__ == '__main__':
    main()
