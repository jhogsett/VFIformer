import os
import argparse
import numpy as np
import gradio as gr
from interpolate_engine import InterpolateEngine
from interpolate import Interpolate
from simple_log import SimpleLog
from simple_config import SimpleConfig
from auto_increment_filename import AutoIncrementFilename
from image_utils import create_gif
from file_utils import create_directories
from simple_utils import max_steps

def main():
    global log, config, engine

    parser = argparse.ArgumentParser(description='VFIformer Web UI')
    parser.add_argument("--config_path", type=str, default="config.yaml", help="path to config YAML file")
    parser.add_argument("--verbose", dest="verbose", default=False, action="store_true", help="Show extra details")
    args = parser.parse_args()

    log = SimpleLog(args.verbose)
    config = SimpleConfig(args.config_path).config_obj()
    create_directories(config.directories)
    engine = InterpolateEngine(config.model, config.gpu_ids)

    app = create_ui()
    app.launch(inbrowser = config.auto_launch_browser, 
                server_name = config.server_name,
                server_port = config.server_port)

def interpolate(img_before_file : str, img_after_file : str, num_splits : float):
    global log, config, engine, file_output

    file_output.update(visible=False)
    if img_before_file and img_after_file:
        interpolater = Interpolate(engine.model, log.log)

        output_path = config.directories["output_interpolate"]
        output_basename = "interpolate"
        img_between_file = AutoIncrementFilename(output_path).next_filename(output_basename, "png")
        log.log("creating frame file " + img_between_file)
        interpolater.create_mid_frame(img_before_file, img_after_file, img_between_file)

        img_output_gif = AutoIncrementFilename(output_path).next_filename(output_basename, "gif")
        log.log("creating animated gif file " + img_between_file)
        duration = config.interpolate_settings["gif_duration"]
        create_gif([img_before_file, img_between_file, img_after_file], img_output_gif, duration=duration)

        download_visible = num_splits == 1
        download_file = img_between_file if download_visible else None
        return gr.Image.update(value=img_output_gif), gr.File.update(value=download_file, visible=download_visible)
    else:
        return None, None

def update_splits_info(num_splits : float):
    # before the splits, there's one time region between the before and after frames
    # after the splits, there are 2 ** num_splits time regions
    # subtracting the original time region yields the number of new regions = number of new frames
    return str(max_steps(num_splits))

def create_ui():
    global config, file_output
    with gr.Blocks(analytics_enabled=False, 
                    title="VFIformer Web UI", 
                    theme=config.user_interface["theme"],
                    css=config.user_interface["css_file"]) as app:
        gr.Markdown("VFIformer Web UI")
        with gr.Tab("Frame Interpolation"):
            with gr.Row(variant="compact"):
                with gr.Column(variant="panel"):
                    img1_input = gr.Image(type="filepath", label="Before Image", tool=None)
                    img2_input = gr.Image(type="filepath", label="After Image", tool=None)
                    with gr.Row(variant="panel"):
                        splits_input = gr.Slider(value=1, minimum=1, maximum=10, step=1, label="Splits")
                        info_output = gr.Textbox(value="1", label="New Frames", max_lines=1, interactive=False)
                with gr.Column(variant="panel"):
                    img_output = gr.Image(type="filepath", label="Animated Preview", interactive=False)
                    file_output = gr.File(type="file", label="Download", visible=False)
            interpolate_button = gr.Button("Interpolate", variant="primary")
        with gr.Tab("Slow Motion"):
            with gr.Row(variant="compact"):
                image_input = gr.Image()
                image_output = gr.Image()
            image_button = gr.Button("Flip")

        interpolate_button.click(interpolate, inputs=[img1_input, img2_input, splits_input], outputs=[img_output, file_output])
        splits_input.change(update_splits_info, inputs=splits_input, outputs=info_output, show_progress=False)
        # image_button.click(flip_image, inputs=image_input, outputs=image_output)
    return app

# todo
# upgrade to create any number of mid frames (infinite slow motion)
#   have a slider for number of splits, show number of new frames to be created
#   include them in the animation, target a single overall play time for the animation regardless of frames
#   need a numbering scheme
#     upgrade auto inc filename:
#       search by file extension to isolate into "runs"
#       maybe auto increment folders instead, put all the stuff into the new folder including copies of original images properly sequenced
#   target gif animation to be a specific time like two seconds with the animation duration adjusted to fit

# for videos
#   need to be able to accept a .mp4 file and dump into a series of PNG frames, perhaps ffmpeg for simplicity
#   need to be able to recreate a video from the PNG frames, maybe ffmpeg
#   ffmpy
#   important: slow process, upscaling or external post processing
#   - splitting frames is a very slow process
#     - would be nice to see the frames as they are being created
#     - need to be able to resume an aborted or failed process, maybe copy the files into the folder in a way that makes it easy to resume
#     - should be able to manually resume with settings
#   - user might want some external process to be run before putting back into a video, so make it easy thru settings 

# gif2mp4 idea
# - input a git, split into pngs, allow for external process on them, recombine into mp4

# timelapse-to-original video

# step2cont(inuous)?

# general mp4 upscaling
# - rip to pngs
# - upscale to whatever size
# - upscale to whatever frame rate
# - recombine into mp4 with quality settings



# how EASY is it to incorporate R-ESRGAN 4x+ directly?

if __name__ == '__main__':
    main()
