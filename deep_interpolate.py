import os
import cv2
import argparse
from tqdm import tqdm
from interpolate_engine import InterpolateEngine
from interpolate import Interpolate
from simple_log import SimpleLog
from simple_utils import max_steps
from file_utils import create_directory

def main():
    global log
    parser = argparse.ArgumentParser(description="Video Frame Interpolation (deep)")
    parser.add_argument("--model", default="./pretrained_models/pretrained_VFIformer/net_220.pth", type=str)
    parser.add_argument("--gpu_ids", type=str, default="0", help="gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU")
    parser.add_argument("--img_before", default="./images/image0.png", type=str, help="Path to before frame image")
    parser.add_argument("--img_after", default="./images/image2.png", type=str, help="Path to after frame image")
    parser.add_argument("--depth", default=1, type=int, help="how many doublings of the frames")
    parser.add_argument("--output_path", default="./output", type=str, help="Output path for interpolated PNGs")
    parser.add_argument("--base_filename", default="interpolated_frame", type=str, help="Base filename for interpolated PNGs")
    parser.add_argument("--verbose", dest="verbose", default=False, action="store_true", help="Show extra details")

    args = parser.parse_args()
    log = SimpleLog(args.verbose)
    engine = InterpolateEngine(args.model, args.gpu_ids)
    interpolater = Interpolate(engine.model, log.log)
    create_directory(args.output_path)

    split_frames(interpolater, 
                args.depth, 
                args.img_before, 
                args.img_after, 
                args.output_path, 
                args.base_filename)

def split_frames(interpolater, 
                num_splits, 
                before_filepath, 
                after_filepath, 
                output_path, 
                base_filename, 
                progress_label="Frame", 
                continued=False):
    init_frame_register()
    reset_split_manager(num_splits)
    num_steps = max_steps(num_splits)
    init_progress(num_splits, num_steps, progress_label)
    output_filepath_prefix = os.path.join(output_path, base_filename)
    set_up_outer_frames(before_filepath, after_filepath, output_filepath_prefix)

    recursive_split_frames(interpolater, 0.0, 1.0, output_filepath_prefix)
    integerize_filenames(output_path, base_filename, continued)
    close_progress()

def recursive_split_frames(interpolater : Interpolate, first_index : float, last_index : float, filepath_prefix : str):
    if enter_split():
        mid_index = first_index + (last_index - first_index) / 2.0
        first_filepath = indexed_filepath(filepath_prefix, first_index)
        last_filepath = indexed_filepath(filepath_prefix, last_index)
        mid_filepath = indexed_filepath(filepath_prefix, mid_index)

        interpolater.create_between_frame(first_filepath, last_filepath, mid_filepath)
        register_frame(mid_filepath)
        step_progress()

        # deal with two new split regions
        recursive_split_frames(interpolater, first_index, mid_index, filepath_prefix)
        recursive_split_frames(interpolater, mid_index, last_index, filepath_prefix)
        exit_split()

def set_up_outer_frames(before_file, after_file, output_filepath_prefix):
    global log
    img0 = cv2.imread(before_file)
    img1 = cv2.imread(after_file)

    # create outer 0.0 and 1.0 versions of original frames
    before_index, after_index = 0.0, 1.0
    before_file = indexed_filepath(output_filepath_prefix, before_index)
    after_file = indexed_filepath(output_filepath_prefix, after_index)

    cv2.imwrite(before_file, img0)
    register_frame(before_file)
    log.log("copied " + before_file)

    cv2.imwrite(after_file, img1)
    register_frame(after_file)
    log.log("copied " + after_file)

def integerize_filenames(output_path, base_name, continued):
    global log
    file_prefix = os.path.join(output_path, base_name)
    frame_files = sorted_registered_frames()
    num_width = len(str(len(frame_files)))
    index = 0

    for file in frame_files:
        if continued and index == 0:
            # if a continuation from a previous set of frames, delete the first frame
            # to maintain continuity since it's duplicate of the previous round last frame
            os.remove(file)
            log.log("removed uneeded " + file)
        else:
            new_filename = file_prefix + str(index).zfill(num_width) + ".png"
            os.replace(file, new_filename)
            log.log("renamed " + file + " to " + new_filename)
        index += 1

global split_count
def reset_split_manager(num_splits):
    global split_count
    split_count = num_splits

def enter_split():
    global split_count
    if split_count < 1:
        return False
    split_count -= 1
    return True

def exit_split():
    global split_count
    split_count += 1

global frame_register
def init_frame_register():
    global frame_register
    frame_register = []

def register_frame(filename : str):
    global frame_register
    frame_register.append(filename)

def sorted_registered_frames():
    global frame_register
    return sorted(frame_register)

global split_progress
split_progress = None
def init_progress(num_splits, max, description):
    global split_progress
    if num_splits < 2:
        split_progress = None
    else:
        split_progress = tqdm(range(max), desc=description, position=-1)

def step_progress():
    global split_progress
    if split_progress:
        split_progress.update()
        split_progress.refresh()

def close_progress():
    global split_progress
    if split_progress:
        split_progress.close()

# filepath prefix representing the split position.
def indexed_filepath(filepath_prefix, index):
    return filepath_prefix + f"{index:1.24f}.png"


if __name__ == '__main__':
    main()


