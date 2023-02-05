import os
import cv2
import argparse
from skimage.color import rgb2yuv, yuv2rgb
from utils.util import setup_logger, print_args
from utils.pytorch_msssim import ssim_matlab
from models.modules import define_G
from tqdm import tqdm
from interpolate_engine import InterpolateEngine
from interpolate import Interpolate
from simple_log import SimpleLog
from simple_utils import max_steps

def main():
    global log
    parser = argparse.ArgumentParser(description='infinite division of video frames')
    parser.add_argument('--model', default='./pretrained_models/pretrained_VFIformer/net_220.pth', type=str)
    parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--save_folder', default='./output', type=str)
    parser.add_argument('--base_path', default='./images', type=str, help="path to png files")

    # parser.add_argument('--base_name', default='image', type=str, help="filename before 0-filled index number")
    # parser.add_argument('--img_first', default=0, type=int, help="first image index")
    # parser.add_argument('--img_last', default=2, type=int, help="last image index")

    parser.add_argument('--img_before', default="./images/image0.png", type=str, help="Path to before frame image")
    parser.add_argument('--img_after', default="./images/image2.png", type=str, help="Path to after frame image")

    parser.add_argument('--num_width', default=5, type=int, help="index width for zero filling")
    parser.add_argument('--num_splits', default=1, type=int, help="how many doublings of the pool of frames")
    parser.add_argument("--verbose", dest="verbose", default=False, action="store_true", help="Show extra details")

    args = parser.parse_args()
    log = SimpleLog(args.verbose)
    engine = InterpolateEngine(args.model, args.gpu_ids)
    interpolater = Interpolate(engine.model, log.log)

    ## save paths
    save_path = args.save_folder
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # basepath = args.base_path
    # basefile = args.base_name
    # start = args.img_first
    # end = args.img_last
    # num_width = args.num_width
    # working_prefix = os.path.join(save_path, basefile)

    split_frames(interpolater, args.num_splits, args.img_before, args.img_after, args.save_folder, "frame", args.num_width, start=0, end=1)


    # pbar_desc = "Frames" if args.num_splits < 2 else "Total"
    # for n in tqdm(range(start, end), desc=pbar_desc, position=0):
    #     continued = n > start
    #     split_frames_in_sequence(interpolater, args.num_splits, basepath, basefile, n, n+1, num_width, working_prefix, save_path, continued)


# Add a long alpha sortable floating point number to a 
# filepath prefix representing the split position.
def indexed_filepath(filepath_prefix, index):
    return filepath_prefix + f"{index:1.24f}.png"

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

# output_filepath_prefix should be the first part of a path+filename
# that will have an position index and file extension added for saving working results
def split_frames(interpolater, num_splits, before_filepath, after_filepath, output_path, 
                base_filename, progress_label="Frame", continued=False, num_width=5, start=0, end=1):
    init_frame_register()
    reset_split_manager(num_splits)
    num_steps = max_steps(num_splits)
    init_progress(num_splits, num_steps, progress_label)

    output_filepath_prefix = os.path.join(output_path, base_filename)
    set_up_outer_frames(before_filepath, after_filepath, output_filepath_prefix)

    recursive_split_frames(interpolater, 0.0, 1.0, output_filepath_prefix)
    integerize_filenames(output_filepath_prefix, output_path, base_filename, start, end, continued, num_width)
    close_progress()


# def split_frames_in_sequence(interpolater, num_splits, basepath, basefile, start, end, num_width, working_prefix, save_path, continued):
#     init_register()
#     reset_split_count(num_splits)
#     num_steps = max_steps(num_splits)
#     init_progress(num_splits, num_steps, "Frame #" + str(start + 1))


#     recursive_split_frames(interpolater, first_index, last_index, working_prefix)
#     integerize_filenames(working_prefix, save_path, basefile, start, end, continued, num_width)
#     close_progress()

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

def integerize_filenames(output_filepath_prefix, save_path, base_name, start, end, continued, num_width):
    global log
    new_prefix = save_path + "//" + base_name + "[" + str(start).zfill(num_width) + "-" + str(end).zfill(num_width) + "]"
    frame_files = sorted_registered_frames()
    this_round_num_width = len(str(len(frame_files)))

    index = 0
    for file in frame_files:
        # orig_filename = working_filepath_prefix + str(f) + ".png"
        if continued and index == 0:
            # if a continuation from a previous set of frames, delete the first frame
            # to maintain continuity since it's duplicate of the previous round last frame
            os.remove(file)
            log.log("removed uneeded " + file)
        else:
            new_filename = new_prefix + str(index).zfill(this_round_num_width) + ".png"
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

# def source_filepath(basepath, basefile, index, num_width):
#     filename = basefile + str(index).zfill(num_width) + ".png"
#     return os.path.join(basepath, filename)



# todo
# with one split the secondary tqdm is not needed, with verbose both not needed
# encode the frame set being worked on in the temporary files for use in inspection
# could presume the basefile is pngsequence to match reqesuenced output
# automatic adding of the number of splits to the output folder
# option to automatically resequence the output
# add -X2 -X4 etc to output path to make it easy to keep separated


# ideas to autodetect settings

# assumtions:
# files share a common base alphanumeric name
# the filenames are the same length
# where the names differ is at end end and only numerals
# the files are sequential and all present 
# the starting index can be >= 0

# process:
# - get list of files from folder
# - sort the list
# - get the first and last entry
# - call a function to return the length of common characters between both filenames
# - use that to compute the base name


if __name__ == '__main__':
    main()


