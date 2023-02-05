from PIL import Image

def create_gif(images : dict, filepath : str, duration : int = 1000):
    if len(images) < 1:
        return None 
    images = [Image.open(image) for image in images]
    if len(images) == 1:
        images[0].save(filepath)
    else:
        frame_duration = duration / len(images)
        images[0].save(filepath, save_all=True, append_images=images[1:], optimize=False, duration=frame_duration, loop=0)
