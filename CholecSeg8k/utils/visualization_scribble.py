import numpy as np
import torch


# Perform dilation on the skeleton images to enlarge the lines
def dilation_color_mapping(input):

    RGB_mapping = {
    0: [128, 128, 128],
    1: [0, 255, 0],
    2: [0, 0, 255],
    3: [255, 255, 0],
    4: [0, 255, 255],
    5:  [255, 0, 255],
    6: [192, 192, 192],
    7: [255, 0, 0],
    8: [128, 0, 0],
    9: [128, 128, 0],
    10: [0, 128, 0],
    11: [128, 0, 128],
    12: [0, 128, 128],
    13: [255, 255, 255],
    }

    scribble_rgb = np.zeros((*input.shape, 3), dtype=np.uint8)
    for k, v in RGB_mapping.items():
        scribble_rgb[input == k] = v

    return scribble_rgb




def class_to_rgb(predictions):

    colormap  = [
    [128, 128, 128],
    [0, 255, 0],
    [0, 0, 255],
    [255, 255, 0],
    [0, 255, 255],
    [255, 0, 255],
    [192, 192, 192],
    [255, 0, 0],
    [128, 0, 0],
    [128, 128, 0],
    [0, 128, 0],
    [128, 0, 128],
    [0, 128, 128]
]


    B, H, W = predictions.shape
    rgb_images = torch.zeros(B, 3, H, W, dtype=torch.uint8)

    for i in range(13):  # Assuming 13 classes
        for b in range(B):
            mask = predictions[b] == i
            rgb_images[b, 0, mask] = colormap[i][0]  # R channel
            rgb_images[b, 1, mask] = colormap[i][1]  # G channel
            rgb_images[b, 2, mask] = colormap[i][2]  # B channel

    return rgb_images
