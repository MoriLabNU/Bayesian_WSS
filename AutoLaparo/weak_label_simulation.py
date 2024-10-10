import os
import numpy as np
import cv2

from skimage.morphology import skeletonize

def scribble_simulation(label):

    c0 = label == 0
    c1 = label == 1
    c2 = label == 2
    c3 = label == 3
    c4 = label == 4
    c5 = label == 5
    c6 = label == 6
    c7 = label == 7
    c8 = label == 8
    c9 = label == 9

    skeleton_c0 = skeletonize(c0)
    skeleton_c1 = skeletonize(c1)
    skeleton_c2 = skeletonize(c2)
    skeleton_c3 = skeletonize(c3)
    skeleton_c4 = skeletonize(c4)
    skeleton_c5 = skeletonize(c5)
    skeleton_c6 = skeletonize(c6)
    skeleton_c7 = skeletonize(c7)
    skeleton_c8 = skeletonize(c8)
    skeleton_c9 = skeletonize(c9)

    # Initialize scribble with the ignored class (10)
    scribble = np.ones_like(label) * 10

    # Assign each scribble class
    scribble[skeleton_c0 == 1] = 0
    scribble[skeleton_c1 == 1] = 1
    scribble[skeleton_c2 == 1] = 2
    scribble[skeleton_c3 == 1] = 3
    scribble[skeleton_c4 == 1] = 4
    scribble[skeleton_c5 == 1] = 5
    scribble[skeleton_c6 == 1] = 6
    scribble[skeleton_c7 == 1] = 7
    scribble[skeleton_c8 == 1] = 8
    scribble[skeleton_c9 == 1] = 9

    return scribble


if __name__ == '__main__':

    current_file_path = __file__
    work_path = os.path.abspath(os.path.dirname(current_file_path))

    sample_list = work_path + '/dataset/target_list.txt'

    with open(sample_list, 'r') as f:
        image_list = f.readlines()

    image_list = [work_path + item.replace('\n', '') for item in image_list]


    colors = {
        'background': (0, 0, 0),
        'tool1m': (20, 20, 20),
        'tool1s': (40, 40, 40),
        'tool2m': (60, 60, 60),
        'tool2s': (80, 80, 80),
        'tool3m': (100, 100, 100),
        'tool3s': (120, 120, 120),
        'tool4m': (140, 140, 140),
        'tool4s': (160, 160, 160),
        'uterus': (180, 180, 180),
    }

    if not os.path.exists(work_path + '/dataset/AutoLaparo_Task3/labels'):
        os.makedirs(work_path + '/dataset/AutoLaparo_Task3/labels')
    if not os.path.exists(work_path + '/dataset/AutoLaparo_Task3/scribbles'):
        os.makedirs(work_path + '/dataset/AutoLaparo_Task3/scribbles')

    for img_file_path in image_list:

        print('processing {}'.format(img_file_path))

        mask_image = cv2.imread(img_file_path.replace('jpg', 'png').replace('imgs', 'masks'))

        class_map = np.zeros(mask_image.shape[:2], dtype=np.uint8)

        for class_number, color in enumerate(colors.values()):
            matches = np.all(mask_image == color, axis=-1)
            class_map[matches] = class_number

        print(np.unique(class_map))
        print(class_map.shape)

        cv2.imwrite(img_file_path.replace('jpg', 'png').replace('imgs', 'labels'), class_map)

        mask_scribble = scribble_simulation(class_map)

        print(np.unique(mask_scribble))
        print(mask_scribble.shape)

        cv2.imwrite(img_file_path.replace('jpg', 'png').replace('imgs', 'scribbles'), mask_scribble)


