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
    c10 = label == 10
    c11 = label == 11
    c12 = label == 12

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
    skeleton_c10 = skeletonize(c10)
    skeleton_c11 = skeletonize(c11)
    skeleton_c12 = skeletonize(c12)

    # Initialize scribble with the ignored class (13)
    scribble = np.ones_like(label) * 13

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
    scribble[skeleton_c10 == 1] = 10
    scribble[skeleton_c11 == 1] = 11
    scribble[skeleton_c12 == 1] = 12

    return scribble

if __name__ == '__main__':

    current_file_path = __file__
    work_path = os.path.abspath(os.path.dirname(current_file_path))

    sample_list = work_path + '/dataset/target_list.txt'

    with open(sample_list, 'r') as f:
        image_list = f.readlines()

    image_list = [work_path + item.replace('\n', '') for item in image_list]

    for img_file_path in image_list:
        print('processing {}'.format(img_file_path))

        mask = cv2.imread(img_file_path.replace('endo.png', 'endo_watershed_mask.png'), 0)

        assert (len(np.unique(mask)) < 13)

        # convert label to 0, 1, 2, 3, 4, 5, ...12
        # # C = {0: 50, 1: 11, 2: 21, 3: 13, 4: 12, 5: 31, 6: 23, 7: 24, 8: 25, 9: 32, 10: 22, 11: 33, 12: 5}
        mask_new = np.zeros_like(mask)

        mask_new[mask == 11] = 1
        mask_new[mask == 21] = 2
        mask_new[mask == 13] = 3
        mask_new[mask == 12] = 4
        mask_new[mask == 31] = 5
        mask_new[mask == 23] = 6
        mask_new[mask == 24] = 7
        mask_new[mask == 25] = 8
        mask_new[mask == 32] = 9
        mask_new[mask == 22] = 10
        mask_new[mask == 33] = 11
        mask_new[mask == 5] = 12

        cv2.imwrite(img_file_path.replace('endo.png', 'endo_watershed_mask_convert_13classes.png'), mask_new)

        mask_scribble = scribble_simulation(mask_new)

        cv2.imwrite(img_file_path.replace('endo.png', 'endo_watershed_mask_convert_13classes_scribble.png'), mask_scribble)