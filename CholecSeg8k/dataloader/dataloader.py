import torch
import numpy as np
import cv2
import logging

from torch.utils.data import Dataset

class CholecSeg8k_loader(Dataset):
    def __init__(self, work_path, transform = None, split ='train', fold = 1, label_type = 'dense'):

        assert split in ['train', 'val']
        assert fold in [1, 2, 3, 4, 5]
        assert label_type in ['dense', 'scribble']

        self.transform = transform
        self.split = split
        self.label_type = label_type

        sample_list = work_path + '/dataset/target_list.txt'
        with open(sample_list, 'r') as f:
            all_image = f.readlines()
        all_image = [item.replace('\n', '') for item in all_image]

        val_path = work_path + '/dataset/val_samples_fold_{}.txt'.format(fold)
        with open(val_path, 'r') as f:
            all_val_image = f.readlines()

        test_list = [work_path + item.replace('\n', '') for item in all_val_image]

        train_list = [work_path + item for item in all_image if item not in test_list]

        if self.split == 'train':
            self.image_list = train_list


        else:
            self.image_list = test_list

        logging.info('{}, {}, found {} files'.format(fold, split, len(self.image_list)))


    def __len__(self):
        return len(self.image_list)


    def __getitem__(self, idx):
        img_file_path = self.image_list[idx]
        image = cv2.imread(img_file_path)

        image =  cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = image.astype(np.float32) / 255.0

        if self.label_type == 'dense':
            label_file_path = img_file_path.replace('endo', 'endo_watershed_mask_convert_13classes')
        else:
            label_file_path = img_file_path.replace('endo', 'endo_watershed_mask_convert_13classes_scribble')

        mask = cv2.imread(label_file_path, 0)

        # resize
        image = cv2.resize(image, (432, 240))
        mask = cv2.resize(mask, (432, 240), interpolation=cv2.INTER_NEAREST)

        data = {"image": image, "mask": mask}

        if self.transform is not None:
            augmented = self.transform(**data)
            image, mask = augmented["image"], augmented["mask"]

        image = image.transpose(2, 0, 1)
        image, mask = torch.from_numpy(image).float(), torch.from_numpy(mask).long()

        return {"image": image, "mask": mask, 'name': img_file_path}



