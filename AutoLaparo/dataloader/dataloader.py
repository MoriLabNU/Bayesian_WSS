import torch
import numpy as np
import cv2
from torch.utils.data import Dataset
import logging


class AutoLaparo_loader(Dataset):
    def __init__(self, work_path, transform=None, split='train', label_type='dense'):

        assert split in ['train', 'val', 'test']
        assert label_type in ['dense', 'scribble']

        self.transform = transform
        self.split = split
        self.label_type = label_type

        sample_list = work_path + '/dataset/target_list.txt'
        with open(sample_list, 'r') as f:
            all_image = f.readlines()
        all_image = [work_path + item.replace('\n', '') for item in all_image]

        # split train, val and test following the official split
        train_list = all_image[:1020]

        val_list = all_image[1020: 1362]

        test_list = all_image[1362:]

        if self.split == 'train':
            self.image_list = train_list

        elif self.split == 'val':
            self.image_list = val_list

        else:
            self.image_list = test_list

        logging.info('{}, found {} files'.format(split, len(self.image_list)))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_file_path = self.image_list[idx]
        image = cv2.imread(img_file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = image.astype(np.float32) / 255.0

        if self.label_type == 'dense':
            label_file_path = img_file_path.replace('jpg', 'png').replace('imgs', 'labels')
        else:
            label_file_path = img_file_path.replace('jpg', 'png').replace('imgs', 'scribbles')

        mask = cv2.imread(label_file_path, 0)

        image = cv2.resize(image, (480, 240))
        mask = cv2.resize(mask, (480, 240), interpolation=cv2.INTER_NEAREST)

        data = {"image": image, "mask": mask}

        if self.transform is not None:
            augmented = self.transform(**data)
            image, mask = augmented["image"], augmented["mask"]

        image = image.transpose(2, 0, 1)
        image, mask = torch.from_numpy(image).float(), torch.from_numpy(mask).long()

        return {"image": image, "mask": mask, 'name': img_file_path}
