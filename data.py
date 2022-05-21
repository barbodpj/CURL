from torch.utils.data import Dataset, DataLoader
import torch
import cv2
from torchvision import transforms
import os
data_list =[]

class IE_Dataset(Dataset):
    def __init__(self, data_dir, target_size):
        self.data_dir = data_dir
        self.data_list = []
        for x in os.listdir(os.path.join(data_dir, "input")):
            self.data_list.append(x)
        self.target_size = target_size
        self.crop_transform = torch.nn.Sequential(
            transforms.RandomCrop(size=target_size),
            transforms.RandomHorizontalFlip(p=0.5)
        )

    def __len__(self):
        return len(self.data_list)

    @staticmethod
    def normalise_image(img):
        """Normalises image data to be a float between 0 and 1
        """
        img = img.astype('float32') / 255
        return img

    def __getitem__(self, idx):
        input_dir = os.path.join(self.data_dir , "input", self.data_list[idx])
        gt_dir = os.path.join(self.data_dir ,"gt", self.data_list[idx])
        input_img = cv2.cvtColor(cv2.imread(input_dir), cv2.COLOR_BGR2RGB)
        target_img = cv2.cvtColor(cv2.imread(gt_dir), cv2.COLOR_BGR2RGB)

        input_img = cv2.resize(input_img, (self.target_size[0], self.target_size[1]))
        target_img = cv2.resize(target_img, (input_img.shape[1], input_img.shape[0]))

        input_img = self.normalise_image(input_img)
        target_img = self.normalise_image(target_img)

        input_img = torch.tensor(input_img)
        target_img = torch.tensor(target_img)

        input_img = input_img.permute(2, 0, 1)
        target_img = target_img.permute(2, 0, 1)

        return input_img,target_img,self.data_list[idx]
