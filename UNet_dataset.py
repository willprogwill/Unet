import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from PIL import Image
import glob

import os
from os import listdir

class PairImges(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.imgs_list = glob.glob(os.path.join(self.img_dir, "*"))
        self.transform = transform

    def __len__(self):
        return len(self.imgs_list)

    def __getitem__(self, idx):
        imgs = [filename for filename in listdir(self.imgs_list[idx]) if not filename.startswith('.')]

        ori_img = Image.open(os.path.join(self.imgs_list[idx], imgs[0]))
        ans_img = Image.open(os.path.join(self.imgs_list[idx], imgs[1]))

        if self.transform is not None:
            ori_img = self.transform(ori_img)
            ans_img = self.transform(ans_img)

        return ori_img, ans_img

#test main code
if __name__ == '__main__':
    import torchvision
    import numpy as np
    import matplotlib.pyplot as plt

    def imshow( img ):
        img = torchvision.utils.make_grid( img )
        img = img / 2 + 0.5
        npimg = img.detach().numpy()
        plt.imshow( np.transpose( npimg, (1, 2, 0) ) )
        plt.show()

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    dataset_dir = "./test_data"

    full_dataset = PairImges(dataset_dir, transform=transform)

    # Split data to 7:3
    train_size = int(0.7 * len(full_dataset))
    test_size = len(full_dataset) - train_size

    trainset, testset = torch.utils.data.random_split(full_dataset, [train_size, test_size])

    batch_size = 32
    #trainloader = DataLoader( full_dataset, batch_size=batch_size, shuffle=True )
    trainloader = DataLoader( trainset, batch_size=batch_size, shuffle=True )
    testloader = DataLoader( testset, batch_size=batch_size, shuffle=False )

    for count, (ori_img, ans_img) in enumerate(trainloader):
        print(f"{count}: ori_img: {ori_img.size()}, ans_img: {ans_img.size()}")

    #test view
    imshow(ori_img[0])
    imshow(ans_img[0])
