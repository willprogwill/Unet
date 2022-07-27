# https://atmarkit.itmedia.co.jp/ait/articles/2007/10/news024_2.html
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision import transforms
from torchvision.datasets import MNIST, CIFAR10
import numpy as np
import matplotlib.pyplot as plt
import pickle
import torch.nn.functional as F
from tqdm import tqdm

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

        return ans_img, ori_img

class DoubleConv(nn.Module):
   """(convolution => [BN] => ReLU) * 2"""

   def __init__(self, in_channels, out_channels, mid_channels=None):
       super().__init__()
       if not mid_channels:
           mid_channels = out_channels
       self.double_conv = nn.Sequential(
           nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
           nn.BatchNorm2d(mid_channels),
           nn.ReLU(inplace=True),
           nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
           nn.BatchNorm2d(out_channels),
           nn.ReLU(inplace=True)
       )

   def forward(self, x):
       return self.double_conv(x)


class Down(nn.Module):
   """Downscaling with maxpool then double conv"""

   def __init__(self, in_channels, out_channels):
       super().__init__()
       self.maxpool_conv = nn.Sequential(
           nn.MaxPool2d(2),
           DoubleConv(in_channels, out_channels)
       )

   def forward(self, x):
       return self.maxpool_conv(x)


class Up(nn.Module):
   """Upscaling then double conv"""

   def __init__(self, in_channels, out_channels, bilinear=True):
       super().__init__()

       # if bilinear, use the normal convolutions to reduce the number of channels
       if bilinear:
           self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
           self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
       else:
           self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
           self.conv = DoubleConv(in_channels, out_channels)

   def forward(self, x1, x2):
       x1 = self.up(x1)
       # input is CHW
       diffY = x2.size()[2] - x1.size()[2]
       diffX = x2.size()[3] - x1.size()[3]

       x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                       diffY // 2, diffY - diffY // 2])
       # if you have padding issues, see
       # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
       # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
       x = torch.cat([x2, x1], dim=1)
       return self.conv(x)


class OutConv(nn.Module):
   def __init__(self, in_channels, out_channels):
       super(OutConv, self).__init__()
       self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

   def forward(self, x):
       return self.conv(x)


class UNet(nn.Module):
   def __init__(self, n_channels, n_classes, bilinear=False):
       super(UNet, self).__init__()
       self.n_channels = n_channels
       self.n_classes = n_classes
       self.bilinear = bilinear

       self.inc = DoubleConv(n_channels, 64)
       self.down1 = Down(64, 128)
       self.down2 = Down(128, 256)
       self.down3 = Down(256, 512)
       factor = 2 if bilinear else 1
       self.down4 = Down(512, 1024 // factor)
       self.up1 = Up(1024, 512 // factor, bilinear)
       self.up2 = Up(512, 256 // factor, bilinear)
       self.up3 = Up(256, 128 // factor, bilinear)
       self.up4 = Up(128, 64, bilinear)
       self.outc = OutConv(64, n_classes)

   def forward(self, x):
       x1 = self.inc(x)
       x2 = self.down1(x1)
       x3 = self.down2(x2)
       x4 = self.down3(x3)
       x5 = self.down4(x4)
       x = self.up1(x5, x4)
       x = self.up2(x, x3)
       x = self.up3(x, x2)
       x = self.up4(x, x1)
       logits = self.outc(x)
       return logits

# main program starts
#input size
inSize = 512

nEpochs = 1
args = sys.argv
if len( args ) == 2:
    nEpochs = int(args[ 1 ])
    print( ' nEpochs = ', nEpochs )

# device config
device = torch.device( 'cuda:0' if torch.cuda.is_available() else 'cpu' )
print( ' device = ', device )

# preparing MNIST as the training data
transform = transforms.Compose( [transforms.ToTensor(),
                                 transforms.Normalize( (0.5,), (0.5,) ) ] )
dataset_dir = "./half"

full_dataset = PairImges(dataset_dir, transform=transform)

# Split data to 7:3
train_size = int(0.7 * len(full_dataset))
test_size = len(full_dataset) - train_size

trainset, testset = torch.utils.data.random_split(full_dataset, [train_size, test_size])

batch_size = 32
trainloader = DataLoader( full_dataset, batch_size=batch_size, shuffle=True )
#trainloader = DataLoader( trainset, batch_size=batch_size, shuffle=True )
testloader = DataLoader( testset, batch_size=batch_size, shuffle=False )

# Prepare the NN model
input_size = inSize * inSize
model = UNet(n_channels=1, n_classes=1)
model.to( device )

# Prepare the loss and optimizer
learning_rate = 0.01
criterion = nn.MSELoss()
optimizer = torch.optim.SGD( model.parameters(), lr = learning_rate )

print( 'Number of iteration for each epoch = ', len( trainloader ) )

# Traning loop
losses = []
output_and_label = []

nBatches = len( trainloader )
print( 'in train' )
counter = 0
for epoch in range( nEpochs ):
    print(f'=== epoch: {epoch} ===')
    total_loss = 0.0

    #for counter, (imgI, imgO) in enumerate( trainloader ):
    for imgI, imgO in tqdm(trainloader):

        imgI = imgI.to( device )
        imgO = imgO.to( device )
        print(imgI.size)
        predicted = model( imgI )

        loss = criterion( predicted, imgO )

        # backward pass: gradients
        loss.backward()
        # update weights
        optimizer.step()
        # zero gradients
        optimizer.zero_grad()
        # update total loss and average
        total_loss += loss.item()

        counter+=1
    avg_loss = total_loss / counter
    # record the average loss
    losses.append( avg_loss )
    print( 'loss:', avg_loss )
    # store the last set of predicted and original images
    output_and_label.append( ( predicted.cpu(), img.cpu() ) )
print( 'finished' )


# save the results
# with open( 'output_and_label.pickle', mode='wb' ) as f:
#     pickle.dump( output_and_label, f )
#
# with open( 'losses.pickle', mode = 'wb' ) as f:
#     pickle.dump( losses, f )
#
filename_output = "MNIST_output_UNet_" + str( nEpochs ).zfill( 4 ) + ".pth"
print( 'saving ', filename_output )
torch.save( output_and_label, filename_output )
#
filename_loss = "MNIST_loss_UNet_" + str( nEpochs ).zfill( 4 ) + ".pth"
print( 'saving ', filename_loss )
torch.save( losses, filename_loss )
#
filename_model = "MNIST_model_UNet_" + str( nEpochs ).zfill( 4 ) + ".pth"
print( 'saving ', filename_model )
torch.save( model.state_dict(), filename_model )
