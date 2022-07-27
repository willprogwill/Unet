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

from sklearn import manifold
from sklearn.manifold import TSNE

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

# Dimensinality reduction: t-SNE
FIXED_SEED = 3
# before training/inference:
np.random.seed( FIXED_SEED )

def tsne( X ):
    X_dataset = np.array( X )
    # print( " X_dataset = ", X_dataset );
    #tSNE = TSNE(n_components = 2, init = 'pca', perplexity = 30.0)
    #tSNE = TSNE(n_components = 2, init = 'pca', perplexity = 5.0)

    # changed for the new version by S.T. on 2021/04/07
    #tSNE = TSNE(n_components = 2, init = 'pca', perplexity = 5.0, random_state = 3 )
    # tSNE = TSNE(n_components = 2, init = 'pca', perplexity = 5.0, random_state = 3, square_distances=True )
    tSNE = TSNE(n_components = 2, init = 'pca', perplexity = 30.0, random_state = 3, square_distances=True )

    #tSNE = TSNE(n_components = 2, init = 'pca', perplexity = 4.0)
    Y_dataset = tSNE.fit_transform( X_dataset )
    Y_dataset = Y_dataset.astype(np.float64)
    print("t-SNE Dimensionality Reduction Done.")
    #return Y_dataset.data.numpy()
    return Y_dataset


# main program starts

#input size
inSize = 512

nEpochs = 2
args = sys.argv
if len( args ) == 2:
    nEpochs = int(args[ 1 ] )
    print( ' nEpochs = ', nEpochs )

# device config
device = torch.device( 'cuda' if torch.cuda.is_available() else 'cpu' )
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

#with open( 'output_and_label.pickle', mode='rb' ) as f:
#    output_and_label = pickle.load( f )
#with open( 'losses.pickle', mode='rb' ) as f:
#    losses = pickle.load( f )
#model.load_state_dict( torch.load( 'MNIST_model.pth' ) )

filename_output = "MNIST_output_UNet_" + str( nEpochs ).zfill( 4 ) + ".pth"
print( 'loading ', filename_output )
output_and_label = torch.load( filename_output, map_location=device )
#
filename_loss = "MNIST_loss_UNet_" + str( nEpochs ).zfill( 4 ) + ".pth"
print( 'loading ', filename_loss )
losses = torch.load( filename_loss, map_location=device )
#
filename_model = "MNIST_model_UNet_" + str( nEpochs ).zfill( 4 ) + ".pth"
print( 'loading ', filename_model )
model.load_state_dict( torch.load( filename_model, map_location=device ) )
#
#
model.eval()

# define the function to preview the image
def imshow( img ):
    img = torchvision.utils.make_grid( img )
    img = img / 2 + 0.5
    npimg = img.detach().numpy()
    plt.imshow( np.transpose( npimg, (1, 2, 0) ) )
    plt.show()


# show how the set of images get better at every two epochs
# output_and_label[ 0 : 10 : 2 ] <= From epoch 0 to 9 by sampling every 2 epochs
#step = int( nEpochs / 5 )
step = 1
for i, ( imgP, imgO ) in enumerate( output_and_label[ 0 : nEpochs : step ] ):
    print( ' set of original images at epoch: ', i*step )
    imshow( imgO.detach().reshape( -1, 1, inSize, inSize ).cpu() )
    print( ' set of predicted images at epoch: ', i*step )
    imshow( imgP.detach().reshape( -1, 1, inSize, inSize ).cpu() )

# show the set of predicted and original images at the last epoch
pred, orig = output_and_label[ -1 ]
#print( ' pred.size() = ', pred.size() )
#print( ' orig.size() = ', orig.size() )
print( ' original image ' )
imshow( orig.reshape( -1, 1, inSize, inSize ).cpu() )
print( ' predicted image ' )
imshow( pred.reshape( -1, 1, inSize, inSize ).cpu() )

# iterator = iter( testloader )
# img, _ = next( iterator )
# img = img.reshape( -1, 28 * 28 )
# output = model( img )
# imshow( img.reshape(-1, 1, 28, 28) )
# imshow( output.reshape(-1, 1, 28, 28) )

sys.exit()

sampleloader = DataLoader( testset, batch_size=1500 )
iterator = iter( sampleloader )
orig, pred = next( iterator )
#z = model.enc( orig.reshape(-1, 28 * 28) )
z = model.enc( orig )
z = z.detach().numpy()  # 後から簡単に使えるようにするための処理
print( z.shape )  # (1500, 2)
z.reshape( 1500, 32*7*7 )
y = tsne( z )
print( y.shape )  # (1500, 2)

nDigits = 10
set_list = [ set() for x in range( nDigits ) ]
for coord, lbl in zip( y.tolist(), pred ):
    set_list[ lbl ].add( tuple( coord ) )

for idx in range( nDigits ):
    print( f'items in set_list[{idx}]:' )
    for cnt, item in enumerate( set_list[ idx ] ):
        # print( ' cnt = ', cnt )
        # print( ' type( item ) = ', type( item ) )
        # print( ' len( item ) = ', len( item ) )
        print( item )
        if cnt > 5:
            break

colorlist = [ "r", "g", "b", "c", "k", "y", "orange", "lightgreen", "hotpink", "yellow" ]
plt.figure( figsize=(10, 10) )
for idx in range( nDigits ):
    for x, y in set_list[ idx ]:
        plt.scatter( x, y, c=colorlist[ idx ] )
description = [ f"{idx}: {colorlist[idx]}" for idx in range(10) ]
print( description )
plt.show()
