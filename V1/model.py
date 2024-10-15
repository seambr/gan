import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
class Discriminator(nn.Module):

    def __init__(self, in_channels:int,intermediate_channels:int,img_dim=64) -> None:

        super().__init__()
        needed_units = int(np.log2(img_dim)) - 3

        self.disc = nn.Sequential(
            nn.Conv2d(in_channels,intermediate_channels,kernel_size=4,stride=2,padding=1),
            nn.LeakyReLU(0.2), # torch.Size([BATCH, intermediate, 0.5*IMG_SIZE, IMG_SIZE*0.5]),
            *[self._unit(intermediate_channels*(2**i),intermediate_channels*(2*(2**i)),kernel_size=4,stride=2,padding=1) for i in range(needed_units)],
            nn.Conv2d(intermediate_channels*(2**(needed_units)),1,kernel_size=4,stride=2,padding=0), # BATCH X 1 X 1 X 1
            nn.Sigmoid()

        )
    def _unit(self,in_channels,out_channels,kernel_size,stride,padding):
        # each _unit halfs the last two dimension sizes
        return nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size,stride,padding,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return self.disc(x)
        

class Generator(nn.Module):

    def __init__(self, noise_dim:int,img_dim:int):
        super().__init__()
        self.gen = nn.Sequential(

            nn.Linear(noise_dim,256),
            nn.LeakyReLU(0.1),
            nn.Linear(256,img_dim),
            nn.Tanh() #Ensures outputs are ∈ [-1,1]
        ) 

    def forward(self,x) -> torch.Tensor:
        """
        img_dim x img_dim x channel
        """
        return self.gen(x)


if __name__ =="__main__":
    lr=3e-4
    IMAGE_SIZE = 64
    CHANNELS_IMG = 3
    BATCH_SIZE = 32
    Z_DIM = 64
    NUM_EPOCHS = 500
    IMAGE_DIM = IMAGE_SIZE*IMAGE_SIZE*CHANNELS_IMG

    # transforming to make data more workable
    transforms = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE,IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.5 for _ in range(CHANNELS_IMG)], [0.5 for _ in range(CHANNELS_IMG)]
            ),
        ]
    )

    # Importing Data Set
    dataset = datasets.ImageFolder(root="./images", transform=transforms)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    example,label = next(iter(loader))
    print(F"INPUT : {example.shape}")


    dis = Discriminator(3,16)
    print(F"CURRENT : {dis(example).shape}")