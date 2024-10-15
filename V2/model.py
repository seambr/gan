import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
class Discriminator(nn.Module):

    def __init__(self, img_channels:int,intermediate_channels:int,img_dim=64) -> None:

        super().__init__()
        needed_units = int(np.log2(img_dim)) - 3

        self.disc = nn.Sequential(
            nn.Conv2d(img_channels,intermediate_channels,kernel_size=4,stride=2,padding=1),
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

    def __init__(self, noise_dim:int,img_channels:int,intermediate_channels:int,img_dim=64):
        
        
        super().__init__()
        needed_units = int(np.log2(img_dim)) - 3
        self.noise_dim = noise_dim
        self.gen = nn.Sequential(
            self._unit(noise_dim,intermediate_channels*16,kernel_size=4,stride=1,padding=0), # B X int. X 4 X 4
            self._unit(intermediate_channels*16,intermediate_channels*8,kernel_size=4,stride=2,padding=1),
            self._unit(intermediate_channels*8,intermediate_channels*4,kernel_size=4,stride=2,padding=1),
            self._unit(intermediate_channels*4,intermediate_channels*2,kernel_size=4,stride=2,padding=1),
            nn.ConvTranspose2d(intermediate_channels*2,img_channels,kernel_size=4,stride=2,padding=1),
            nn.Tanh()
            
        ) 
    def _unit(self,in_channels,out_channels,kernel_size,stride,padding):
    
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels,out_channels,kernel_size,stride,padding,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
    
    def get_noise(self,batch_size):

        return torch.randn((batch_size,self.noise_dim,1,1))

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

    def test_dis():

        example,label = next(iter(loader))
        print(F"INPUT : {example.shape}")


        dis = Discriminator(3,16)
        print(F"CURRENT : {dis(example).shape}")

    def test_gen():

        example,label = next(iter(loader))
        gen = Generator(100,3,16,64)
        noise = gen.get_noise(32)
        print(F"INPUT : {noise.shape}")


        print(F"CURRENT : {gen(noise).shape}")
    
    test_dis()
    test_gen()