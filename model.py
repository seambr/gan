import torch
import torch.nn as nn


class Discriminator(nn.Module):

    def __init__(self, img_dim:int) -> None:
        super().__init__()
        self.disc = nn.Sequential(

            nn.Linear(img_dim,128),
            nn.LeakyReLU(0.1),
            nn.Linear(128,1),
            nn.Sigmoid() #Ensures output is ∈ [0,1]
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

