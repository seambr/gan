import torch
import torch.nn as nn




class Discriminator(nn.Module):

    def __init__(self, in_features) -> None:
        super().__init__()
        self.disc = nn.Sequential(

            nn.Linear(in_features,128),
            nn.LeakyReLU(0.1),
            nn.Linear(128,1),
            nn.Sigmoid() #Ensures output is in [0,1]
        )

        def forward(self):

