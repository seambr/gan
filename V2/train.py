import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import time
from model import Discriminator, Generator
from pathlib import Path
device = "cuda" if torch.cuda.is_available() else "cpu"


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


# Check if models are in directory
state_path = Path("./models/V1.pt")


if (state_path.exists()):
    print("Loading Models")
    state = torch.load(state_path)
    start = state["epoch"] + 1
    disc = Discriminator(IMAGE_DIM).to(device)
    disc.load_state_dict(state["disc"])

    gen = Generator(Z_DIM,IMAGE_DIM).to(device)
    gen.load_state_dict(state["gen"])
else:
    start = 0
    disc = Discriminator(IMAGE_DIM).to(device)
    gen = Generator(Z_DIM,IMAGE_DIM).to(device)

opt_disc = optim.Adam(disc.parameters(),lr=lr)
opt_gen = optim.Adam(gen.parameters(),lr=lr)
criterion = nn.BCELoss()

noise = torch.randn((BATCH_SIZE,Z_DIM)).to(device)

writer_baseline = SummaryWriter("./logs/V1/baseline")
writer_generated = SummaryWriter("./logs/V1/generated")


# INPUT IS BATCH_SIZE X CHANNELS X IMG_DIM x IMG_DIM

for epoch in range(start,start+NUM_EPOCHS):
    for batch_idx, (baseline,_) in enumerate(loader):
        
        baseline = baseline.view(-1,IMAGE_DIM).to(device)
        batch_size = baseline.shape[0]

        # Discriminator Training
        noise = torch.randn((BATCH_SIZE,Z_DIM)).to(device)
        generated = gen(noise) # BATCH_SIZE X IMG_DIM
        

        disc_baseline = disc(baseline).view(-1)
        loss_disc_baseline = criterion(disc_baseline, torch.ones_like(disc_baseline))

        disc_gen = disc(generated).view(-1)
        loss_disc_gen = criterion(disc_gen,torch.zeros_like(disc_gen))

        loss_disc = (loss_disc_baseline + loss_disc_gen) /2

        disc.zero_grad()
        loss_disc.backward(retain_graph=True)
        opt_disc.step()
        
        # Generator Training
        output = disc(generated).view(-1)
        loss_gen = criterion(output,torch.ones_like(output))
        gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()


        if batch_idx == 0:
            print(f"EPOCH {epoch} / {NUM_EPOCHS}")

            with torch.no_grad():
                generated = gen(noise).reshape(-1,CHANNELS_IMG,IMAGE_SIZE,IMAGE_SIZE)
                data = baseline.reshape(-1,CHANNELS_IMG,IMAGE_SIZE,IMAGE_SIZE)
                
                img_grid_generated = torchvision.utils.make_grid(generated,normalize=True)
                img_grid_baseline = torchvision.utils.make_grid(data,normalize=True)

                
                writer_baseline.add_image(f"baseline", img_grid_baseline,global_step=epoch)
                writer_generated.add_image(f"generated", img_grid_generated,global_step=epoch)
    if epoch % 100 == 0:
        # save model
        print("Saving Model")
        Path.mkdir(Path("./models"),exist_ok=True)
        torch.save({
            "gen":gen.state_dict(),
            "disc":disc.state_dict(),
            "epoch":epoch
            }, "./models/V1.pt")

            
