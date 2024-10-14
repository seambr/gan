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


device = "cuda" if torch.cuda.is_available() else "cpu"


lr=3e-4
IMAGE_SIZE = 64
CHANNELS_IMG = 3
BATCH_SIZE = 64
Z_DIM = 64
NUM_EPOCHS = 50


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


test_writer = SummaryWriter(f"logs/test")

# add some batches to tensorboard
for i in range(10):
    time.sleep(1)
    examples = iter(loader)

    images, labels = next(examples)

    test_writer.add_images(f'{i}_IMAGES', images)