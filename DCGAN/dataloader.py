import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

def get_dataloader(batch_size):
    transform = transforms.Compose([
        transforms.Resize(64),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = torchvision.datasets.CelebA(root="./data", split="train", transform=transform, download=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    return dataloader
