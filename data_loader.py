import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

def get_celeba_dataloader(root="./data", batch_size=64, image_size=64, shuffle=True, num_workers=4, drop_last=True):
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = datasets.CelebA(root=root, split="train", transform=transform, download=True)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=drop_last)
    return loader
