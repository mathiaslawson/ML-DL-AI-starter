import torchvision
import torch
import torchvision.transforms as transforms


transform = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)


dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)


model = torchvision.models.resnet18(pretrained=True)

print (model)