import os
import torch
import argparse
import torch.nn as nn

from glob import glob
from tqdm import tqdm
from dataset import PetDataset
from torchvision import transforms
from torch.utils.data import DataLoader
from model import UNet, MulticlassDiceLoss

IMAGE_PATH = 'data/images'
MASK_PATH = 'data/annotations/trimaps'

def get_args():
    parser = argparse.ArgumentParser(description='PyTorch UNet Training')
    parser.add_argument('--max_pool', action=argparse.BooleanOptionalAction)
    parser.add_argument('--transpose_conv', action=argparse.BooleanOptionalAction)
    parser.add_argument('--cross_entropy_loss', action=argparse.BooleanOptionalAction)
    args = parser.parse_args()
    return args

def _train(model, device, train_loader, optimizer, criterion):
    model.train()

    pbar = tqdm(train_loader)

    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        pred = model(data)
        
        loss = criterion(pred, target)

        loss.backward()
        optimizer.step()

        pbar.set_description(desc= f'Train: Loss={loss.item():0.4f} Batch_id={batch_idx}')

def start_training(num_epochs, model, device, train_loader, optimizer, criterion):
    for epoch in range(1, num_epochs+1):
        _train(model, device, train_loader, optimizer, criterion)

def main():
    args = get_args()

    os.makedirs('images', exist_ok=True)

    cuda = torch.cuda.is_available()
    print("CUDA Available?", cuda)

    image_list = glob(os.path.join(IMAGE_PATH, '*.jpg'))
    image_list = [x.replace('\\', '/') for x in image_list]
    mask_list = glob(os.path.join(MASK_PATH, '*.png'))
    mask_list = [x.replace('\\', '/') for x in mask_list]
    
    image_transform = transforms.Compose([transforms.Resize((240, 240)), 
                                    transforms.ToTensor()])
    
    mask_transform = transforms.Compose([transforms.PILToTensor(),
                                         transforms.Resize((240, 240)),
                                         transforms.Lambda(lambda x: (x - 1).squeeze().type(torch.LongTensor))])

    train_dataset = PetDataset(root='./data', split='trainval', image_transform=image_transform, 
                               mask_transform=mask_transform, download=True)

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)

    device = torch.device("cuda" if cuda else "cpu")
    model = UNet(32, args.max_pool, args.transpose_conv, 3)
    model =  model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    if args.cross_entropy_loss:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = MulticlassDiceLoss(num_classes=3, softmax_dim=1)

    start_training(25, model, device, train_dataloader, optimizer, criterion)

if __name__ == "__main__":
    main()