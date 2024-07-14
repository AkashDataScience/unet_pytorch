from torch.utils.data import Dataset
from PIL import Image

class PetDataset(Dataset):
    def __init__(self, images, masks, transform=None):
        super().__init__()
        self.images = images
        self.masks = masks
        self.transform = transform

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image_path = self.images[idx]
        mask_path = self.masks[idx]

        image = Image.open(image_path)
        mask = Image.open(mask_path)

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask) * 255.0 - 1.0

        return image, mask
        