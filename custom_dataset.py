from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as T
import config


class CustomDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.transform = T.Compose([
            T.Resize([90]),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        self.data = ImageFolder(self.root_dir, transform=self.transform)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


# Checking the dataset
if __name__ == '__main__':
    train_dir = config.TRAIN_DIR
    train_data = CustomDataset(train_dir)
    train_loader = DataLoader(train_data, batch_size=config.BATCH_SIZE, num_workers=config.NUM_WORKER, shuffle=True)

    for image, label in train_loader:
        print(image.shape, label.shape)
        break
