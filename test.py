import torch
from torch.utils.data import DataLoader
from custom_dataset import CustomDataset
from model import EmotionClassifier
import config


test_dataset = CustomDataset(config.TEST_DIR)
test_loader = DataLoader(
    test_dataset,
    batch_size=config.BATCH_SIZE,
    num_workers=config.NUM_WORKER,
    shuffle=False
)

model = EmotionClassifier()
model.load_state_dict(config.MODEL_PATH)
model.to(config.DEVICE)
model.eval()

with torch.inference_mode():
    for image, label in test_loader:
        image, label = image.to(config.DEVICE), label.to(config.DEVICE)

        output = model(image)
        prediction = torch.argmax(output)
