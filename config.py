import torch
from pathlib import Path


DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
LEARNING_RATE = 0.001
BATCH_SIZE = 64
NUM_WORKER = 4
NUM_EPOCH = 300

# train dataset directory
TRAIN_DIR = Path.cwd()/'datasets'/'train'

# test dataset directory
TEST_DIR = Path.cwd()/'datasets'/'test'

# model is saved in this path after training
MODEL_PATH = Path.cwd()/'models'/'model.pt'
