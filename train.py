import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from model import UNET
from utils import get_loaders




#HYPERPARAMETERS
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
NUM_EPOCHS = 100
NUM_WORKERS = 2
IMAGE_HEIGHT = 512
IMAGE_WIDTH = 512
PIN_MEMORY = True
LOAD_MODEL = True
TRAIN_IMG_DIR = "train"
VAL_IMG_DIR  = "val"


def train_fn(loader, model, optimizer, loss_fn, scaler):
    
    
    loop = tqdm(loader)

    #train one epoch, the loader returns data, targets. 
    for batch_index, (data, targets) in enumerate(loop):
        
        #we send data to device
        data = data.to(device=DEVICE)
        
        #we add one dimension because masks have 1 channel
        targets = targets.float().unsqueeze(1).to(device=DEVICE)
        
        #forward, we compute predictions based on model, autocast basically improves training speed
        with torch.cuda.amp.autocast():
            predictions = model(data)
            #we compute loss
            loss = loss_fn(predictions, targets)
        
        #Reset optimizer values
        optimizer.zero_grad()
        
        #We go backward calculating gradients of the weights
        scaler.scale(loss).backward()
        
        #We optimize the weights through the optimizer 
        scaler.step(optimizer)
        
        #We update the weights
        scaler.update()
        
        #Shows loss value in bar
        loop.set_postfix(loss = loss.item())
    
    
def main():
    
    train_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    val_transforms = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )
    
    #we get the model 
    model = UNET(in_channels=3, out_channels=3).to(device=DEVICE)
    #we get the loss fn
    loss_fn = nn.CrossEntropyLoss()
    #we get the optimizer 
    optimizer = optim.Adam(model.parameters(), lr = LEARNING_RATE)
    
    #we get the loaders
    train_loader, val_loader = get_loaders(
        TRAIN_IMG_DIR,
        VAL_IMG_DIR,
        BATCH_SIZE,
        train_transform,
        val_transforms,
        NUM_WORKERS,
        PIN_MEMORY,
    )
    
    #we get the scaler
    scaler = torch.cuda.amp.GradScaler()
    
    #we train through all the EPOCHS
    for epoch in range(NUM_EPOCHS):
        
        train_fn(train_loader, model, optimizer, loss_fn, scaler)
        
        # save model
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer":optimizer.state_dict(),
        }
        
        save_checkpoint(checkpoint)

        # check accuracy
        check_accuracy(val_loader, model, device=DEVICE)

        # print some examples to a folder
        save_predictions_as_imgs(
            val_loader, model, folder="saved_images/", device=DEVICE
        )
    
    
    

