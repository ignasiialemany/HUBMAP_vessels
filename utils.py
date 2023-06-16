from dataset import HUBMapDataset
import json
from torch.utils.data import DataLoader
import torch


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    print("Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    
    
def get_loaders(train_dir, validation_dir, batch_size, train_transform, validation_transform, num_workers, pin_memory=True):
    label_dir = "/kaggle/input/hubmap-hacking-the-human-vasculature/polygons.jsonl"
    id_to_annot = {}
    with open(label_dir, 'r') as json_file:
        for i,line in enumerate(json_file):
            data_line = json.loads(line)
            id_to_annot[data_line['id']]=data_line["annotations"]
            
            
    training_dataset = HUBMapDataset(train_dir,id_to_annot, transform = train_transform)
    validation_dataset = HUBMapDataset(validation_dir,id_to_annot,transform = validation_transform)
    
    training_loader = DataLoader(training_dataset,batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, shuffle=True)
    validation_loader = DataLoader(validation_dataset,batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)
    
    return training_loader, validation_loader


def save_predictions_as_imgs(loader, model, folder="saved_images/", device="cuda"):
    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        with torch.no_grad():
            preds = model(x)
            preds = torch.argmax(preds, dim=1). float()
        torchvision.utils.save_image(
            preds, f"{folder}/pred_{idx}.png"
        )
        torchvision.utils.save_image(y.unsqueeze(1), f"{folder}{idx}.png")

    model.train()
    


    
    
    
    
    
    