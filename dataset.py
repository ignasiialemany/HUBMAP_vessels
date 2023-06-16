import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np


class HUBMapDataset(Dataset):
    def __init__(self, image_dir, id_annot):
        self.id2label = {0:"background",1:"blood_vessel", 2:"non_blood_vessel"}    
        self.ids = [f[:-4] for f in os.listdir(image_dir) if f.endswith(".tif")]
        self.image_dir = image_dir
        self.id_annot = id_annot
        
    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        # Load image
        image_path = os.path.join(self.image_dir, self.ids[idx] + ".tif")
        image = Image.open(image_path)
        
        # Initialize mask
        mask = np.zeros((512, 512), dtype=np.float32)

        # Process annotations
        for annot in self.id_annot[self.ids[idx]]:
            cords = annot['coordinates']
            if annot['type'] == "blood_vessel":
                for cord in cords:
                    rr, cc = np.array([i[1] for i in cord]), np.asarray([i[0] for i in cord])
                    mask[rr, cc] = 1
            elif annot['type'] == "glomerulus":
                for cord in cords:
                    rr, cc = np.array([i[1] for i in cord]), np.asarray([i[0] for i in cord])
                    mask[rr, cc] = 2
            else:
                for cord in cords:
                    rr, cc = np.array([i[1] for i in cord]), np.asarray([i[0] for i in cord])
                    mask[rr, cc] = 2

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]
        
        return image, mask