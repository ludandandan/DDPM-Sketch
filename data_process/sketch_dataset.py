import os
from data_process.image_folder import make_dataset, is_image_file
from torch.utils.data import Dataset
import torchvision.transforms as transform
from PIL import Image

class SketchDataset(Dataset):
    def __init__(self, dataset_dir,transform):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.transform = transform
        self.image_filenames = make_dataset(dataset_dir) # return a list of image path
    
    
    def __getitem__(self, index):
        image_path = self.image_filenames[index]
        image = self.transform(Image.open(image_path))
        return image
    
    def __len__(self):
        return len(self.image_filenames)
    
    def name(self):
        return 'SketchDataset'


