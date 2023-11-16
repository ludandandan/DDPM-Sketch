import os
import cv2
import base64
from PIL import Image

import torch
from torch.utils.data import DataLoader

import torchvision
import torchvision.transforms as transform
import torchvision.datasets as datasets
from IPython.display import display, HTML
from data_process.sketch_dataset import SketchDataset

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os
def setup_log_directory(config, inference=False):
    """
    Log and Model checkpoint directory Setup

    :param config: BaseConfig
    :param inference: True if this is used in the inference code
    :return:
    """

    if config.log_folder is not None:
        version_name = config.log_folder
    else:
        status = False
        # already has previous version
        if os.path.isdir(config.root):
            # Get all folders numbers in the root_log_dir
            folder_numbers =[]
            for file in os.listdir(config.root):
                try:
                    version_idx = int(file.replace("version_", ""))
                except:
                    pass
                else:
                    folder_numbers.append(version_idx)

            if len(folder_numbers) == 0:
                status = True
            else:
                # Find the latest version number present in the log_dir
                last_version_number = max(folder_numbers)
                # New version name
                version_name = f"version_{last_version_number + 1}" if not inference \
                    else f"version_{last_version_number}"
        else:
            status = True

        # no previous version
        if status:
            if inference:
                print('no trained model')
                return None
            else:
                os.makedirs(config.root, exist_ok=True)
                version_name = 'version_0'

    # Update the training config default directory
    log_dir = os.path.join(config.root, version_name, "Training_inference") if not inference \
        else os.path.join(config.root, version_name, "Inference_results")
    checkpoint_dir = os.path.join(config.root, version_name, "checkpoints")

    # Create new directory for saving new experiment version, if already there (in inference) auto skip
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    print(f"Logging at: {log_dir}")
    print(f"Model Checkpoint at: {checkpoint_dir}")

    return log_dir, checkpoint_dir






def get(element: torch.Tensor, idxs:torch.Tensor):
    """
    Get values from "element" by index positions (idxs) and
    reshape it to have the same dimension as a batch of images
    """
    ele = element.gather(-1, idxs) # size: B(same as idxs)
    return ele.reshape(-1,1,1,1) # size: B,1,1,1

def inverse_transform(tensors):
    """Convert tensor from [-1., 1.] to [0., 255]"""
    return ((tensors.clamp(-1,1) + 1.0)/2.0)*255.0

def make_a_grid_based_cv2_npy(tensor_img_batch: torch.Tensor, nrow:int):
    # the generated tensor_img_batch is B,C,H,W and C is RGB format (PIL), values in 0-1 range
    x_inv = inverse_transform(tensor_img_batch).type(torch.uint8) # move image to 0-255

    # place imgs into grid [C, H', W'], RGB:0-255
    grid = torchvision.utils.make_grid(x_inv, nrow=nrow, pad_value=255.0).to("cpu")

    # make the grid img to a frame format (cv2 image: HWC, RGB:0-255)
    # [C,H',W'],RGB:0-255 -> [H',W',C],RGB:0-255 -> npy -> [H',W',C_reverse],BGR:0-255,npy
    grid_cv2_npy = torch.permute(grid,(1,2,0)).numpy()[:, :, ::-1]
    return grid_cv2_npy

def cv2_to_pil(cv2_npy_img):
    # [H,W,C_reverse],BGR:0-255 -> [H,W,C],RGB:0-255 -> Image.fromarray -> [C,H,W],RGB:0-1
    pil_image = Image.fromarray(cv2_npy_img[:, :, ::-1])
    return pil_image

def make_a_grid_based_PIL_npy(tensor_img_batch:torch.Tensor, nrow:int):
    # the generated tensor_img_batch is B,C,H,W, and C is RGB format (PIL), values in 0-1 range
    tensor_img_batch = inverse_transform(tensor_img_batch).type(torch.uint8) # move image to 0-255
    #place imags into grid [C,H',W']
    grid = torchvision.utils.make_grid(tensor_img_batch, nrow=nrow, pad_value=255.0).to("cpu")
    # make the [C,H,W],RGB:0-255 tensor figure to a PIL npy image[C,H,W],RGB:0-1
    pil_image = transform.functional.to_pil_image(grid) # 看看是不是0-1，还是0-255
    return pil_image

def frames2vid_for_cv2frames(frames_list, save_path):
    """
    Convert a list of numpy image frames into a mp4 video
    :param frames_list
    :param save_path
    :return:
    """
    WIDTH = frames_list[0].shape[1]
    HEIGHT = frames_list[0].shape[0]
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(save_path, fourcc, 25, (WIDTH, HEIGHT))
    # Appending the images to the video one by one
    for image in frames_list:
        video.write(image)
    # Deallocating memories taken for window creation
    video.release()

#def get_default_device():
    #return torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
def my_transform(t):
    return (t*2)-1
def get_dataset(dataset_name='MNIST'):
    if dataset_name=="SketchDataset":
        shape = (256, 256)
    else:
        shape = (32, 32)
    transforms = transform.Compose(
        [
            transform.ToTensor(),
            transform.Resize(shape,
                              interpolation=transform.InterpolationMode.BICUBIC,
                              antialias=True),
            transform.Lambda(my_transform) #scale between [-1, 1]
        ]
    )
    if dataset_name.upper() == "MNIST":
        dataset = datasets.MNIST(root="data", train=True, download=True,transform=transforms)
    elif dataset_name == "Cifar-10":
        dataset = datasets.CIFAR10(root="data", train=True, download=True, transform=transforms)
    elif dataset_name == "Cifar-100":
        dataset = datasets.CIFAR100(root="data", train=True, download=True, transform=transforms)
    elif dataset_name == "Flowers":
        dataset = datasets.ImageFolder(root="/kaggle/input/flowers-recognition/flowers", transform=transforms)
    elif dataset_name == "SketchDataset":
        dataset = SketchDataset(dataset_dir="/home/ludan/code/ld/DDPM-sketch/data/CUFS/CUHKStudent/train/sketch", transform=transforms)

    return dataset

def to_device(data, device):
    """Move tensor to chosen device"""
    if isinstance(data, (list, tuple)):
        return [to_device(x,device) for x in data]
    return data.to(device, non_blocking=True)

class DeviceDataLoader:
    """Wrap a dataloader to move data a device """
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl:
            yield to_device(b, self.device)
    def __len__(self):
        """Number of batch"""
        return len(self.dl)



def get_dataloader(dataset_name='MNIST', batch_size=32, pin_memory=False, shuffle=True, num_workers=0):
    dataset = get_dataset(dataset_name=dataset_name)
    if not shuffle:
        dataloader = DataLoader(dataset, batch_size=batch_size, pin_memory=pin_memory, num_workers=num_workers,
                                shuffle=shuffle,sampler=DistributedSampler(dataset))
    else:
        dataloader = DataLoader(dataset, batch_size=batch_size, pin_memory=pin_memory, num_workers=num_workers,
                            shuffle=shuffle)
    return dataloader