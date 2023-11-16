from dataclasses import dataclass
from tools import * 
from unet import *
from DDPM import *
import matplotlib.pyplot as plt

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os

def ddp_setup(rank:int, world_size: int):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'INFO'
    init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


@dataclass
class BaseConfig:
    DATASET = "SketchDataset" # "MNIST", "Cifar-10","Cifar-100", "Flowers","SketchDataset"

    # Path to log inference images and save checkpoints
    root = "./logs_Checkpoints"
    os.makedirs(root, exist_ok=True)

    # Current log and checkpoint direcotry
    # by default start from "version_0", in_training, given a value to a new name folder
    log_folder = None # in inference:specific a folder name to load, by default will be the latest version
    checkpoint_name = "ddpm.tar"

@dataclass # auto generate __init__ function for class
class TrainingConfig:
    TIMESTEPS = 1000 # Define number of diffusion timesteps
    if BaseConfig.DATASET == "MNIST":
        IMG_SHAPE = (1, 32, 32)
    elif BaseConfig.DATASET == "SketchDataset":
        IMG_SHAPE = (1, 256, 256)
    else:
        IMG_SHAPE = (3, 32, 32)
    NUM_EPOCHS = 1000
    BATCH_SIZE = 4 #128,4
    LR = 2e-4
    NUM_WORKERS = 2  # 0 on cpu device

@dataclass
class ModelConfig: # setting up attention unet
    BASE_CH = 64 # 64,128,256,512
    BASE_CH_MULT = (1,2,4,8) # 32,16,8,4
    APPLY_ATTENTION = (False, False, True, False)
    DROPOUT_RATE = 0.1
    TIME_EMB_MULT = 128 # 128

def main(rank:int, world_size:int, trainingConfig:TrainingConfig, modelConfig:ModelConfig, baseConfig:BaseConfig):

    ddp_setup(rank, world_size) # initialize process group
    model = UNet(
        input_channels=trainingConfig.IMG_SHAPE[0],
        output_channels=trainingConfig.IMG_SHAPE[0],
        base_channels=modelConfig.BASE_CH,
        base_channels_multiples=modelConfig.BASE_CH_MULT,
        apply_attention=modelConfig.APPLY_ATTENTION,
        dropout_rate=modelConfig.DROPOUT_RATE,
        time_multiple=modelConfig.TIME_EMB_MULT,
    )
    #model = nn.DataParallel(model, device_ids=[0, 1], output_device=0)
    model.to(rank)
    model = DDP(model, device_ids=[rank],find_unused_parameters=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=TrainingConfig.LR)

    dataloader = get_dataloader(
        dataset_name=baseConfig.DATASET,
        batch_size=trainingConfig.BATCH_SIZE,
        pin_memory=True,
        shuffle=False,
        num_workers=trainingConfig.NUM_WORKERS
    )

    loss_fn = nn.MSELoss()
    sd = Diffusion_setting(num_diffusion_timesteps=trainingConfig.TIMESTEPS,
                       img_shape=trainingConfig.IMG_SHAPE,
                       device=rank)
    scaler = amp.GradScaler()

    log_dir, checkpoint_dir = setup_log_directory(config=baseConfig)

    generate_video = False

    train(model, sd, dataloader, optimizer, scaler, loss_fn, img_shape=trainingConfig.IMG_SHAPE,
        total_epochs=trainingConfig.NUM_EPOCHS, timesteps=trainingConfig.TIMESTEPS, log_dir=log_dir,
        checkpoint_dir=checkpoint_dir, generate_video=generate_video, device=rank,
        checkpoint_name=baseConfig.checkpoint_name)
    destroy_process_group()

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    trainingConfig = TrainingConfig()
    modelConfig = ModelConfig()
    baseConfig = BaseConfig()
    mp.spawn(main, args=(world_size, trainingConfig, modelConfig, baseConfig), nprocs=world_size)