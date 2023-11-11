from dataclasses import dataclass
from tools import * 
from unet import *
from DDPM import *
import matplotlib.pyplot as plt

@dataclass
class BaseConfig:
    DEVICE = get_default_device()
    DATASET = "SketchDataset" # "MNIST", "Cifar-10","Cifar-100", "Flowers"

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
        IMG_SHAPE = (3, 256, 256)
    else:
        IMG_SHAPE = (3, 32, 32)
    NUM_EPOCHS = 1000
    BATCH_SIZE = 2 #128
    LR = 2e-4
    NUM_WORKERS = 2 if str(BaseConfig.DEVICE) != "cpu" else 0 # 0 on cpu device

@dataclass
class ModelConfig: # setting up attention unet
    BASE_CH = 64 # 64,128,256,512
    BASE_CH_MULT = (1,2,4,8) # 32,16,8,4
    APPLY_ATTENTION = (False, False, True, False)
    DROPOUT_RATE = 0.1
    TIME_EMB_MULT = 2 # 128

model = UNet(
    input_channels=TrainingConfig.IMG_SHAPE[0],
    output_channels=TrainingConfig.IMG_SHAPE[0],
    base_channels=ModelConfig.BASE_CH,
    base_channels_multiples=ModelConfig.BASE_CH_MULT,
    apply_attention=ModelConfig.APPLY_ATTENTION,
    dropout_rate=ModelConfig.DROPOUT_RATE,
    time_multiple=ModelConfig.TIME_EMB_MULT,
)
model.to(BaseConfig.DEVICE)

optimizer = torch.optim.AdamW(model.parameters(), lr=TrainingConfig.LR)

dataloader = get_dataloader(
    dataset_name=BaseConfig.DATASET,
    batch_size=TrainingConfig.BATCH_SIZE,
    device=BaseConfig.DEVICE,
    pin_memory=True,
    num_workers=TrainingConfig.NUM_WORKERS,
)

loss_fn = nn.MSELoss()
sd = Diffusion_setting(num_diffusion_timesteps=TrainingConfig.TIMESTEPS,
                       img_shape=TrainingConfig.IMG_SHAPE,
                       device=BaseConfig.DEVICE)
scaler = amp.GradScaler()

log_dir, checkpoint_dir = setup_log_directory(config=BaseConfig)

generate_video = False

train(model, sd, dataloader, optimizer, scaler, loss_fn, img_shape=TrainingConfig.IMG_SHAPE,
      total_epochs=TrainingConfig.NUM_EPOCHS, timesteps=TrainingConfig.TIMESTEPS, log_dir=log_dir,
      checkpoint_dir=checkpoint_dir, generate_video=generate_video, device=BaseConfig.DEVICE,
      checkpoint_name=BaseConfig.checkpoint_name)
