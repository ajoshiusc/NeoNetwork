# %% [markdown]
# ## Setup environment


# %% [markdown]
# ## Setup imports

# %%
import os
import shutil
import tempfile
import time
import matplotlib.pyplot as plt
from monai.apps import DecathlonDataset
from monai.config import print_config
from monai.data import DataLoader, decollate_batch, CacheDataset, Dataset
from monai.handlers.utils import from_engine
from monai.losses import DiceLoss
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.networks.nets import SegResNet
from myutils import ConvertToMultiChannelHeadRecod
from monai.transforms import (
    Activations,
    Activationsd,
    AsDiscrete,
    AsDiscreted,
    Compose,
    Invertd,
    LoadImaged,
    MapTransform,
    NormalizeIntensityd,
    Orientationd,
    RandFlipd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandSpatialCropd,
    Resized,
    Spacingd,
    EnsureTyped,
    EnsureChannelFirstd,
)
from monai.utils import set_determinism
import onnxruntime
from tqdm import tqdm

import torch


date_time = time.strftime("%m_%d_%Y_%H_%M")

##

model_name = "best_metric_model_fullres_augmentation" + date_time

##


print("Name of the model: ", model_name)

print_config()

# %%
set_determinism(seed=0)

# %%
import monai
import numpy as np


class RandomConvexCombination(monai.transforms.RandomizableTransform):
    def randomize(self):
        super().randomize(None)
        self._alpha = self.R.uniform(low=0, high=1)

    def __call__(self, data):
        d = dict(data)
        self.randomize()
        if not self._do_transform:
            return d

        d["image"] = self._alpha * d["t1_image"] + (1.0 - self._alpha) * d["t2_image"]

        return d


# %% [markdown]
# ## Define a new transform to convert brain tumor labels
#
# Here we convert the multi-classes labels into multi-labels segmentation task in One-Hot format.

# %% [markdown]
# ## Setup transforms for training and validation

# %%

val_transform = Compose(
    [
        LoadImaged(keys=["t1_image", "t2_image", "label"]),
        ConvertToMultiChannelHeadRecod(keys="label"),
        EnsureChannelFirstd(keys=["t1_image", "t2_image"]),
        EnsureTyped(keys=["t1_image", "t2_image", "label"]),
        Orientationd(keys=["t1_image", "t2_image", "label"], axcodes="RAS"),
        # Spacingd(
        #    keys=["image", "label"],
        #    pixdim=(2.0, 2.0, 2.0),
        #    mode=("bilinear", "nearest"),
        # ),
        #Resized(
        #    keys=["t1_image", "t2_image", "label"],
        #    spatial_size=[96, 96, 96],
        #    mode=("trilinear", "trilinear", "nearest"),
        #),
        NormalizeIntensityd(
            keys=["t1_image", "t2_image"], nonzero=True, channel_wise=True
        ),
        RandomConvexCombination(),
    ]
)

# %%
headreco_dir = "/project/ajoshi_27/headreco_out/"  #'/home/ajoshi/project_ajoshi_27/headreco_out/' #

if not os.path.exists(headreco_dir):
    headreco_dir = "/home/ajoshi/project_ajoshi_27/headreco_out/"  #'/project/ajoshi_27/headreco_out/' #


root_dir = "/project/ajoshi_1183/Projects/CRSeg/models"  #'/home/ajoshi/Projects/CRSeg/models' #

if not os.path.exists(root_dir):
    root_dir = "/home/ajoshi/Projects/CRSeg/models"  #'/project/ajoshi_1183/Projects/CRSeg/models' #


# check if root_dir has home in it, if yes you are running on local machine. So use smaller dataset

if "home" in root_dir:
    mode = "train_small"
else:
    mode = "train"

# Read the list of subjects
with open(mode + ".txt", "r") as myfile:
    sub_lst = myfile.read().splitlines()

train_t1_images = list()
train_t2_images = list()
train_labels = list()

for subname in tqdm(sub_lst):

    subdir = os.path.join(headreco_dir, 'm2m_'+subname)
    train_t1_images.append(os.path.join(subdir, 'T1fs_conform.nii.gz'))
    train_t2_images.append(os.path.join(subdir, 'T2_conform.nii.gz'))
    train_labels.append(os.path.join(subdir, subname + '_masks_contr.nii.gz'))


#train_images = sorted(glob.glob(os.path.join(data_dir, "imagesTr", "*.nii.gz")))
#train_labels = sorted(glob.glob(os.path.join(data_dir, "labelsTr", "*.nii.gz")))

data_dicts = [{"t1_image": t1_name, "t2_image": t2_name , "label": label_name} for t1_name, t2_name, label_name in zip(train_t1_images, train_t2_images, train_labels)]
train_files, val_files = data_dicts[:-9], data_dicts[-9:]



VAL_AMP = True

# standard PyTorch program style: create SegResNet, DiceLoss and Adam optimizer
device = torch.device("cuda:0")
model = SegResNet(
    blocks_down=[1, 2, 2, 4],
    blocks_up=[1, 1, 1],
    init_filters=16,
    in_channels=1,
    out_channels=3,
    dropout_prob=0.2,
).to(device)
loss_function = DiceLoss(smooth_nr=0, smooth_dr=1e-5, squared_pred=True, to_onehot_y=False, sigmoid=True)

dice_metric = DiceMetric(include_background=True, reduction="mean")
dice_metric_batch = DiceMetric(include_background=True, reduction="mean_batch")

post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])


# define inference method
def inference(input):
    def _compute(input):
        return sliding_window_inference(
            inputs=input,
            roi_size=(96, 96, 96),
            sw_batch_size=1,
            predictor=model,
            overlap=0.5,
        )

    if VAL_AMP:
        with torch.cuda.amp.autocast():
            return _compute(input)
    else:
        return _compute(input)


# use amp to accelerate training
scaler = torch.amp.GradScaler('cuda')
# enable cuDNN benchmark
torch.backends.cudnn.benchmark = True

# %% [markdown]
# ## Execute a typical PyTorch training process

# %%

# %%
val_transform2 = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        ConvertToMultiChannelHeadRecod(keys="label"),
        EnsureChannelFirstd(keys=["image"]),
        EnsureTyped(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        #Spacingd(
        #    keys=["image", "label"],
        #    pixdim=(2.0, 2.0, 2.0),
        #    mode=("bilinear", "nearest"),
        #),
        #Resized(keys=["image", "label"], spatial_size=[96, 96, 96], mode=("trilinear", "nearest")),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
    ]
)

val_org_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        ConvertToMultiChannelHeadRecod(keys="label"),
        #EnsureChannelFirstd(keys=["image"]),
        Orientationd(keys=["image"], axcodes="RAS"),
        #Resized(keys=["image"], spatial_size=[96, 96, 96], mode=("trilinear")),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
    ]
)

# val_ds = CacheDataset(data=val_files, transform=val_transform, cache_rate=1.0, num_workers=4)
val_org_ds = Dataset(data=val_files, transform=val_transform)
val_org_loader = DataLoader(val_org_ds, batch_size=1, shuffle=False, num_workers=4)

post_transforms = Compose(
    [
        Invertd(
            keys="pred",
            transform=val_org_transforms,
            orig_keys="image1",
            meta_keys="pred_meta_dict",
            orig_meta_keys="image_meta_dict",
            meta_key_postfix="meta_dict",
            nearest_interp=False,
            to_tensor=True,
            device="cpu",
        ),
        Activationsd(keys="pred", sigmoid=True),
        AsDiscreted(keys="pred", threshold=0.5),
    ]
)




val_ds = Dataset(data=val_files, transform=val_transform)

# %%

model.load_state_dict(torch.load(os.path.join(root_dir, "best/best_metric_model_fullres_augmentation10_17_2024_10_14epoch_159.pth")))
model.eval()
with torch.no_grad():
    # select one image to evaluate and visualize the model output
    val_input = val_ds[6]["t1_image"].unsqueeze(0).to(device)
    sw_batch_size = 4
    val_output = inference(val_input)
    print('shape of labels ',val_ds[6]["label"].shape)


    val_output = post_trans(val_output[0])
    plt.figure("t1_image", (24, 6))
    plt.subplot(3, 3, 1)
    plt.title(f"image channel t1")
    plt.imshow(val_ds[6]["t1_image"][0, :, :, 128].detach().cpu(), cmap="gray")
        
    # visualize the 3 channels label corresponding to this image
    for i in range(3):
        plt.subplot(3, 3, i + 4)
        plt.title(f"label channel {i}")
        plt.imshow(val_ds[6]["label"][i, :, :, 128].detach().cpu())
    # visualize the 3 channels model output corresponding to this image
    for i in range(3):
        plt.subplot(3, 3, i + 7)
        plt.title(f"output channel {i}")
        plt.imshow(val_output[i, :, :, 128].detach().cpu())
    
    plt.show()

# %% [markdown]
# ## Evaluation on original image spacings

