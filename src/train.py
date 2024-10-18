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

model_name = "best_metric_model_augmentation" + date_time

##


print("Name of the model: ", model_name)

print_config()

# %%
set_determinism(seed=0)

# %%
import monai
import numpy as np

class RandomConvexCombination(monai.transforms.RandomizableTransform):
    """def __init__(self, keys=[], prob=0.5):
        super().__init__()
        self.keys = keys
        self.prob = prob
    """

    def randomize(self):
        super().randomize(None)
        self._alpha = self.R.uniform(low=0, high=1)

    def __call__(self, data):
        d = dict(data)
        """for key in self.keys:
            if self.prob > 0 and self.randomize() < self.prob:
                img1 = d[key]
                img2 = d[key]  # Assuming T1 and T2 have the same key
                alpha = self.randomize()
                d["image"] = alpha * img1 + (1 - alpha) * img2
        """
        self.randomize()
        if not self._do_transform:
            return d

        d["image"] = self._alpha * d["t1_image"] + (1.0-self._alpha) * d["t2_image"]
        
        return d

# %% [markdown]
# ## Define a new transform to convert brain tumor labels
# 
# Here we convert the multi-classes labels into multi-labels segmentation task in One-Hot format.

# %% [markdown]
# ## Setup transforms for training and validation

# %%
train_transform = Compose(
    [
        # load 4 Nifti images and stack them together
        LoadImaged(keys=["t1_image","t2_image",  "label"]),
        ConvertToMultiChannelHeadRecod(keys="label"),
        EnsureChannelFirstd(keys=["t1_image","t2_image"]),
        EnsureTyped(keys=["t1_image","t2_image", "label"]),
        Orientationd(keys=["t1_image","t2_image", "label"], axcodes="RAS"),
        #Spacingd(
        #    keys=["image", "label"],
        #    pixdim=(2.0, 2.0, 2.0),
        #    mode=("bilinear", "nearest"),
        #),
        Resized(keys=["t1_image","t2_image", "label"], spatial_size=[96, 96, 96], mode=("trilinear","trilinear", "nearest")),
       # RandSpatialCropd(keys=["image", "label"], roi_size=[224, 224, 144], random_size=False),
        RandFlipd(keys=["t1_image","t2_image", "label"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["t1_image","t2_image", "label"], prob=0.5, spatial_axis=1),
        RandFlipd(keys=["t1_image","t2_image", "label"], prob=0.5, spatial_axis=2),
        NormalizeIntensityd(keys=["t1_image","t2_image"], nonzero=True, channel_wise=True),
        RandScaleIntensityd(keys=["t1_image","t2_image"], factors=0.1, prob=1.0),
        RandShiftIntensityd(keys=["t1_image","t2_image"], offsets=0.1, prob=1.0),
        RandomConvexCombination()
    ]
)
val_transform = Compose(
    [
        LoadImaged(keys=["t1_image","t2_image", "label"]),
        ConvertToMultiChannelHeadRecod(keys="label"),
        EnsureChannelFirstd(keys=["t1_image","t2_image"]),
        EnsureTyped(keys=["t1_image","t2_image", "label"]),
        Orientationd(keys=["t1_image","t2_image", "label"], axcodes="RAS"),
        #Spacingd(
        #    keys=["image", "label"],
        #    pixdim=(2.0, 2.0, 2.0),
        #    mode=("bilinear", "nearest"),
        #),
        Resized(keys=["t1_image","t2_image", "label"], spatial_size=[96, 96, 96], mode=("trilinear", "trilinear","nearest")),
        NormalizeIntensityd(keys=["t1_image","t2_image"], nonzero=True, channel_wise=True),
        RandomConvexCombination()
    ]
)

# %%
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
with open(mode+'.txt', 'r') as myfile:
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

train_ds = CacheDataset(data=train_files, transform=train_transform, cache_rate=1.0, num_workers=4)
#train_ds = Dataset(data=train_files, transform=train_transform)

# use batch_size=2 to load images and use RandCropByPosNegLabeld
# to generate 2 x 4 images for network training
train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=4)

val_ds = CacheDataset(data=val_files, transform=val_transform, cache_rate=1.0, num_workers=4)
#val_ds = Dataset(data=val_files, transform=val_transform)
val_loader = DataLoader(val_ds, batch_size=2, num_workers=4)


# %% [markdown]
# ## Check data shape and visualize

# %%
for jj in range(5):
    val_data_example = train_ds[jj]
    print(f"image shape: {val_data_example['t1_image'].shape}")
    print(f"image shape: {val_data_example['t2_image'].shape}")

    plt.figure("t1 and t2 images", (12, 3))
    plt.subplot(1, 3, 1)
    plt.title(f"t1 image ")
    plt.imshow(val_data_example["t1_image"][0, :, :, 48].detach().cpu(), cmap="gray")
    plt.subplot(1, 3, 2)
    plt.title(f"t2 image")
    plt.imshow(val_data_example["t2_image"][0, :, :, 48].detach().cpu(), cmap="gray")
    plt.subplot(1, 3, 3)
    plt.title(f"mixed image ")
    plt.imshow(val_data_example["image"][0, :, :, 48].detach().cpu(), cmap="gray")
    plt.show()


    # also visualize the 3 channels label corresponding to this image
    print(f"label shape: {val_data_example['label'].shape}")
    plt.figure("label", (12, 3))
    for i in range(3):
        plt.subplot(1, 3, i + 1)
        plt.title(f"label channel {i}")
        plt.imshow(val_data_example["label"][i, :, :, 48].detach().cpu())
    plt.show()


# %% [markdown]
# ## Create Model, Loss, Optimizer

# %%
max_epochs = 300
val_interval = 5
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
optimizer = torch.optim.Adam(model.parameters(), 1e-4, weight_decay=1e-5)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)

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
best_metric = -1
best_metric_epoch = -1
best_metrics_epochs_and_time = [[], [], []]
epoch_loss_values = []
metric_values = []
metric_values_tc = []
metric_values_wt = []
metric_values_et = []

total_start = time.time()
for epoch in range(max_epochs):
    epoch_start = time.time()
    print("-" * 10)
    print(f"epoch {epoch + 1}/{max_epochs}")
    model.train()
    epoch_loss = 0
    step = 0
    for batch_data in train_loader:
        step_start = time.time()
        step += 1
        inputs, labels = (
            batch_data["image"].to(device),
            batch_data["label"].to(device),
        )

        optimizer.zero_grad()


        with torch.cuda.amp.autocast():
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        epoch_loss += loss.item()
        print(
            f"{step}/{len(train_ds) // train_loader.batch_size}"
            f", train_loss: {loss.item():.4f}"
            f", step time: {(time.time() - step_start):.4f}"
        )
    lr_scheduler.step()
    epoch_loss /= step
    epoch_loss_values.append(epoch_loss)
    print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

    if (epoch + 1) % val_interval == 0:
        model.eval()
        with torch.no_grad():
            for val_data in val_loader:
                val_inputs, val_labels = (
                    val_data["image"].to(device),
                    val_data["label"].to(device),
                )
                val_outputs = inference(val_inputs)
                val_outputs = [post_trans(i) for i in decollate_batch(val_outputs)]
                dice_metric(y_pred=val_outputs, y=val_labels)
                dice_metric_batch(y_pred=val_outputs, y=val_labels)

            metric = dice_metric.aggregate().item()
            metric_values.append(metric)
            metric_batch = dice_metric_batch.aggregate()
            metric_tc = metric_batch[0].item()
            metric_values_tc.append(metric_tc)
            metric_wt = metric_batch[1].item()
            metric_values_wt.append(metric_wt)
            metric_et = metric_batch[2].item()
            metric_values_et.append(metric_et)
            dice_metric.reset()
            dice_metric_batch.reset()

            if metric > best_metric:
                best_metric = metric
                best_metric_epoch = epoch + 1
                best_metrics_epochs_and_time[0].append(best_metric)
                best_metrics_epochs_and_time[1].append(best_metric_epoch)
                best_metrics_epochs_and_time[2].append(time.time() - total_start)
                torch.save(
                    model.state_dict(),
                    os.path.join(root_dir, model_name + f"epoch_{epoch}"+".pth"),
                )
                print("saved new best metric model")
            print(
                f"current epoch: {epoch + 1} current mean dice: {metric:.4f}"
                f" tc: {metric_tc:.4f} wt: {metric_wt:.4f} et: {metric_et:.4f}"
                f"\nbest mean dice: {best_metric:.4f}"
                f" at epoch: {best_metric_epoch}"
            )
    print(f"time consuming of epoch {epoch + 1} is: {(time.time() - epoch_start):.4f}")
total_time = time.time() - total_start
print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")

# %%


# %%
print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}, total time: {total_time}.")

# %% [markdown]
# ## Plot the loss and metric

# %%
plt.figure("train", (12, 6))
plt.subplot(1, 2, 1)
plt.title("Epoch Average Loss")
x = [i + 1 for i in range(len(epoch_loss_values))]
y = epoch_loss_values
plt.xlabel("epoch")
plt.plot(x, y, color="red")
plt.subplot(1, 2, 2)
plt.title("Val Mean Dice")
x = [val_interval * (i + 1) for i in range(len(metric_values))]
y = metric_values
plt.xlabel("epoch")
plt.plot(x, y, color="green")
plt.savefig(os.path.join(root_dir, model_name + "_loss_curve.png"))

plt.figure("train", (18, 6))
plt.subplot(1, 3, 1)
plt.title("Val Mean Dice TC")
x = [val_interval * (i + 1) for i in range(len(metric_values_tc))]
y = metric_values_tc
plt.xlabel("epoch")
plt.plot(x, y, color="blue")
plt.subplot(1, 3, 2)
plt.title("Val Mean Dice WT")
x = [val_interval * (i + 1) for i in range(len(metric_values_wt))]
y = metric_values_wt
plt.xlabel("epoch")
plt.plot(x, y, color="brown")
plt.subplot(1, 3, 3)
plt.title("Val Mean Dice ET")
x = [val_interval * (i + 1) for i in range(len(metric_values_et))]
y = metric_values_et
plt.xlabel("epoch")
plt.plot(x, y, color="purple")
plt.savefig(os.path.join(root_dir, model_name + "_metric_curve.png"))

plt.show()


# save the metric and loss values to disk
np.save(os.path.join(root_dir, model_name + "_loss_values.npy"), np.array(epoch_loss_values))
np.save(os.path.join(root_dir, model_name + "_metric_values.npy"), np.array(metric_values))
np.save(os.path.join(root_dir, model_name + "_metric_values_tc.npy"), np.array(metric_values_tc))
np.save(os.path.join(root_dir, model_name + "_metric_values_wt.npy"), np.array(metric_values_wt))
np.save(os.path.join(root_dir, model_name + "_metric_values_et.npy"), np.array(metric_values_et))


