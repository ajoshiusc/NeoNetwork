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
    ]
)

# %%
headreco_dir = '/home/ajoshi/project_ajoshi_27/headreco_out/'
root_dir = '/home/ajoshi/Projects/CRSeg/models'
mode = 'train'

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

#train_ds = CacheDataset(data=train_files, transform=train_transform, cache_rate=1.0, num_workers=4)
train_ds = Dataset(data=train_files, transform=train_transform)

# use batch_size=2 to load images and use RandCropByPosNegLabeld
# to generate 2 x 4 images for network training
train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=4)

#val_ds = CacheDataset(data=val_files, transform=val_transform, cache_rate=1.0, num_workers=4)
val_ds = Dataset(data=val_files, transform=val_transform)
val_loader = DataLoader(val_ds, batch_size=2, num_workers=4)


# %% [markdown]
# ## Check data shape and visualize

# %%
for jj in range(1):
    train_data_example = train_ds[jj]
    print(f"image shape: {train_data_example['t1_image'].shape}")
    print(f"image shape: {train_data_example['t2_image'].shape}")

    plt.figure("t1 and t2 images", (12, 3))
    plt.subplot(2, 3, 1)
    plt.title(f"t1 image ")
    plt.imshow(train_data_example["t1_image"][0, :, :, 48].detach().cpu(), cmap="gray")
    plt.subplot(2, 3, 2)
    plt.title(f"t2 image")
    plt.imshow(train_data_example["t2_image"][0, :, :, 48].detach().cpu(), cmap="gray")
    plt.subplot(2, 3, 3)
    plt.title(f"mixed image ")
    plt.imshow(train_data_example["image"][0, :, :, 48].detach().cpu(), cmap="gray")
    
    # also visualize the 3 channels label corresponding to this image
    for i in range(3):
        plt.subplot(2, 3, i + 4)
        plt.title(f"label channel {i}")
        plt.imshow(train_data_example["label"][i, :, :, 48].detach().cpu())
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
# %% [markdown]
# ## Check best pytorch model output with the input image and label

# %%
model.load_state_dict(torch.load(os.path.join(root_dir, "best/best_metric_model.pth")))
model.eval()
with torch.no_grad():
    # select one image to evaluate and visualize the model output
    val_input = val_ds[6]["t1_image"].unsqueeze(0).to(device)
    roi_size = (96, 96, 96)
    sw_batch_size = 4
    val_output = inference(val_input)
    print('shape of labels ',val_ds[6]["label"].shape)


    val_output = post_trans(val_output[0])
    plt.figure("t1_image", (24, 6))
    plt.subplot(3, 3, 1)
    plt.title(f"image channel {i}")
    plt.imshow(val_ds[6]["t1_image"][0, :, :, 48].detach().cpu(), cmap="gray")
        
    # visualize the 3 channels label corresponding to this image
    for i in range(3):
        plt.subplot(3, 3, i + 4)
        plt.title(f"label channel {i}")
        plt.imshow(val_ds[6]["label"][i, :, :, 48].detach().cpu())
    # visualize the 3 channels model output corresponding to this image
    for i in range(3):
        plt.subplot(3, 3, i + 7)
        plt.title(f"output channel {i}")
        plt.imshow(val_output[i, :, :, 48].detach().cpu())
    
    plt.show()

# %% [markdown]
# ## Evaluation on original image spacings

# %%
val_transform = Compose(
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
        Resized(keys=["image", "label"], spatial_size=[96, 96, 96], mode=("trilinear", "nearest")),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
    ]
)

val_org_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        ConvertToMultiChannelHeadRecod(keys="label"),
        #EnsureChannelFirstd(keys=["image"]),
        Orientationd(keys=["image"], axcodes="RAS"),
        Resized(keys=["image"], spatial_size=[96, 96, 96], mode=("trilinear")),
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

# %%
model.load_state_dict(torch.load(os.path.join(root_dir, "best/best_metric_model.pth")))
model.eval()
device = 'cuda:0'
with torch.no_grad():
    for val_data in val_org_loader:
        val_inputs = val_data["image"].to(device)
        val_data["pred"] = inference(val_inputs)
        print('shape of labels ',val_data["label"].shape)
        val_data = decollate_batch(val_data)
        print('shape of labels ',len(val_data))
        val_data = [post_transforms(i) for i in val_data]
        val_outputs, val_labels = from_engine(["pred", "label"])(val_data)
        dice_metric(y_pred=val_outputs, y=val_labels)
        dice_metric_batch(y_pred=val_outputs, y=val_labels)

    metric_org = dice_metric.aggregate().item()
    metric_batch_org = dice_metric_batch.aggregate()

    dice_metric.reset()
    dice_metric_batch.reset()

metric_tc, metric_wt, metric_et = metric_batch_org[0].item(), metric_batch_org[1].item(), metric_batch_org[2].item()

print("Metric on original image spacing: ", metric_org)
print(f"metric_tc: {metric_tc:.4f}")
print(f"metric_wt: {metric_wt:.4f}")
print(f"metric_et: {metric_et:.4f}")

# %% [markdown]
# ## Convert torch to onnx model

# %%
dummy_input = torch.randn(1, 4, 240, 240, 160).to(device)
onnx_path = os.path.join(root_dir, "best_metric_model.onnx")
torch.onnx.export(model, dummy_input, onnx_path, verbose=False)

# %% [markdown]
# ## Inference onnx model
# Here we change the model used by predictor to onnx_infer, both of which are used to obtain a tensor after the input has been reasoned by the neural network.
# 
# Note: If the warning `pthread_setaffinity_np failed` appears when executing this cell, this is a known problem with the onnxruntime and does not affect the execution result. If you want to disable the warning, you can cancel the following comment to solve the problem.

# %%
# Using the following program snippet will not affect the execution time.
# options = ort.SessionOptions()
# options.intra_op_num_threads = 1
# options.inter_op_num_threads = 1

# %%
def onnx_infer(inputs):
    ort_inputs = {ort_session.get_inputs()[0].name: inputs.cpu().numpy()}
    ort_outs = ort_session.run(None, ort_inputs)
    return torch.Tensor(ort_outs[0]).to(inputs.device)


def predict(input):
    def _compute(input):
        return sliding_window_inference(
            inputs=input,
            roi_size=(96, 96, 96),
            sw_batch_size=1,
            predictor=onnx_infer,
            overlap=0.5,
        )

    if VAL_AMP:
        with torch.cuda.amp.autocast():
            return _compute(input)
    else:
        return _compute(input)

# %%
onnx_model_path = os.path.join(root_dir, "best_metric_model.onnx")
ort_session = onnxruntime.InferenceSession(onnx_model_path)

for val_data in tqdm(val_loader, desc="Onnxruntime Inference Progress"):
    val_inputs, val_labels = (
        val_data["image"].to(device),
        val_data["label"].to(device),
    )

    ort_outs = predict(val_inputs)
    val_outputs = post_trans(torch.Tensor(ort_outs[0]).to(device)).unsqueeze(0)

    dice_metric(y_pred=val_outputs, y=val_labels)
    dice_metric_batch(y_pred=val_outputs, y=val_labels)
onnx_metric = dice_metric.aggregate().item()
onnx_metric_batch = dice_metric_batch.aggregate()
onnx_metric_tc = onnx_metric_batch[0].item()
onnx_metric_wt = onnx_metric_batch[1].item()
onnx_metric_et = onnx_metric_batch[2].item()

print(f"onnx metric: {onnx_metric}")
print(f"onnx_metric_tc: {onnx_metric_tc:.4f}")
print(f"onnx_metric_wt: {onnx_metric_wt:.4f}")
print(f"onnx_metric_et: {onnx_metric_et:.4f}")

# %% [markdown]
# ## Check best onnx model output with the input image and label

# %%
onnx_model_path = os.path.join(root_dir, "best_metric_model.onnx")
ort_session = onnxruntime.InferenceSession(onnx_model_path)
model.load_state_dict(torch.load(os.path.join(root_dir, "best_metric_model.pth")))
model.eval()

with torch.no_grad():
    # select one image to evaluate and visualize the model output
    val_input = val_ds[6]["image"].unsqueeze(0).to(device)
    val_output = inference(val_input)
    val_output = post_trans(val_output[0])
    ort_output = predict(val_input)
    ort_output = post_trans(torch.Tensor(ort_output[0]).to(device)).unsqueeze(0)
    plt.figure("image", (24, 6))
    for i in range(4):
        plt.subplot(1, 4, i + 1)
        plt.title(f"image channel {i}")
        plt.imshow(val_ds[6]["image"][i, :, :, 70].detach().cpu(), cmap="gray")
    plt.show()
    # visualize the 3 channels label corresponding to this image
    plt.figure("label", (18, 6))
    for i in range(3):
        plt.subplot(1, 3, i + 1)
        plt.title(f"label channel {i}")
        plt.imshow(val_ds[6]["label"][i, :, :, 70].detach().cpu())
    plt.show()
    # visualize the 3 channels model output corresponding to this image
    plt.figure("output", (18, 6))
    for i in range(3):
        plt.subplot(1, 3, i + 1)
        plt.title(f"pth output channel {i}")
        plt.imshow(val_output[i, :, :, 70].detach().cpu())
    plt.show()
    plt.figure("output", (18, 6))
    for i in range(3):
        plt.subplot(1, 3, i + 1)
        plt.title(f"onnx output channel {i}")
        plt.imshow(ort_output[0, i, :, :, 70].detach().cpu())
    plt.show()


