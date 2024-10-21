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
from glob import glob


date_time = time.strftime("%m_%d_%Y_%H_%M")
model_name = "best_metric_model_fullres_augmentation" + date_time
root_dir = "/project/ajoshi_1183/Projects/CRSeg/models"  #'/home/ajoshi/Projects/CRSeg/models' #

if not os.path.exists(root_dir):
    root_dir = "/home/ajoshi/Projects/CRSeg/models"  #'/project/ajoshi_1183/Projects/CRSeg/models' #


##
data_dir = (
    "/deneb_disk/BOBS_Repo/V1.0"  #'/home/ajoshi/project_ajoshi_27/headreco_out/' #
)


sub_lst = glob(data_dir + "/sub-*")

t1_images = list()
t2_images = list()
labels = list()
sub_sess = list()

for subdir in tqdm(sub_lst):

    sesslst = os.listdir(subdir)
    sub_name = os.path.basename(subdir)

    for sessname in sesslst:
        sessdir = os.path.join(subdir, sessname)
        t1_img = glob(os.path.join(sessdir, "anat") + "/*T1w.nii.gz")
        t2_img = glob(os.path.join(sessdir, "anat") + "/*T2w.nii.gz")
        label = glob(os.path.join(sessdir, "anat") + "/*dseg.nii.gz")

        print(sub_name, sessname, len(t1_img), len(t2_img), len(label))

        if len(t1_img) > 0 and len(t2_img) > 0 and len(label) > 0:
            t1_images.append(t1_img[0])
            t2_images.append(t2_img[0])
            labels.append(label[0])
            sub_sess.append(sub_name + "_" + sessname)

        else:
            print("Not all files found for ", sessdir)


# train_images = sorted(glob.glob(os.path.join(data_dir, "imagesTr", "*.nii.gz")))
# train_labels = sorted(glob.glob(os.path.join(data_dir, "labelsTr", "*.nii.gz")))

data_dicts = [
    {"t1_image": t1_name, "t2_image": t2_name, "label": label_name}
    for t1_name, t2_name, label_name in zip(t1_images, t2_images, labels)
]
train_files, val_files = data_dicts[:-9], data_dicts[-9:]


print("Name of the model: ", model_name)

print_config()

# %%
set_determinism(seed=0)

# %%
import monai
import numpy as np

print("MONAI version:", monai.__version__)

test_transform = Compose(
    [
        LoadImaged(keys=["t1_image", "t2_image", "label"]),
        EnsureChannelFirstd(keys=["t1_image", "t2_image", "label"]),
        EnsureTyped(keys=["t1_image", "t2_image", "label"]),
        # Orientationd(keys=["t1_image", "t2_image","label"], axcodes="RAS"),
        NormalizeIntensityd(
            keys=["t1_image", "t2_image"], nonzero=True, channel_wise=True
        ),
    ]
)

from monai.transforms import InvertibleTransform, NormalizeIntensity, Orientation

# check if applied transforms are instances of InvertibleTransform
t = NormalizeIntensityd(keys=["t1_image", "t2_image"], nonzero=True, channel_wise=True)
print(isinstance(t, InvertibleTransform))
t = Orientationd(keys=["t1_image", "t2_image","label"], axcodes="RAS")
print(isinstance(t, InvertibleTransform))


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
        with torch.amp.autocast("cuda"):
            return _compute(input)
    else:
        return _compute(input)


# enable cuDNN benchmark
torch.backends.cudnn.benchmark = True

# %% [markdown]
# ## Execute a typical PyTorch training process

# %%

test_org_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        ConvertToMultiChannelHeadRecod(keys="label"),
        EnsureChannelFirstd(keys=["image"]),
        Orientationd(keys=["image"], axcodes="RAS"),
        # Resized(keys=["image"], spatial_size=[96, 96, 96], mode=("trilinear")),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
    ]
)


post_transforms = Compose(
    [
        Invertd(
            keys="pred",
            transform=test_org_transforms,
            orig_keys="image",
            meta_keys="pred_meta_dict",
            orig_meta_keys="image_meta_dict",
            meta_key_postfix="meta_dict",
            nearest_interp=False,
            to_tensor=True,
            device="cpu",
        ),
        # Activationsd(keys="pred", sigmoid=True),
        # AsDiscreted(keys="pred", threshold=0.5),
    ]
)


val_ds = Dataset(data=val_files, transform=test_transform)

# %%
import nilearn as nl
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from nilearn import plotting
from nilearn.image import resample_to_img
from nilearn.plotting import plot_roi
from nilearn.plotting import plot_anat, plot_stat_map
from nilearn.plotting import plot_img
from nilearn.plotting import plot_stat_map


def plot_overlay(t1_img, t2_img, label_img, pred_label_img, slice_no):
    # use plot_roi fron nilearn to plot the overlay of the label and predicted label on the T1 and T2 images

    plot_anat(t1_img)
    plot_anat(t2_img)
    plot_anat(label_img)
    plot_anat(pred_label_img)


    plt.show()


model.load_state_dict(
    torch.load(
        os.path.join(
            root_dir,
            "best/best_metric_model_fullres_augmentation10_17_2024_10_14epoch_159.pth",
        )
    )
)
model.eval()

with torch.no_grad():

    for sub_data in data_dicts:
        val_data = test_transform(sub_data)
        val_data["pred"] = inference(val_data["t1_image"].unsqueeze(0).to(device))
        #val_output = post_trans(val_output[0])
        val_output = post_transforms(val_data)
        #val_output = post_transforms(
        #    {"pred": val_input["t1_image"], "image": val_input["t1_image"]}
        #)

        nifti_header_t1 = nib.load(sub_data["t1_image"]).header
        nifti_header_t2 = nib.load(sub_data["t2_image"]).header
        nifti_header_label = nib.load(sub_data["label"]).header
        nifti_affine_label = nib.load(sub_data["label"]).affine

        pred_label_img = nib.Nifti1Image(
            val_output["pred"][0,1].cpu(), nifti_affine_label, header=nifti_header_t1
        )
        # save the predicted label image
        pred_label_filename = os.path.join(
            os.path.dirname(sub_data["label"]), "pred_label.nii.gz"
        )
        nib.save(pred_label_img, pred_label_filename)

        plot_overlay(
            sub_data["t1_image"],
            sub_data["t2_image"],
            sub_data["label"],
            pred_label_filename,
            91,
        )

    # select one image to evaluate and visualize the model output
    val_input = val_ds[6]["t1_image"].unsqueeze(0).to(device)
    sw_batch_size = 4
    val_output = inference(val_input)
    print("shape of labels ", val_ds[6]["label"].shape)

    val_output = post_trans(val_output[0])
    plt.figure("t1_image", (24, 6))
    plt.subplot(3, 3, 1)
    plt.title(f"image channel t1")
    plt.imshow(val_ds[6]["t1_image"][0, :, :, 91].detach().cpu(), cmap="gray")

    # visualize the 3 channels label corresponding to this image
    for i in range(1):
        plt.subplot(3, 3, i + 4)
        plt.title(f"label channel {i}")
        plt.imshow(val_ds[6]["label"][i, :, :, 91].detach().cpu())
    # visualize the 3 channels model output corresponding to this image
    for i in range(3):
        plt.subplot(3, 3, i + 7)
        plt.title(f"output channel {i}")
        plt.imshow(val_output[i, :, :, 91].detach().cpu())

    plt.show()


# loop over all subjects. For each subject, load the T1, T2 and label files. Use the trained model to predict the labels for the T1 and T2 images. Save the predicted labels in the same folder as the T1 and T2 images.
# plot the overlay of the predicted labels on the T1 and T2 images. Save the overlay images in the same folder as the T1 and T2 images.

with torch.no_grad():

    for i in range(len(val_ds)):
        val_input = val_ds[i]["t1_image"].unsqueeze(0).to(device)
        val_output = inference(val_input)
        # create nibabel image from the predicted label
        pred_label_img = nib.Nifti1Image(val_output[0].cpu().numpy(), np.eye(4))
        # save the predicted label image
        nib.save(
            pred_label_img,
            os.path.join(os.path.dirname(val_ds[i]["t1_image"]), "pred_label.nii.gz"),
        )

        plot_overlay(
            val_ds[i]["t1_image"],
            val_ds[i]["t2_image"],
            val_ds[i]["label"],
            val_output[0],
            91,
        )

        val_input = val_ds[i]["t2_image"].unsqueeze(0).to(device)
        val_output = inference(val_input)
        plot_overlay(
            val_ds[i]["t1_image"],
            val_ds[i]["t2_image"],
            val_ds[i]["label"],
            val_output[0],
            91,
        )
        plt.show()
# %%
# %%
