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

import nibabel as nib

# enable cuDNN benchmark
torch.backends.cudnn.benchmark = True


root_dir = "/project/ajoshi_1183/Projects/CRSeg/models"  #'/home/ajoshi/Projects/CRSeg/models' #

if not os.path.exists(root_dir):
    root_dir = "/home/ajoshi/Projects/CRSeg/models"  #'/project/ajoshi_1183/Projects/CRSeg/models' #

best_model_file = os.path.join(
    root_dir, "best/best_metric_model_fullres_augmentation10_17_2024_10_14epoch_159.pth"
)

##
data_dir = (
    "/deneb_disk/BOBS_Repo/V1.0"  #'/home/ajoshi/project_ajoshi_27/headreco_out/' #
)

output_dir = "/deneb_disk/BOBS_Repo/outputs"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


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


data_dicts = [
    {"t1_image": t1_name, "t2_image": t2_name, "label": label_name}
    for t1_name, t2_name, label_name in zip(t1_images, t2_images, labels)
]

# sub_files, val_files = data_dicts[:-9], data_dicts[-9:]

print_config()
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
        Orientationd(keys=["t1_image", "t2_image", "label"], axcodes="RAS"),
        NormalizeIntensityd(
            keys=["t1_image", "t2_image"], nonzero=True, channel_wise=True
        ),
    ]
)

from monai.transforms import InvertibleTransform

# check if applied transforms are instances of InvertibleTransform
t = NormalizeIntensityd(keys=["t1_image", "t2_image"], nonzero=True, channel_wise=True)
t = EnsureTyped(keys=["t1_image", "t2_image", "label"])
print(isinstance(t, InvertibleTransform))
t = Orientationd(keys=["t1_image", "t2_image", "label"], axcodes="RAS")
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


post_transforms = Compose(
    [
        Invertd(
            keys="pred",
            transform=test_transform,
            orig_keys="t1_image",
            meta_keys="pred_meta_dict",
            orig_meta_keys="t1_image_meta_dict",
            meta_key_postfix="meta_dict",
            nearest_interp=False,
            to_tensor=True,
            device="cuda:0",
        ),
        Activationsd(keys="pred", sigmoid=True),
        # AsDiscreted(keys="pred", threshold=0.5),
    ]
)


val_ds = Dataset(data=data_dicts, transform=test_transform)
model.load_state_dict(torch.load(best_model_file, map_location=torch.device("cuda:0")))
model.eval()

t1t2 = "t2"

with torch.no_grad():

    for sub_data in data_dicts:
        val_data = test_transform(sub_data)
        val_data["pred"] = inference(val_data[t1t2 + "_image"].unsqueeze(0).to(device))[
            0
        ]
        val_output = post_transforms(val_data)

        sub_name = os.path.basename(sub_data[t1t2 + "_image"]).split("_space")[0]

        nifti_header_label = nib.load(sub_data["label"]).header
        nifti_affine_label = nib.load(sub_data["label"]).affine

        # get the predicted label image
        pvc_frac_data = val_output["pred"].cpu().numpy()

        mask = np.max(pvc_frac_data, axis=0) > 0.05
        pvc_frac_data = (
            3 * pvc_frac_data[0] + 2 * pvc_frac_data[1] + 1 * pvc_frac_data[2]
        ) * mask

        pred_label_img = nib.Nifti1Image(
            pvc_frac_data, nifti_affine_label, header=nifti_header_label
        )
        # save the predicted label image
        pred_label_filename = os.path.join(
            output_dir, sub_name + ".pred_from_" + t1t2 + ".pvc.frac.nii.gz"
        )
        nib.save(pred_label_img, pred_label_filename)
