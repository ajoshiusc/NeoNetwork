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
from myutils import ConvertToGrayWhiteCSF
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

from mymodels import SegResNetLatentOut

print_config()

# %%
set_determinism(seed=0)

# %%
import monai
import numpy as np

root_dir ="./models"
if not os.path.exists(root_dir):
    os.makedirs(root_dir)



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
# %%
train_transform = Compose(
    [
        # load 4 Nifti images and stack them together
        LoadImaged(keys=["t1_image","t2_image",  "label"],allow_missing_keys=True),
        ConvertToGrayWhiteCSF(keys="label",allow_missing_keys=True),
        EnsureChannelFirstd(keys=["t1_image","t2_image"]),
        EnsureTyped(keys=["t1_image","t2_image", "label"],allow_missing_keys=True),
        Orientationd(keys=["t1_image","t2_image", "label"], axcodes="RAS",allow_missing_keys=True),
        Resized(keys=["t1_image","t2_image", "label"], spatial_size=[64, 64, 64], mode=("trilinear","trilinear", "nearest"),allow_missing_keys=True),
       # RandSpatialCropd(keys=["image", "label"], roi_size=[224, 224, 144], random_size=False),
        #RandFlipd(keys=["t1_image","t2_image", "label"], prob=0.5, spatial_axis=0,allow_missing_keys=True),
        #RandFlipd(keys=["t1_image","t2_image", "label"], prob=0.5, spatial_axis=1,allow_missing_keys=True),
        #RandFlipd(keys=["t1_image","t2_image", "label"], prob=0.5, spatial_axis=2,allow_missing_keys=True),
        #NormalizeIntensityd(keys=["t1_image","t2_image"], nonzero=True, channel_wise=True,allow_missing_keys=True),
        #RandScaleIntensityd(keys=["t1_image","t2_image"], factors=0.1, prob=1.0,allow_missing_keys=True),
        #RandShiftIntensityd(keys=["t1_image","t2_image"], offsets=0.1, prob=1.0,allow_missing_keys=True),
        #RandomConvexCombination()
    ]
)
val_transform = Compose(
    [
        LoadImaged(keys=["t1_image","t2_image", "label"],allow_missing_keys=True),
        ConvertToGrayWhiteCSF(keys="label",allow_missing_keys=True),
        EnsureChannelFirstd(keys=["t1_image","t2_image"],allow_missing_keys=True),
        EnsureTyped(keys=["t1_image","t2_image", "label"],allow_missing_keys=True),
        Orientationd(keys=["t1_image","t2_image", "label"], axcodes="RAS",allow_missing_keys=True),
        #Spacingd(
        #    keys=["image", "label"],
        #    pixdim=(2.0, 2.0, 2.0),
        #    mode=("bilinear", "nearest"),
        #),
        Resized(keys=["t1_image","t2_image", "label"], spatial_size=[64, 64, 64], mode=("trilinear", "trilinear","nearest"),allow_missing_keys=True),
        NormalizeIntensityd(keys=["t1_image","t2_image"], nonzero=True, channel_wise=True,allow_missing_keys=True),
    ]
)

# %%
# Load BOBs repository data
data_dir = '/home/ajoshi/project_ajoshi_27/BOBS_Repo/V1.0'
if not os.path.exists(data_dir):
    sub_data = '/project/ajoshi_27/BOBS_Repo/V1.0'

# Read tsv file
import pandas as pd
df = pd.read_csv(os.path.join(data_dir, 'participants.tsv'), sep='\t')
# Read the following comments from the tsv file : ID, Session, Age, Gestational_age_at_birth, Sex

# Read the following comments from the tsv file : ID, Session, Age, Gestational_age
df = pd.read_csv(os.path.join(data_dir, 'participants.tsv'), sep='\t')


# Display the first few rows of the dataframe
print(df.head())

t1_files = []
t2_files = []
label_files = []
ages = []
for idx, row in df.iterrows():
    t1file = os.path.join(data_dir, 'sub-'+str(row['ID']), 'ses-'+str(row['Session'])+'mo', 'anat', 'sub-'+str(row['ID'])+'_ses-'+str(row['Session'])+'mo'+'_space-INFANTMNIacpc'+'_T1w.nii.gz')

    if os.path.exists(t1file):
        t1_files.append(t1file)
    else:
        print('File not found: ', t1file)


    t2file = os.path.join(data_dir, 'sub-'+str(row['ID']), 'ses-'+str(row['Session'])+'mo', 'anat', 'sub-'+str(row['ID'])+'_ses-'+str(row['Session'])+'mo'+'_space-INFANTMNIacpc'+'_T2w.nii.gz')

    if os.path.exists(t2file):
        t2_files.append(t2file)
    else:
        print('File not found: ', t2file)

    labelfile = os.path.join(data_dir, 'sub-'+str(row['ID']), 'ses-'+str(row['Session'])+'mo', 'anat', 'sub-'+str(row['ID'])+'_ses-'+str(row['Session'])+'mo'+'_space-INFANTMNIacpc'+'_desc-aseg_dseg.nii.gz')

    if os.path.exists(labelfile):
        label_files.append(labelfile)
    else:
        print('File not found: ', labelfile)


    #t2_files.append(os.path.join(data_dir, 'sub-'+row['ID'], 'ses-'+row['Session'], 'anat', 'sub-'+row['ID']+'_ses-'+row['Session']+'_T2w.nii.gz'))
    #label_files.append(os.path.join(data_dir, 'sub-'+row['ID'], 'ses-'+row['Session'], 'anat', 'sub-'+row['ID']+'_ses-'+row['Session']+'_label.nii.gz'))
    ages.append(row['Age'])

# Display the first few rows of the dataframe
print(t1_files[:5])
print(t2_files[:5])
print(label_files[:5])
print(ages[:5])

# Load the data
data = [{"t1_image": t1, "t2_image": t2, "label": label, "age": age} for t1, t2, label, age in zip(t1_files, t2_files, label_files, ages)]
print(len(data))

# Split the data into training and validation sets
n_train = int(0.8 * len(data))
n_val = len(data) - n_train
train_data, val_data = torch.utils.data.random_split(data, [n_train, n_val])

# Create the training and validation datasets
train_ds = CacheDataset(data=train_data, transform=train_transform, cache_rate=1.0, num_workers=4)
val_ds = CacheDataset(data=val_data, transform=val_transform, cache_rate=1.0, num_workers=4)

# Create the training and validation data loaders
train_loader_bob = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=4)
val_loader_bob = DataLoader(val_ds, batch_size=2, num_workers=4)







# %% [markdown]
# ## Check data shape and visualize

# %%
for jj in range(5):
    val_data_example = train_ds[jj]
    print(f"image shape: {val_data_example['t1_image'].shape}")
    print(f"image shape: {val_data_example['t2_image'].shape}")
    #print(val_data_example.get('age', None))

    plt.figure("t1 and t2 images", (12, 3))
    plt.subplot(1, 5, 1)
    plt.title(f"t1 image ")
    plt.imshow(val_data_example["t1_image"][0, :, :, 32].detach().cpu(), cmap="gray")
    plt.subplot(1, 5, 2)
    plt.title(f"t2 image")
    plt.imshow(val_data_example["t2_image"][0, :, :, 32].detach().cpu(), cmap="gray")
    plt.subplot(1, 5, 3)
    plt.title(f"label image 0")
    plt.imshow(val_data_example["label"][0, :, :, 32].detach().cpu(), cmap="hot")
    plt.subplot(1, 5, 4)
    plt.title(f"label image 1")
    plt.imshow(val_data_example["label"][1, :, :, 32].detach().cpu(), cmap="hot")
    plt.subplot(1, 5, 5)
    plt.title(f"label image 2")
    plt.imshow(val_data_example["label"][2, :, :, 32].detach().cpu(), cmap="hot")

    plt.draw()
    plt.show()
    plt.pause(1)
    print(f"age: {val_data_example['age']}")






# %% [markdown]
# ## Create Model, Loss, Optimizer

# %%
max_epochs = 300
val_interval = 5
VAL_AMP = True

# standard PyTorch program style: create SegResNet, DiceLoss and Adam optimizer
device = torch.device("cuda:0")
model = SegResNetLatentOut(
    blocks_down=[1, 2, 2, 4],
    blocks_up=[1, 1, 1],
    init_filters=8,
    in_channels=1,
    out_channels=3,
    dropout_prob=0.2,
).to(device)


# at the latent layter, add a small neural network to predict brain age
#model.add_module("fc1", torch.nn.Linear(512, 1))


loss_function = DiceLoss(smooth_nr=0, smooth_dr=1e-5, squared_pred=True, to_onehot_y=False, sigmoid=True)
optimizer = torch.optim.Adam(model.parameters(), 1e-4, weight_decay=1e-5)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)

dice_metric = DiceMetric(include_background=True, reduction="mean")
dice_metric_batch = DiceMetric(include_background=True, reduction="mean_batch")

post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])


def model_seg(x):
    return model(x)[0]

# define inference method
def inference(input):
    def _compute(input):
        return sliding_window_inference(
            inputs=input,
            roi_size=(64, 64, 64),
            sw_batch_size=1,
            predictor=model_seg,
            overlap=0.5,
        )

    if VAL_AMP:
        with torch.amp.autocast('cuda'):
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
    for batch_data in train_loader_bob:
        step_start = time.time()
        step += 1
        inputs = batch_data["t1_image"].to(device)
        #print(inputs.shape)
        #print(inputs)
        labels = batch_data.get("label").to(device)
        #.to(device) #torch.zeros_like(inputs)
        age = batch_data.get("age").to(device)
        #print(f"age: {age}")
        #continue
        optimizer.zero_grad()


        with torch.amp.autocast('cuda'):
            #inputs.shape
            #model(inputs)
            outputs, out_age = model(inputs)
            loss = loss_function(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        epoch_loss += loss.item()
        print(
            f"{step}/{len(train_ds) // train_loader_bob.batch_size}"
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
            for val_data in val_loader_bob:
                val_inputs, val_labels = (
                    val_data["t1_image"].to(device),
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
                    os.path.join(root_dir, "best_metric_model.pth"),
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
plt.show()

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
plt.show()

# %% [markdown]
# ## Check best pytorch model output with the input image and label

# %%
model.load_state_dict(torch.load(os.path.join(root_dir, "best_metric_model.pth")))
model.eval()
with torch.no_grad():
    # select one image to evaluate and visualize the model output
    val_input = val_ds[6]["image"].unsqueeze(0).to(device)
    roi_size = (96, 96, 96)
    sw_batch_size = 4
    val_output = inference(val_input)
    print('shape of labels ',val_ds[6]["label"].shape)


    val_output = post_trans(val_output[0])
    plt.figure("image", (24, 6))
    for i in range(1):
        plt.subplot(1, 4, i + 1)
        plt.title(f"image channel {i}")
        plt.imshow(val_ds[6]["image"][i, :, :, 48].detach().cpu(), cmap="gray")
    plt.show()
    
    # visualize the 3 channels label corresponding to this image
    plt.figure("label", (18, 6))
    for i in range(3):
        plt.subplot(1, 3, i + 1)
        plt.title(f"label channel {i}")
        plt.imshow(val_ds[6]["label"][i, :, :, 48].detach().cpu())
    plt.show()
    # visualize the 3 channels model output corresponding to this image
    plt.figure("output", (18, 6))
    for i in range(3):
        plt.subplot(1, 3, i + 1)
        plt.title(f"output channel {i}")
        plt.imshow(val_output[i, :, :, 48].detach().cpu())
    plt.show()

# %% [markdown]
# ## Evaluation on original image spacings

# %%
val_transform = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        ConvertToGrayWhiteCSF(keys="label"),
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
model.load_state_dict(torch.load(os.path.join(root_dir, "best_metric_model.pth")))
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


