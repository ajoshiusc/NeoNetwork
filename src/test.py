import os
from monai.transforms import MapTransform, Compose, LoadImage, EnsureChannelFirst, LoadImaged
from monai.data import CacheDataset, DataLoader

# Custom transform to check if file exists
class CheckFileExistsd(MapTransform):
    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            # If the file doesn't exist, set the key to None
            if not os.path.exists(d.get(key, "")):
                d[key] = None
        return d

# Custom filter to remove entries with None
class FilterNoned(MapTransform):
    def __call__(self, data):
        return {k: v for k, v in data.items() if v is not None}

# Define your transformation pipeline
transforms = Compose([
    CheckFileExistsd(keys=["image", "label"]),  # First, check if files exist
    FilterNoned(keys=["image", "label"]),       # Filter out None values
    LoadImaged(keys=["image", "label"], image_only=True, ensure_channel_first=True,  allow_missing_keys=True),
    #EnsureChannelFirst(),
    # Add other transforms here as needed
])

# Example data dictionary with possible missing files
data_dicts = [
    {"image": "/home/ajoshi/project_ajoshi_27/zenodo_upload_v2/s0829/t1.nii.gz", "label": "/home/ajoshi/project_ajoshi_27/zenodo_upload_v2/s0829/t2.nii.gz"},
    {"image": "/home/ajoshi/project_ajoshi_27/zenodo_upload_v2/s0829/t2.nii.gz", "label": "/home/ajoshi/project_ajoshi_27/zenodo_upload_v2/s0829/t2.nii.gz"},
    {"image": "/home/ajoshi/project_ajoshi_27/zenodo_upload_v2/s0829/t2.nii.gz"},
]

# Create CacheDataset with the filtering pipeline
dataset = CacheDataset(data=data_dicts, transform=transforms, cache_rate=1.0)
dataloader = DataLoader(dataset, batch_size=2)

# Iterate through the data to verify
for batch in dataloader:
    print(batch)
