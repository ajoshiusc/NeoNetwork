from monai.transforms import MapTransform
import torch


class ConvertToMultiChannelHeadRecod(MapTransform):
    """
    Convert labels to multi channels based on PVC classes:
    channel 1 is the CSF
    channel 2 is the WM
    channel 3 is the GM

    """

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            #result = []
            # if key does not exist in data, skip it
            if key not in d:
                continue
            result = list([d[key] == 1])
            result.append(d[key] == 2)
            result.append(torch.logical_or(d[key] == 3, d[key] == 8))
            d[key] = torch.stack(result, dim=0)
        return d
    


from typing import Dict, Sequence
import numpy as np
import nibabel as nib
from monai.transforms import MapTransform

class ConvertToGrayWhiteCSF(MapTransform):
    """
    A MONAI-compatible MapTransform to convert FreeSurfer aseg_dseg.nii.gz segmentation files
    into a 3-channel array representing Gray Matter (GM), White Matter (WM), and Cerebrospinal Fluid (CSF).
    """
    def __init__(self, keys: Sequence[str], label_mapping: Dict[str, Sequence[int]] = None, allow_missing_keys: bool = False):
        """
        Initializes the transform with keys to process and optional label mappings.

        Args:
            keys (Sequence[str]): The keys of the data to transform (e.g., ["label"]).
            label_mapping (dict): A dictionary mapping tissue types to their corresponding FreeSurfer labels.
                                  Default uses typical aseg labels for GM, WM, and CSF.
            allow_missing_keys (bool): Whether to allow missing keys in the input data dictionary.
        """
        super().__init__(keys, allow_missing_keys)
        # Default label mapping if none is provided
        self.label_mapping = label_mapping or {
            "GM": [3, 42, 8, 47, 11, 50, 17, 53, 18, 54],  # Example GM labels
            "WM": [2, 41, 46, 7, 251, 252, 253, 254, 255],  # Example WM labels
            "CSF": [0, 4, 43, 14, 15, 24],  # Example CSF labels, use 0=background as csf
        }

    def __call__(self, data: Dict):
        """
        Applies the transform to the specified keys in the data dictionary.

        Args:
            data (Dict): A dictionary containing the data to transform.

        Returns:
            Dict: The dictionary with transformed data for the specified keys.
        """
        d = dict(data)  # Make a copy of the input dictionary
        for key in self.keys:
            if key not in d:
                continue  # Skip missing keys if allow_missing_keys is True

            if isinstance(d[key], str):  # If the value is a file path, load it
                img = nib.load(d[key])
                array = img.get_fdata()
            else:  # Otherwise, assume it's a NumPy array
                array = np.asarray(d[key])

            # Initialize the output array (C, X, Y, Z)
            channels = len(self.label_mapping)
            segmented_array = np.zeros((channels, *array.shape), dtype=np.float32)

            # Populate the channels based on the label mapping
            for i, (key_name, labels) in enumerate(self.label_mapping.items()):
                segmented_array[i] = np.isin(array, labels).astype(np.float32)

            # Replace the data at the key with the transformed array
            d[key] = segmented_array

        return d

# Example usage with MONAI data loader:
# transform = ConvertToGrayWhiteCSF(keys=["label"])
# transformed_data = transform({"label": "aseg_dseg.nii.gz"})
# print(transformed_data["label"].shape)  # Expected: (3, X, Y, Z)
