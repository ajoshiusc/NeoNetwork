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
    

