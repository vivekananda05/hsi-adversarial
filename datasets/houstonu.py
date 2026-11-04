"""HoustonU hyperspectral dataset."""
from dassl.data.datasets import DATASET_REGISTRY
from .hsi import HSIDataset


@DATASET_REGISTRY.register()
class HoustonU(HSIDataset):
    dataset_dir = "HoustonU"
