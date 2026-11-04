"""PaviaC hyperspectral dataset."""
from dassl.data.datasets import DATASET_REGISTRY
from .hsi import HSIDataset


@DATASET_REGISTRY.register()
class PaviaC(HSIDataset):
    dataset_dir = "PaviaC"
