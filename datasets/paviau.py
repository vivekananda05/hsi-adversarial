"""PaviaU hyperspectral dataset."""
from dassl.data.datasets import DATASET_REGISTRY
from .hsi import HSIDataset


@DATASET_REGISTRY.register()
class PaviaU(HSIDataset):
    dataset_dir = "PaviaU"
