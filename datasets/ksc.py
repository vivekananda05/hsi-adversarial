"""KSC hyperspectral dataset."""
from dassl.data.datasets import DATASET_REGISTRY
from .hsi import HSIDataset


@DATASET_REGISTRY.register()
class KSC(HSIDataset):
    dataset_dir = "KSC"
