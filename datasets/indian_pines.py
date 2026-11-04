"""Indian_pines hyperspectral dataset."""
from dassl.data.datasets import DATASET_REGISTRY
from .hsi import HSIDataset


@DATASET_REGISTRY.register()
class Indian_pines(HSIDataset):
    dataset_dir = "Indian_pines"
