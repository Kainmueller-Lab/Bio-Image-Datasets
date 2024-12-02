from abc import ABC, abstractmethod
from typing import Callable, Optional


class Dataset(ABC):
    """Abstract class to define the structure of a dataset.

    Args:
        local_path (str): Path to the dataset.
        transform (Optional[Callable]): Transform to apply to the data.
    """
    def __init__(self, local_path: str, transform: Optional[Callable] = None):
        self.local_path = local_path
        self.transform = transform

    @abstractmethod
    def __len__(self):
        """Return the length / number of samples of the dataset."""
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, idx):
        """Return the item at the given index."""
        raise NotImplementedError
    
    @abstractmethod
    def get_he(self, idx):
        """Return the he at the given index."""
        raise NotImplementedError

    @abstractmethod
    def get_semantic_mask(self, idx):
        """Return the semantic mask at the given index."""
        raise NotImplementedError

    def get_instance_mask(self, idx):
        """Return the instance mask at the given index."""
        raise NotImplementedError

    @abstractmethod
    def get_sample_name(self, idx):
        """Return the sample name at the given index."""
        raise NotImplementedError

    @abstractmethod
    def get_sample_names(self):
        """Return the list of sample names."""
        raise NotImplementedError

    @abstractmethod
    def __repr__(self):
        """Return the string representation of the dataset."""
        return self.__class__.__name__ + ' (' + self.local_path + ')'
