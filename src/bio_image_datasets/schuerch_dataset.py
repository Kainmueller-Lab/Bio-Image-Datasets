import os
import h5py
import numpy as np
from bio_image_datasets.dataset import Dataset
from copy import copy


mapping_dict = {
    0: "background",
    1: "B cells",
    2: "CD11b+ monocytes",
    3: "CD11b+CD68+ macrophages",
    4: "CD11c+ DCs",
    5: "CD163+ macrophages",
    6: "CD3+ T cells",
    7: "CD4+ T cells",
    8: "CD4+ T cells CD45RO+",
    9: "CD4+ T cells GATA3+",
    10: "CD68+ macrophages",
    11: "CD68+ macrophages GzmB+",
    12: "CD68+CD163+ macrophages",
    13: "CD8+ T cells",
    14: "NK cells",
    15: "Tregs",
    16: "adipocytes",
    17: "dirt",
    18: "granulocytes",
    19: "immune cells",
    20: "immune cells / vasculature",
    21: "lymphatics",
    22: "nerves",
    23: "plasma cells",
    24: "smooth muscle",
    25: "stroma",
    26: "tumor cells",
    27: "tumor cells / immune cells",
    28: "undefined",
    29: "vasculature",
}


coarse_mapping = {
    0: 'Background',
    1: 'B cells',
    3: 'Macrophages/Monocytes',
    4: 'Adipocytes',
    5: 'Dendritic cells',
    6: 'T cells',
    7: 'Granulocytes',
    8 : 'NK cells',
    9 : 'Nerves',
    10: 'Plasma cells',
    11: 'Smooth muscle',
    12: 'Stroma',
    13: 'Tumor cells',
    14: 'Vasculature/Lymphatics',
    15: 'Other cells',
}


ct_rename_dict = {
    "background": "Background",
    "B cells": "B cells",
    "CD11b+ monocytes": "Macrophages/Monocytes",
    "CD11b+CD68+ macrophages": "Macrophages/Monocytes",
    "CD11c+ DCs": "Dendritic cells",
    "CD163+ macrophages": "Macrophages/Monocytes",
    "CD3+ T cells": "T cells",
    "CD4+ T cells": "T cells",
    "CD4+ T cells CD45RO+": "T cells",
    "CD4+ T cells GATA3+": "T cells",
    "CD68+ macrophages": "Macrophages/Monocytes",
    "CD68+ macrophages GzmB+": "Macrophages/Monocytes",
    "CD68+CD163+ macrophages": "Macrophages/Monocytes",
    "CD8+ T cells": "T cells",
    "NK cells": "NK cells",
    "Tregs": "T cells",
    "adipocytes": "Adipocytes",
    "granulocytes": "Granulocytes",
    "immune cells": "Other cells",
    "immune cells / vasculature": "Other cells",
    "lymphatics": "Vasculature/Lymphatics",
    "nerves": "Nerves",
    "plasma cells": "Plasma cells",
    "smooth muscle": "Smooth muscle",
    "stroma": "Stroma",
    "tumor cells": "Tumor cells",
    "tumor cells / immune cells": "Other cells",
    "undefined": "Other cells",
    "vasculature": "Vasculature/Lymphatics",
}


coarse_mapping_reverse = {v: k for k, v in coarse_mapping.items()}
ct_rename_dict_new_class_names = {k: coarse_mapping_reverse[v] for k, v in ct_rename_dict.items()}
semantic_id_old_to_new = {
    k: ct_rename_dict_new_class_names[v] for k, v in mapping_dict.items() if v != "dirt"
}


def transform_semantic_mask(semantic_mask, mapping_dict):
    """Transform the semantic mask using the given mapping dictionary.
    
    Args:
        semantic_mask (np.ndarray): The semantic mask.
        mapping_dict (dict): A dictionary mapping class indices to class names.
    Returns:
        np.ndarray: The transformed semantic mask.
    """
    transformed_mask = np.zeros_like(semantic_mask)
    for k, v in mapping_dict.items():
        transformed_mask[semantic_mask == k] = v
    return transformed_mask


def exclude_classes(semantic_mask, exclude_classes, instance_mask=None):
    """Exclude classes from the data.
    
    Args:
        semantic_mask (np.ndarray): The semantic mask.
        exclude_classes (list): List of classes to exclude.
        instance_mask (np.ndarray): The instance mask.
    Returns:
        np.ndarray: The updated semantic mask.
        optional np.ndarray: The updated instance mask.
    """
    # Make a copy of the input masks
    semantic_mask = copy(semantic_mask)	
    if instance_mask is not None:
        instance_mask = copy(instance_mask)
    # Exclude classes
    for cls in exclude_classes:
        if instance_mask is not None:
            instance_mask[semantic_mask == cls] = 0
        semantic_mask[semantic_mask == cls] = 0
    if instance_mask is not None:
        return semantic_mask, instance_mask
    return semantic_mask


class SchuerchDataset(Dataset):
    def __init__(self, local_path):
        """
        Initializes the SchuerchDataset with the given local path and optional transform.
        The dataset is located on the /fast file system on the MDC cluster under this path
        '/fast/AG_Kainmueller/jrumber/data/SchuerchData/preprocessed'.
        Args:
            local_path (str): Path to the directory containing the HDF files.
        """
        super().__init__(local_path)
        self.files = [f for f in os.listdir(local_path)]
        if not self.files:
            raise ValueError("No HDF files found in the specified directory.")
        self.file_paths = [os.path.join(local_path, f) for f in self.files]
        self.transform_labels = lambda x: transform_semantic_mask(x, semantic_id_old_to_new)

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.files)

    def __getitem__(self, idx):
        """Return a sample as a dictionary at the given index.
        
        Args:
            idx (int): Index of the sample.
        Returns:
            dict: A dictionary containing the following keys:
                - "gt_ct": Ground truth cell type mask.
                - "gt_inst": Ground truth instance mask.
                - "immunoflourescence_img": Immunofluorescence image.
                - "he_img": Hematoxylin and eosin (HE) image
        """
        if idx >= len(self):
            raise IndexError("Index out of bounds.")
        file_path = self.file_paths[idx]
        with h5py.File(file_path, 'r') as f:
            data = {
                "gt_ct": np.squeeze(f["gt_ct"][:]),
                "gt_inst": np.squeeze(f["gt_inst"][:]),
                "immunoflourescence_img": np.squeeze(f["ifl"][:]),
                "he_img": np.squeeze(f["img"][:]),
            }
        return data

    def get_he(self, idx):
        """
        Load the hematoxylin and eosin (HE) image for the given index.
        Args:
            idx (int): Index of the sample.
        Returns:
            np.ndarray: The HE image.
        """
        if idx >= len(self):
            raise IndexError("Index out of bounds.")
        file_path = self.file_paths[idx]
        with h5py.File(file_path, 'r') as f:
            return np.squeeze(f["img"][:])

    def get_if(self, idx):
        """Load immunofluorescence (IFL) data for the given index.
        
        Args:
            idx (int): Index of the sample.
        Returns:
            np.ndarray: The immunofluorescence data.
        """
        if idx >= len(self):
            raise IndexError("Index out of bounds.")
        file_path = self.file_paths[idx]
        with h5py.File(file_path, 'r') as f:
            return np.squeeze(f["ifl"][:])

    def get_class_mapping(self):
        """Return the class mapping for the dataset.
        
        Returns:
            dict: A dictionary mapping class indices to class names.
        """
        return coarse_mapping

    def get_instance_mask(self, idx):
        """Return the instance mask at the given index.
        
        Args:
            idx (int): Index of the sample.
        Returns:
            np.ndarray: The instance mask.
        """
        if idx >= len(self):
            raise IndexError("Index out of bounds.")
        file_path = self.file_paths[idx]
        with h5py.File(file_path, 'r') as f:
            instance_mask = np.squeeze(f["gt_inst"][:])
            semantic_mask = np.squeeze(f["gt_ct"][:])
            _, instance_mask = exclude_classes(
                semantic_mask=semantic_mask, exclude_classes=[17], instance_mask=instance_mask
            )
        return instance_mask

    def get_semantic_mask(self, idx):
        """Return the semantic mask at the given index.
        
        Args:
            idx (int): Index of the sample.
        Returns:
            np.ndarray: The semantic mask.
        """
        if idx >= len(self):
            raise IndexError("Index out of bounds.")
        file_path = self.file_paths[idx]
        with h5py.File(file_path, 'r') as f:
            semantic_mask = np.squeeze(f["gt_ct"][:])
            semantic_mask = exclude_classes(semantic_mask, [17])
        return self.transform_labels(semantic_mask)

    def get_sample_name(self, idx):
        """Return the sample name for the given index.
        
        Args:
            idx (int): Index of the sample.
        Returns:
            str: The sample name.
        """
        if idx >= len(self):
            raise IndexError("Index out of bounds.")
        return self.files[idx]

    def get_sample_names(self):
        """Return the list of all sample names."""
        return self.files

    def __repr__(self):
        """Return the string representation of the dataset."""
        return f"SchuerchDataset ({self.local_path}, {len(self)} samples)"
