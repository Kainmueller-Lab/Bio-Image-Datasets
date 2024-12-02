import os
import h5py
import numpy as np
from bio_image_datasets.dataset import Dataset


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
        return mapping_dict

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
            return np.squeeze(f["gt_inst"][:])

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
            return np.squeeze(f["gt_ct"][:])

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