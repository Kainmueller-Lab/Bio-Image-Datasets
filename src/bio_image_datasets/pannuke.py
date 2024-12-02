import os
import h5py
import numpy as np
from bio_image_datasets.dataset import Dataset

mapping_dict = {
            0: "Neoplastic cells",
            1: "Inflammatory",
            2: "Connective/Soft tissue cells",
            3: "Dead Cells",
            4: "Epithelial",
            6: "Background",
        }


class PanNukeDataset(Dataset):
    def __init__(self, local_path):
        """
        Initializes the PanNukeDataset with the given local path and optional transform.
        The three PanNuke datasets are located on the /fast file system on the MDC cluster under the paths
        '/fast/AG_Kainmueller/data/pannuke/fold1', '/fast/AG_Kainmueller/data/pannuke/fold2', and '/fast/AG_Kainmueller/data/pannuke/fold3'.
        Args:
            local_path (str): Path to the directory containing the files.
        """
        # get fold number from local_path
        if local_path == '/fast/AG_Kainmueller/data/pannuke/fold1':
            fold = 1
        elif local_path == '/fast/AG_Kainmueller/data/pannuke/fold2':
            fold = 2
        elif local_path == '/fast/AG_Kainmueller/data/pannuke/fold3':
            fold = 3
        else:
            raise ValueError("Invalid local path.")

        super().__init__(local_path)
        self.images_file = os.path.join(local_path, f'/images/fold{fold}/images.npy')
        self.types_file = os.path.join(local_path, f'/images/fold{fold}/types.npy')
        self.masks_file = os.path.join(local_path, f'/masks/fold{fold}/masks.py')

        if not self.images_file:
            raise ValueError("No images file found in the specified directory.")

        if not self.types_file:
            raise ValueError("No types file found in the specified directory.")

        if not self.masks_file:
            raise ValueError("No masks file found in the specified directory.")


    def __len__(self):
        """Return the number of samples in the dataset."""
        types = np.load(self.types_file)
        return np.shape(types)[0]

    def __getitem__(self, idx):
        """Return a sample as a dictionary at the given index.
        
        Args:
            idx (int): Index of the sample.
        Returns:
            dict: A dictionary containing the following keys:
                - "gt_ct": Ground truth cell type mask.
                - "gt_inst": Ground truth instance mask.
                - "he_img": Hematoxylin and eosin (HE) image
        """
        if idx >= len(self):
            raise IndexError("Index out of bounds.")

        images = np.load(self.images_file)
        types = np.load(self.types_file)
        masks = np.load(self.masks_file)

        data = {
            "image": image[idx],
            "type": types,
            "mask": np.squeeze(f["ifl"][:]),
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
        pass

    def get_tissue_type(self, idx):
        """Load tissue type.
        
        Args:
            idx (int): Index of the sample.
        Returns:
            tissue type (string): The tissue type.
        """
        pass

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
        pass

    def get_semantic_mask(self, idx):
        """Return the semantic mask at the given index.
        
        Args:
            idx (int): Index of the sample.
        Returns:
            np.ndarray: The semantic mask.
        """
        pass

    def get_sample_name(self, idx):
        """Return the sample name for the given index.
        
        Args:
            idx (int): Index of the sample.
        Returns:
            str: The sample name.
        """
        pass

    def get_sample_names(self):
        """Return the list of all sample names."""
        pass

    def __repr__(self):
        """Return the string representation of the dataset."""
        pass