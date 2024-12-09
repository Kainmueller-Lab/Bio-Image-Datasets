import os
import numpy as np
from bio_image_datasets.dataset import Dataset
from skimage.measure import label as relabel

mapping_dict = {
            0: "Background",
            1: "Epithelial",
            2: "Dead Cells",
            3: "Connective/Soft tissue cells",
            4: "Inflammatory",
            5: "Neoplastic cells",
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
        if 'fold1' in local_path:
            self.fold = 1
        elif 'fold2' in local_path:
            self.fold = 2
        elif 'fold3' in local_path:
            self.fold = 3
        else:
            raise ValueError("Invalid local path.")

        super().__init__(local_path)
        self.images_file = os.path.join(local_path, f'images/fold{self.fold}/images.npy')
        self.types_file = os.path.join(local_path, f'images/fold{self.fold}/types.npy')
        self.masks_file = os.path.join(local_path, f'masks/fold{self.fold}/masks.npy')

        if not self.images_file:
            raise ValueError("No images file found in the specified directory.")

        if not self.types_file:
            raise ValueError("No types file found in the specified directory.")

        if not self.masks_file:
            raise ValueError("No masks file found in the specified directory.")

        # put last channel in images to first
        self.images = np.load(self.images_file)
        self.images = np.moveaxis(self.images, -1, 1)
        self.types = np.load(self.types_file)
        self.masks = np.load(self.masks_file)

        self.semantic_masks = self.prepare_semantic_masks(self.masks)
        self.instance_masks = self.prepare_instance_masks(self.masks)


    def __len__(self):
        """Return the number of samples in the dataset."""
        return np.shape(self.types)[0]

    def __getitem__(self, idx):
        """Return a sample as a dictionary at the given index.
        
        Args:
            idx (int): Index of the sample.
        Returns:
            dict: A dictionary containing the following keys:
                - "image": Hematoxylin and eosin (HE) image
                - "type": Tissue type where the sample comes from
                - "semantic_mask": Ground truth semantic mask
                - "instance_mask": Ground truth instance mask
                - "sample_name": Index of the sample as string
        """
        if idx >= len(self):
            raise IndexError("Index out of bounds.")

        data = {
            "image": self.images[idx],
            "type": self.types[idx],
            "semantic_mask": self.semantic_masks[idx],
            "instance_mask": self.instance_masks[idx],
            'sample_name': self.get_sample_name(idx)
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
        return self.images[idx]

    def get_tissue_type(self, idx):
        """Load tissue type.
        
        Args:
            idx (int): Index of the sample.
        Returns:
            tissue type (string): The tissue type.
        """
        return self.types[idx]

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
        return self.instance_masks[idx]

    def get_semantic_mask(self, idx):
        """Return the semantic mask at the given index.
        
        Args:
            idx (int): Index of the sample.
        Returns:
            np.ndarray: The semantic mask.
        """
        return self.semantic_masks[idx]


    def get_sample_name(self, idx):
        """Return the sample name for the given index.
        
        Args:
            idx (int): Index of the sample.
        Returns:
            str: The sample name.
        """
        return f"fold{self.fold}_{str(idx)}"

    def get_sample_names(self):
        """Return the list of all sample names."""
        return [f"fold{self.fold}_{str(i)}" for i in range(int(len(self)))]

    def __repr__(self):
        """Return the string representation of the dataset."""
        return f"PanNuke Dataset ({self.local_path}, {len(self)} samples)"

    def prepare_semantic_masks(self, masks):
        """ Prepare the semantic segmentation mask based on the pannuke mask which contains each semantic class
        in an individual channel which need to be collapsed via argmax

        Args:
            masks (np.array): B x H x W x C 
        """
        # reverse order of last dim of masks to adhere to class mapping specified in mapping_dict
        masks = np.flip(masks, axis=-1)
        semantic_masks = np.argmax(masks, -1)
        return semantic_masks

    def prepare_instance_masks(self, masks):
        """ Prepare the instance segmentation mask based on the pannuke mask which contains each semantic class
        in an individual channel which need to be collapsed via max

        Args:
            masks (np.array): B x H x W x C 
        """
        # reverse order of last dim of masks to adhere to class mapping specified in mapping_dict
        instance_masks = np.max(masks[..., :-1], -1)
        # iterate over first dimension and use relabel on each individual tile
        for i in range(instance_masks.shape[0]):
            instance_masks[i] = relabel(instance_masks[i], background=0)
        return instance_masks