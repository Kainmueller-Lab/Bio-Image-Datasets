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
        Initializes the PanNukeDataset with the given local path.
        The three PanNuke dataset is located on the /fast file system on the MDC cluster under the path
        '/fast/AG_Kainmueller/data/pannuke'.
        Args:
            local_path (str): Path to the directory containing the files.
        """
        super().__init__(local_path)

        print("LOCAL PATH", local_path)

        self.images_files= {}
        self.types_files = {}
        self.masks_files = {}

        self.images = {}
        self.types = {}
        self.masks = {}

        self.semantic_masks = {}
        self.instance_masks = {}

        folds = [1, 2, 3]
        for fold in folds:
            self.images_files[f'fold{fold}'] = os.path.join(local_path, f'fold{fold}/images/fold{fold}/images.npy')
            self.types_files[f'fold{fold}'] = os.path.join(local_path, f'fold{fold}/images/fold{fold}/types.npy')
            self.masks_files[f'fold{fold}'] = os.path.join(local_path, f'fold{fold}/masks/fold{fold}/masks.npy')

            # put last channel in images to first
            self.images[f'fold{fold}'] = np.moveaxis(np.load(self.images_files[f'fold{fold}']), -1, 1)
            self.types[f'fold{fold}'] = np.load(self.types_files[f'fold{fold}'])
            self.masks[f'fold{fold}'] = np.load(self.masks_files[f'fold{fold}'])

            self.semantic_masks[f'fold{fold}'] = self.prepare_semantic_masks(self.masks[f'fold{fold}'])
            self.instance_masks[f'fold{fold}'] = self.prepare_instance_masks(self.masks[f'fold{fold}'])


    def __len__(self):
        """Return the number of samples in the dataset."""
        return np.shape(self.types[f'fold1'])[0] + np.shape(self.types[f'fold2'])[0] + np.shape(self.types[f'fold3'])[0]

    def get_length_per_fold(self, fold):
        """Return the number of samples in the given fold.
        
        Args:
            fold (int): Number of the fold.
        Returns:
            int: Number of samples in the given fold.
        """
        return np.shape(self.types[f'fold{fold}'])[0]

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

        fold, local_idx = self.get_fold_and_local_index(idx)

        data = {
            "image": self.images[f'fold{fold}'][local_idx],
            "type": self.types[f'fold{fold}'][local_idx],
            "semantic_mask": self.semantic_masks[f'fold{fold}'][local_idx],
            "instance_mask": self.instance_masks[f'fold{fold}'][local_idx],
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
        fold, local_idx = self.get_fold_and_local_index(idx)
        return self.images[f'fold{fold}'][local_idx]

    def get_tissue_type(self, idx):
        """Load tissue type.
        
        Args:
            idx (int): Index of the sample.
        Returns:
            tissue type (string): The tissue type.
        """
        fold, local_idx = self.get_fold_and_local_index(idx)
        return self.types[f'fold{fold}'][local_idx]

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
        fold, local_idx = self.get_fold_and_local_index(idx)
        return self.instance_masks[f'fold{fold}'][local_idx]

    def get_semantic_mask(self, idx):
        """Return the semantic mask at the given index.
        
        Args:
            idx (int): Index of the sample.
        Returns:
            np.ndarray: The semantic mask.
        """
        fold, local_idx = self.get_fold_and_local_index(idx)
        return self.semantic_masks[f'fold{fold}'][local_idx]

    def get_sample_name(self, idx):
        """Return the sample name for the given index.
        
        Args:
            idx (int): Index of the sample.
        Returns:
            str: The sample name, consisting of the fold and local index, e.g. fold1_0
        """
        fold, local_idx = self.get_fold_and_local_index(idx)
        return f"fold{fold}_{local_idx}"

    def get_sample_names(self):
        """Return the list of all sample names."""
        return [self.get_sample_name(idx) for idx in range(int(len(self)))]

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
        instance_masks = instance_masks.astype(np.uint8)
        return instance_masks

    def get_fold_and_local_index(self, idx):
        """
        Determine the fold number and adjusted index within that fold for a given global index.

        Args:
            idx (int): Global index of the sample.

        Returns:
            tuple: A tuple containing the fold number and the adjusted index within that fold.

        Raises:
            IndexError: If the index is out of bounds for the dataset.
        """
        total_length = 0
        for fold in [1, 2, 3]:
            fold_length = np.shape(self.types[f'fold{fold}'])[0]
            if idx < total_length + fold_length:
                return fold, idx - total_length
            total_length += fold_length
        raise IndexError("Index out of bounds.")