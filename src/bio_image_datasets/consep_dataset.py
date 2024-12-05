import os
import scipy.io as sio
import numpy as np
from PIL import Image
from pathlib import Path
import warnings
from bio_image_datasets.dataset import Dataset

mapping_dict = {
            1: "Normal Epithelial",
            2: "Malignant/dysplastic epithelial",
            3: "Fibroblast",
            4: "Muscle",
            5: "Inflammatory",
            6: "Endothelial",
            7: "Miscellaneous"
        }


class ConSePDataset(Dataset): 
    """Dataset class for the ConSeP dataset. 

    The ConSeP dataset is introduced in the HoverNet-paper (https://arxiv.org/pdf/1812.06499). 
    The dataset consists of 41 H&E stained image tiles, each of size 1,000×1,000 pixels at 40× 
    objective magnification. Images were extracted from 16 colorectal adenocarcinoma (CRA) WSIs, 
    each belonging to an individual patient, and scanned with an Omnyx VL120 scanner. Cell types 
    present are: normal epithelial, malignant/dysplastic epithelial, fibroblast, muscle, inflammatory,
    endothelial or miscellaneous. 
    
    The original dataset is linked at: https://warwick.ac.uk/fac/sci/dcs/research/tia/data/hovernet/. 
    At the time of writing, the (tiled) dataset is available at:
    https://www.kaggle.com/datasets/rftexas/tiled-consep-224x224px/code
    
    """

    def __init__(self, 
                 local_path): 
        """ Initialize the ConSeP dataset.

        Args:
            local_path (str): Path to the directory containing the dataset.
        Returns:
            ConSePDataset: An instance of the ConSePDataset.
        """

        self.local_path = Path(local_path)
        self.image_path = self.local_path.joinpath("tiles")
        self.label_path = self.local_path.joinpath("labels")
        
        self.image_files = os.listdir(self.image_path)
        self.label_files = os.listdir(self.label_path)

    def __len__(self):
        """Return the length / number of samples of the dataset.
        
        Args:
            None
        Returns:
            int: Number of samples in the dataset.
        """
        return len(self.image_files)

    def __getitem__(self, idx):
        """Return the image and its corresponding label at the given index.
        
        Args:
            idx (int): Index of the sample to return.
        
        Returns:
            dict: A dictionary containing the following keys: 
                image: H&E-stained image, 
                semantic mask: semantic segmentation mask, 
                instance mask: instance segmentation mask, 
                sample_name: name of the sample.    
            
        """
        image = self.get_he(idx)
        semantic_mask = self.get_semantic_mask(idx)
        instance_mask = self.get_instance_mask(idx)
        sample = {
            'image': image,
            'semantic_mask': semantic_mask,
            'instance_mask': instance_mask,
            'sample_name': self.get_sample_name(idx)
        }
        return sample
    
    def get_he(self, idx):
        """Return the he at the given index.
        
        Args:
            idx (int): Index of the sample to return.
        Returns:
            np.array: H&E-stained image.
        """
        img_array = np.array(Image.open(self.image_path.joinpath(self.image_files[idx])))
        return  np.transpose(img_array, (2, 0, 1)) # HWC to CHW 
    
    def get_semantic_mask(self, idx):
        """Return the semantic mask at the given index.
        
        Args:
            idx (int): Index of the sample to return.
        Returns:
            np.array: Semantic segmentation mask.
        """
        label_dict = sio.loadmat(self.label_path.joinpath(self.label_files[idx]))

        instance_mask = label_dict["instance_map"]
        class_ids = label_dict["class_labels"][0]

        semantic_mask = np.zeros(instance_mask.shape, dtype=np.uint8)

        semantic_dict = {}
        for instance_id, class_id in enumerate(class_ids):
            if class_id not in semantic_dict.keys():
                semantic_dict[class_id] = []
                semantic_dict[class_id].append(instance_id)

        for class_id, instance_ids in semantic_dict.items():
            semantic_mask[np.isin(instance_mask, instance_ids)] = class_id 

        return semantic_mask

    def get_instance_mask(self, idx):
        """Return the instance mask at the given index.
        
        Args:
            idx (int): Index of the sample to return.
        Returns:
            np.array: Instance segmentation mask.
        """
        label_dict = sio.loadmat(self.label_path.joinpath(self.label_files[idx]))
        return label_dict["instance_map"]

    def get_sample_name(self, idx):
        """Return the sample name at the given index.
        
        Args:
            idx (int): Index of the sample to return.
        Returns:
            str: Name of the sample.
        """
        return self.image_files[idx]

    def get_sample_names(self):
        """Return the list of sample names.
        
        Args:
            None
        Returns:
            list: List of sample names.
        """
        return self.image_files
    
    def get_class_mapping(self):
        """Return the class mapping for the dataset.
        
        Returns:
            dict: A dictionary mapping class indices to class names.
        """
        warnings.warn("The class mappings might be incorrect")
        return mapping_dict

    def __repr__(self):
        """Return the string representation of the dataset.
        
        Args:
            None
        Returns:
            str: String representation of the dataset.
        """
        return self.__class__.__name__ + ' (' + self.local_path + ')'
        



