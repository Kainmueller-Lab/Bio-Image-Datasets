import os
from typing import Optional
import scipy.io as sio
import numpy as np
from PIL import Image
from bio_image_datasets.dataset import Dataset
import pandas as pd

class LizardDataset(Dataset):
    """Dataset class for the Lizard dataset."""

    def __init__(self, 
                 local_path: str = '~/projects/lab_hackathon_2024/Bio-Image-Datasets/downloads'):
        super().__init__(local_path)
        self.local_path = os.path.expanduser(local_path)
        self.image_paths = []
        self.label_paths = []
        self.sample_names = []
        self.sample_splits = []  # New list to store splits

        # Define image directories
        image_dirs = [
            os.path.join(self.local_path, 'lizard_images1', 'Lizard_Images1'),
            os.path.join(self.local_path, 'lizard_images2', 'Lizard_Images2')
        ]
        # Define label directory
        label_dir = os.path.join(self.local_path, 'lizard_labels', 'Lizard_Labels', 'Labels')
        # Define path to info.csv
        info_csv_path = os.path.join(self.local_path, 'lizard_labels', 'Lizard_Labels', 'info.csv')

        # Read info.csv
        if os.path.exists(info_csv_path):
            info_df = pd.read_csv(info_csv_path)
            # Create a mapping from Filename to Split
            filename_to_split = dict(zip(info_df['Filename'], info_df['Split']))
        else:
            print(f"Warning: info.csv not found at {info_csv_path}")
            filename_to_split = {}

        # Collect image and label file paths
        for image_dir in image_dirs:
            for file_name in os.listdir(image_dir):
                if file_name.endswith('.png'):
                    image_path = os.path.join(image_dir, file_name)
                    label_name = file_name.replace('.png', '.mat')
                    label_path = os.path.join(label_dir, label_name)
                    if os.path.exists(label_path):
                        sample_name = file_name.replace('.png', '')
                        self.image_paths.append(image_path)
                        self.label_paths.append(label_path)
                        self.sample_names.append(sample_name)
                        # Get split from mapping
                        split = filename_to_split.get(sample_name, None)
                        if split is None:
                            print(f"Warning: Split not found for sample {sample_name}")
                            self.sample_splits.append(None)
                        else:
                            self.sample_splits.append(split)
                    else:
                        print(f"Warning: Label file {label_name} not found for image {file_name}")

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.image_paths)

    def __getitem__(self, idx):
        """Return the image and its corresponding label at the given index."""
        image = self.get_he(idx)
        semantic_mask = self.get_semantic_mask(idx)
        instance_mask = self.get_instance_mask(idx)
        sample = {
            'image': image,
            'semantic_mask': semantic_mask,
            'instance_mask': instance_mask,
            'sample_name': self.get_sample_name(idx),
            'split': self.get_sample_split(idx)  # Added split information
        }
        return sample

    def get_he(self, idx):
        """Return the H&E-stained image at the given index."""
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        image = np.array(image)
        image = np.transpose(image, (2, 0, 1))  # HWC to CHW 
        return image

    def get_semantic_mask(self, idx):
        """Return the semantic segmentation mask at the given index."""
        label_data = self._load_label(idx)
        inst_map = label_data['inst_map']
        classes = np.atleast_1d(np.squeeze(label_data['class']))
        nuclei_id = np.atleast_1d(np.squeeze(label_data['id']))
        semantic_mask = np.zeros_like(inst_map, dtype=np.uint8)

        # Map instance labels to semantic labels
        for nucleus_id, class_id in zip(nuclei_id, classes):
            semantic_mask[inst_map == nucleus_id] = class_id

        return semantic_mask

    def get_instance_mask(self, idx):
        """Return the instance segmentation mask at the given index."""
        label_data = self._load_label(idx)
        inst_map = label_data['inst_map']
        return inst_map

    def get_sample_name(self, idx):
        """Return the sample name at the given index."""
        return self.sample_names[idx]

    def get_sample_names(self):
        """Return the list of sample names."""
        return self.sample_names

    def get_sample_split(self, idx):
        """Return the split value for the sample at the given index."""
        return self.sample_splits[idx]

    def get_sample_splits(self):
        """Return the list of sample splits."""
        return self.sample_splits

    def _load_label(self, idx):
        """Helper function to load label data from a .mat file."""
        label_path = self.label_paths[idx]
        label_data = sio.loadmat(label_path)
        return label_data

    def __repr__(self):
        """Return the string representation of the dataset."""
        return f"{self.__class__.__name__} ({self.local_path}) with {self.__len__()} samples"


if __name__ == '__main__':
    import os
    from matplotlib import pyplot as plt

    # Define the folder to save visualizations
    visualization_folder = './visualizations'
    os.makedirs(visualization_folder, exist_ok=True)

    # Instantiate the dataset with a sample transform
    dataset = LizardDataset(
        local_path='~/projects/lab_hackathon_2024/Bio-Image-Datasets/downloads',
    )

    # Print basic information about the dataset
    print(f"Number of samples in the dataset: {len(dataset)}")
    print(f"Sample names: {dataset.get_sample_names()[:5]}")  # Display first 5 sample names

    # Test each method for the first sample
    sample_idx = 111
    print(f"\nTesting methods for sample index: {sample_idx}")

    # Get the sample name
    sample_name = dataset.get_sample_name(sample_idx)
    print(f"Sample name: {sample_name}")

    # Load and save the H&E image
    image = dataset.get_he(sample_idx)
    plt.figure()
    # Transpose the image back to HWC for display
    image = np.transpose(image, (1, 2, 0))
    plt.imshow(image)
    plt.title(f"H&E Image - {sample_name}")
    plt.axis('off')
    he_image_path = os.path.join(visualization_folder, f'{sample_name}_he_image.png')
    plt.savefig(he_image_path)
    plt.close()
    print(f"H&E image saved to: {he_image_path}")

    # Load and save the semantic mask
    semantic_mask = dataset.get_semantic_mask(sample_idx)
    plt.figure()
    plt.imshow(semantic_mask, cmap='jet')
    plt.title(f"Semantic Mask - {sample_name}")
    plt.axis('off')
    semantic_mask_path = os.path.join(visualization_folder, f'{sample_name}_semantic_mask.png')
    plt.savefig(semantic_mask_path)
    plt.close()
    print(f"Semantic mask saved to: {semantic_mask_path}")

    # Load and save the instance mask
    instance_mask = dataset.get_instance_mask(sample_idx)
    plt.figure()
    plt.imshow(instance_mask, cmap='jet')
    plt.title(f"Instance Mask - {sample_name}")
    plt.axis('off')
    instance_mask_path = os.path.join(visualization_folder, f'{sample_name}_instance_mask.png')
    plt.savefig(instance_mask_path)
    plt.close()
    print(f"Instance mask saved to: {instance_mask_path}")

    # Load the full sample using __getitem__
    sample = dataset[sample_idx]
    print(f"Full sample: {sample_name}")
    print(f"Image shape: {sample['image'].shape}")
    print(f"Semantic mask shape: {sample['semantic_mask'].shape}")
    print(f"Instance mask shape: {sample['instance_mask'].shape}")

    # Save the full sample visualization
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    # Transpose the image back to HWC for display
    sample['image'] = np.transpose(sample['image'], (1, 2, 0))
    ax[0].imshow(sample['image'])
    ax[0].set_title("H&E Image")
    ax[0].axis('off')

    ax[1].imshow(sample['semantic_mask'], cmap='jet')
    ax[1].set_title("Semantic Mask")
    ax[1].axis('off')

    ax[2].imshow(sample['instance_mask'], cmap='jet')
    ax[2].set_title("Instance Mask")
    ax[2].axis('off')

    full_sample_path = os.path.join(visualization_folder, f'{sample_name}_full_sample.png')
    plt.suptitle(f"Sample Visualization - {sample_name}")
    plt.tight_layout()
    plt.savefig(full_sample_path)
    plt.close()
    print(f"Full sample visualization saved to: {full_sample_path}")
