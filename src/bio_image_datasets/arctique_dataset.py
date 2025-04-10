import os
from functools import partial

import skimage

from bio_image_datasets.dataset import Dataset

# cell IDS from https://zenodo.org/records/14016860
mapping_dict = {
    0: "Background",
    1: "Epithelial",
    2: "Plasma Cells",
    3: "Lymphocytes",
    4: "Eosinophils",
    5: "Fibroblasts",
}


class ArctiqueDataset(Dataset):
    def __init__(self, local_path, do_resize=True):
        """Initializes the ArctiqueDataset with the given local path.

        The dataset is located on the /fast file system on the MDC cluster under the path
        '/fast/AG_Kainmueller/data/patho_foundation_model_bench_data/arctique_dataset/arctique'.
        Args:
            local_path (str): Path to the directory containing the files.
        """
        super().__init__(local_path)

        print(f"LOCAL PATH: {local_path}")

        self.images_folder = os.path.join(local_path, "images")
        self.semantic_masks_folder = os.path.join(local_path, "masks", "semantic")
        self.instance_masks_folder = os.path.join(local_path, "masks", "instance")
        self.sample_IDs = [int(name.split("_")[1].split(".")[0]) for name in os.listdir(self.images_folder)]
        self.do_resize = do_resize
        self.resize_func = partial(skimage.transform.resize, output_shape=(448, 448))

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.sample_IDs)

    def __getitem__(self, idx):
        """Return a sample as a dictionary at the given index.

        Args:
            idx (int): Index of the sample.

        Returns
        -------
            dict: A dictionary containing the following keys:
                - "image": Hematoxylin and eosin (HE) image
                - "semantic_mask": Ground truth semantic mask
                - "instance_mask": Ground truth instance mask
                - "sample_name": Index of the sample as string
        """
        if idx >= len(self):
            raise IndexError("Index out of bounds.")

        data = {
            "image": self.get_he(idx),
            "semantic_mask": self.get_semantic_mask(idx),
            "instance_mask": self.get_instance_mask(idx),
            "sample_name": self.get_sample_name(idx),
        }
        return data

    def get_he(self, idx):
        """Load the hematoxylin and eosin (HE) image for the given index.

        Args:
            idx (int): Index of the sample.

        Returns
        -------
            np.ndarray: The HE image.
        """
        sample_ID = self.get_sample_name(idx)
        img = skimage.io.imread(os.path.join(self.images_folder, f"img_{sample_ID}.png"))
        if self.do_resize:
            img = self.resize_func(img)
        return img.transpose()  # Transpose to have the channels first

    def get_class_mapping(self):
        """Return the class mapping for the dataset.

        Returns
        -------
            dict: A dictionary mapping class indices to class names.
        """
        return mapping_dict

    def get_instance_mask(self, idx):
        """Return the instance mask at the given index.

        Args:
            idx (int): Index of the sample.

        Returns
        -------
            np.ndarray: The instance mask.
        """
        sample_ID = self.get_sample_name(idx)
        instance_mask = skimage.io.imread(os.path.join(self.instance_masks_folder, f"{sample_ID}.png"))
        if self.do_resize:
            instance_mask = self.resize_func(instance_mask)
        return instance_mask

    def get_semantic_mask(self, idx):
        """Return the semantic mask at the given index.

        Args:
            idx (int): Index of the sample.

        Returns
        -------
            np.ndarray: The semantic mask.
        """
        sample_ID = self.get_sample_name(idx)
        semantic_mask = skimage.io.imread(os.path.join(self.semantic_masks_folder, f"{sample_ID}.png"))
        if self.do_resize:
            semantic_mask = self.resize_func(semantic_mask)
        return semantic_mask

    def get_sample_name(self, idx):
        """Return the sample name for the given index.

        Args:
            idx (int): Index of the sample.

        Returns
        -------
            str: The sample name, consisting of the fold and local index, e.g. fold1_0
        """
        sample_ID = self.sample_IDs[idx]
        return sample_ID

    def get_sample_names(self):
        """Return the list of all sample names."""
        return self.sample_IDs

    def __repr__(self):
        """Return the string representation of the dataset."""
        return f"Arctique Dataset ({self.local_path}, {len(self)} samples)"
